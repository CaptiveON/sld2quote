"""RT-DETR symbol detector: fine-tune on the synthetic COCO tiles, evaluate any split, time whole sheets on cpu and mps.

  python3 -m sld2quote.train_detector train --data data/coco --out weights/rtdetr_r18 --model PekingU/rtdetr_r18vd --epochs 3
  python3 -m sld2quote.train_detector eval --weights weights/rtdetr_r18 --data data/coco --split test_unseen
  python3 -m sld2quote.train_detector bench --models PekingU/rtdetr_r18vd,PekingU/rtdetr_r50vd --device cpu,mps
  python3 -m sld2quote.train_detector summary --runs weights/rtdetr_r18_full,weights/rtdetr_r50_full --out weights/overnight_summary.md

The export at data/coco holds manifest.json and <split>/annotations.json with <split>/images/*.png: grey 640 px tiles, COCO boxes,
0-based category ids in classes.CLASSES order. detect() is the one inference path (the processor's detections, then class-agnostic
NMS); evaluation reports every split with NMS and raw. Precision and recall match greedily by score, as evaluate.py does, at
IoU >= 0.5; mAP comes from pycocotools. The data is synthetic: every tile is cut from a sheet drawn by generate.py."""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.metadata as md, io, json, logging, math, os, platform, random, shlex, shutil, statistics, subprocess, sys, time
from collections import Counter, defaultdict
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision import ops
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, get_cosine_schedule_with_warmup, set_seed
from transformers.utils import logging as hf_logging
from .classes import CLASSES, CLASS_ID
from .evaluate import iou
from .tiles import cut_tile, tile_grid

log = logging.getLogger("sld2quote.train_detector")
MATCH_IOU = 0.5                                                # a detection finds a box of its class when they overlap this much
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)         # operating thresholds tried on val
EVAL_SPLITS = ("val", "test_clean", "test_scan", "test_unseen")
STYLE_KEYS = ("copy", "breaker", "level", "font", "label_pos")  # image keys the export adds, for recall by style
EDGE_PX = 2                                                    # a detection this close to the tile border touches the edge
VAL_KEYS = ("map50", "map", "mean_precision", "mean_recall", "min_precision", "min_recall")  # kept per epoch in history.json
LABEL = {"nms": "with NMS", "raw": "raw"}
PLAIN = {"breaker": "breakers", "fuse_switch": "fuse switches", "isolator": "isolators", "contactor": "contactors", "transformer": "transformers",
         "generator": "generators", "motor": "motors", "board": "downstream boards", "ct": "current transformers", "meter": "meters",
         "relay": "protection relays", "spd": "surge arresters", "capacitor": "capacitors", "earth": "earth symbols"}  # for the owner's summary


def r4(v):
    return None if v is None else round(float(v), 4)


def fmt(v) -> str:
    return "-" if v is None else str(v)


# data

class Tiles(Dataset):
    """One split of the COCO export, read one tile at a time. aug = (brightness, contrast, scale, min_visible) for training."""

    def __init__(self, root: str, split: str, proc, aug: tuple | None = None, limit: int | None = None, seed: int = 0):
        js = json.load(open(os.path.join(root, split, "annotations.json")))
        if {c["id"]: c["name"] for c in js["categories"]} != dict(enumerate(CLASSES)):
            raise ValueError(f"{root}/{split}: categories do not match sld2quote/classes.py")
        self.dir, self.proc, self.aug, self.seed, self.epoch = os.path.join(root, split, "images"), proc, aug, seed, 0
        self.images, self.categories = sorted(js["images"], key=lambda im: im["id"])[:limit], js["categories"]
        keep, self.anns = {im["id"] for im in self.images}, defaultdict(list)
        for an in js["annotations"]:
            if an["image_id"] in keep:
                self.anns[an["image_id"]].append(an)

    def __len__(self):
        return len(self.images)

    def annotations(self) -> list:
        return [an for im in self.images for an in self.anns[im["id"]]]

    def __getitem__(self, i):
        im = self.images[i]; anns = self.anns[im["id"]]
        img = Image.open(os.path.join(self.dir, os.path.basename(im["file_name"]))).convert("RGB")
        if self.aug:  # seeded by tile and epoch, so a run repeats whatever the number of workers and survives a resume
            img, anns = jitter(img, anns, random.Random(f"{self.seed}:{self.epoch}:{i}"), *self.aug)
        enc = self.proc(images=img, annotations={"image_id": im["id"], "annotations": anns}, return_tensors="pt")
        return enc["pixel_values"][0], enc["labels"][0]


def jitter(img, anns, rng, brightness: float, contrast: float, scale: float, min_visible: float):
    """light augmentation for line drawings: a small zoom about the tile centre, then brightness and contrast. Never a flip,
    since mirrored text is unrealistic. A box the zoom pushes past the edge is clipped, and dropped below min_visible of its symbol."""
    W, H = img.size; s = rng.uniform(1 - scale, 1 + scale); w, h = round(W * s), round(H * s); ox, oy = (W - w) // 2, (H - h) // 2
    out = Image.new("RGB", (W, H), (255, 255, 255)); out.paste(img.resize((w, h), Image.BILINEAR), (ox, oy))
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(1 - brightness, 1 + brightness))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(1 - contrast, 1 + contrast))
    kept = []
    for an in anns:
        x, y, bw, bh = an["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        x0, y0, x1, y1 = x * w / W + ox, y * h / H + oy, (x + bw) * w / W + ox, (y + bh) * h / H + oy
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if cx1 > cx0 and cy1 > cy0 and an.get("visible_frac", 1.0) * (cx1 - cx0) * (cy1 - cy0) / ((x1 - x0) * (y1 - y0)) >= min_visible:
            kept.append({**an, "bbox": [cx0, cy0, cx1 - cx0, cy1 - cy0], "area": (cx1 - cx0) * (cy1 - cy0)})
    return out, kept


def collate(batch):
    return {"pixel_values": torch.stack([b[0] for b in batch]), "labels": [b[1] for b in batch]}


def loader(ds: Tiles, a, train: bool) -> DataLoader:
    """workers come from a fork server: under spawn, on this Mac (torch 2.5.1, macOS 27), every pass ended with a 5 s join
    timeout per worker although the workers had exited, 20 s per split or epoch with 4 workers"""
    return DataLoader(ds, batch_size=a.batch, shuffle=train, drop_last=train, num_workers=a.workers, collate_fn=collate,
                      multiprocessing_context="forkserver" if a.workers else None, generator=torch.Generator().manual_seed(ds.seed + ds.epoch))


# model

def load_model(name: str, **kw):
    """RT-DETR with the 14-class head: new on a hub checkpoint (every pretrained tensor kept, the class heads re-initialised),
    as trained on a weights directory"""
    return RTDetrForObjectDetection.from_pretrained(name, num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)), label2id=CLASS_ID,
                                                    ignore_mismatched_sizes=True, **kw)


def load_processor(name: str, **kw):
    """tiles are already 640 px, so the processor only rescales to 0..1 and formats the labels"""
    return RTDetrImageProcessor.from_pretrained(name, do_resize=False, **kw)


def optimiser(model, lr: float, backbone_lr: float, weight_decay: float):
    """AdamW in two groups: the pretrained backbone at its own, lower rate first, then the rest"""
    groups = [[], []]
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups[not n.startswith("model.backbone.")].append(p)
    return torch.optim.AdamW([{"params": groups[0], "lr": backbone_lr}, {"params": groups[1], "lr": lr}], weight_decay=weight_decay)


def step(model, opt, sched, batch: dict, device: str, grad_clip: float) -> float:
    """one optimisation step; a non-finite loss stops the run before it can reach a checkpoint"""
    out = model(pixel_values=batch["pixel_values"].to(device), labels=[{k: v.to(device) for k, v in l.items()} for l in batch["labels"]])
    loss = out.loss.item()
    if not math.isfinite(loss):
        raise FloatingPointError(f"loss became {loss}; the last checkpoint is untouched")
    opt.zero_grad(set_to_none=True); out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip); opt.step(); sched.step()
    return loss


def nms(r: dict, iou_thr: float) -> dict:
    """class-agnostic non-maximum suppression on one tile's detections: of boxes that overlap by more than iou_thr only the
    best-scoring survives, whatever its class, since a symbol is one thing and a second box or label on it is a false positive"""
    keep = ops.nms(r["boxes"].float().cpu(), r["scores"].float().cpu(), iou_thr)
    return {k: v.cpu()[keep] for k, v in r.items()}


def detect(model, proc, pixel_values, sizes: list, floor: float, nms_iou: float) -> list[dict]:
    """the detector's one inference path, for evaluation, the bench and the reader: per tile the processor's detections at or
    above floor (scores, labels and boxes x0, y0, x1, y1 in tile pixels), then class-agnostic NMS at nms_iou (0 disables it).
    A box below a threshold never suppresses one above it, so a floor at the operating threshold keeps exactly what evaluation counts."""
    res = proc.post_process_object_detection(model(pixel_values=pixel_values), threshold=floor, target_sizes=sizes)
    return [nms(r, nms_iou) for r in res] if nms_iou else res


def variants(a) -> list:
    """the post-processing reported: with NMS first, the operating one, when --nms-iou is set; and raw"""
    return (["nms"] if a.nms_iou else []) + ["raw"]


def coco_dets(image_id: int, r: dict) -> list:
    return [{"image_id": image_id, "category_id": c, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s}
            for s, c, (x0, y0, x1, y1) in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist())]


@torch.no_grad()
def predict(model, proc, ds: Tiles, a) -> tuple[dict, float]:
    """per variant, detections at or above the score floor in COCO form (tile pixels): raw as the processor gives them, nms as
    detect() gives them; and images per second including loading"""
    model.eval(); dets = {v: [] for v in variants(a)}; t0 = time.time()
    for b in loader(ds, a, train=False):
        raw = detect(model, proc, b["pixel_values"].to(a.device), [tuple(l["orig_size"].tolist()) for l in b["labels"]], a.score_floor, 0)
        for l, r in zip(b["labels"], raw):
            for v in dets:
                dets[v] += coco_dets(int(l["image_id"]), nms(r, a.nms_iou) if v == "nms" else r)
    return dets, len(ds) / max(time.time() - t0, 1e-9)


# metrics

def match(anns: list, dets: list, iou_thr: float = MATCH_IOU) -> tuple[np.ndarray, np.ndarray]:
    """greedy by score: each detection takes the unmatched box of its class on its tile that it overlaps most, if that reaches
    iou_thr. Returns per box the score of the detection that took it (-1 if none), and per detection whether it took one. The
    matching runs in score order, so at a threshold t the boxes found are exactly those whose score is >= t, and one pass
    serves every threshold."""
    free = defaultdict(list)
    for i, g in enumerate(anns):
        free[g["image_id"], g["category_id"]].append(i)
    got, hit = np.full(len(anns), -1.0), np.zeros(len(dets), bool)
    for j in sorted(range(len(dets)), key=lambda j: -dets[j]["score"]):
        d = dets[j]; cand = free.get((d["image_id"], d["category_id"]))
        if cand:
            v, i = max((iou(anns[i]["bbox"], d["bbox"]), i) for i in cand)
            if v >= iou_thr:
                got[i], hit[j] = d["score"], True; cand.remove(i)
    return got, hit


def scored(anns: list, dets: list):
    """what precision, recall and the threshold choice need: class and matched score per box, class, score and hit per detection"""
    got, hit = match(anns, dets)
    return SimpleNamespace(anns=anns, dets=dets, got=got, hit=hit, gt_cls=np.array([g["category_id"] for g in anns], int),
                           det_cls=np.array([d["category_id"] for d in dets], int), det_score=np.array([d["score"] for d in dets], float))


def pr(sc, thresholds) -> tuple:
    """per-class precision, recall and F1 at each threshold (classes x thresholds), and the counts behind them. A ratio with
    nothing to count is 0 here; split_metrics reports it as undefined."""
    def reach(cls, s):  # per class, how many scores are at or above each threshold
        return np.stack([len(v) - np.searchsorted(v, thresholds) for v in (np.sort(s[cls == c]) for c in range(len(CLASSES)))])
    tp, npred = reach(sc.gt_cls, sc.got), reach(sc.det_cls, sc.det_score)
    ngt = np.bincount(sc.gt_cls, minlength=len(CLASSES))[:, None]
    return tp / np.maximum(npred, 1), tp / np.maximum(ngt, 1), 2 * tp / np.maximum(npred + ngt, 1), tp, npred, ngt


def choose_threshold(sc) -> float:
    """the threshold with the highest mean F1 over the classes present; a tie goes to the middle of the tied thresholds"""
    f1, ngt = pr(sc, THRESHOLDS)[2], np.bincount(sc.gt_cls, minlength=len(CLASSES))
    if not ngt.any():
        raise ValueError("no ground-truth boxes to choose a threshold on")
    m = f1[ngt > 0].mean(0); best = np.flatnonzero(m >= m.max() - 1e-9)
    return float(THRESHOLDS[best[len(best) // 2]])


def recall_by(anns: list, found, key) -> dict:
    """recall over the boxes grouped by key(box); boxes whose key is None are left out"""
    n, hit = Counter(), Counter()
    for g, f in zip(anns, found):
        k = key(g)
        if k is not None:
            n[k] += 1; hit[k] += bool(f)
    return {k: r4(hit[k] / n[k]) for k in sorted(n, key=str)}


def coco_map(ds: Tiles, dets: list) -> tuple:
    """(mAP@0.5, mAP@0.5:0.95) from pycocotools over the split's tiles; None when the split has no box to find"""
    anns = ds.annotations()
    if not anns:
        return None, None
    if not dets:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = {"images": ds.images, "annotations": anns, "categories": ds.categories}; gt.createIndex()
        ev = COCOeval(gt, gt.loadRes([dict(d) for d in dets]), "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])


def at_edge(box: list, im: dict) -> bool:
    x, y, w, h = box
    return min(x, y) <= EDGE_PX or x + w >= im["width"] - EDGE_PX or y + h >= im["height"] - EDGE_PX


def split_metrics(ds: Tiles, dets: list, images_per_s: float, sc, t: float) -> dict:
    """mAP, then per-class precision and recall at threshold t (None where there is nothing to count: no detections for
    precision, no boxes for recall), recall by copy, style and visibility, and the false positives that touch a tile edge,
    where a symbol cut below the export's min_visible is unlabelled background"""
    p, r, f1, tp, npred, ngt = (x[:, 0] for x in pr(sc, [t]))
    map50, map_ = coco_map(ds, dets); found = sc.got >= t; img = {im["id"]: im for im in ds.images}
    hp, hr = npred > 0, ngt > 0  # precision is defined where something was detected, recall where there is a box
    stat = lambda f, v, m: r4(f(v[m])) if m.any() else None
    fp = [d for d, s, h in zip(sc.dets, sc.det_score, sc.hit) if s >= t and not h]
    by = {k: recall_by(sc.anns, found, lambda g, k=k: img[g["image_id"]].get(k)) for k in STYLE_KEYS}
    by["visible"] = recall_by(sc.anns, found, lambda g: "partial" if g.get("visible_frac", 1.0) < 1 else "whole")
    brk = sc.gt_cls == CLASS_ID["breaker"]
    return {"images": len(ds), "boxes": len(sc.anns), "map50": r4(map50), "map": r4(map_), "images_per_s": r4(images_per_s),
            "mean_precision": stat(np.mean, p, hp), "mean_recall": stat(np.mean, r, hr), "mean_f1": stat(np.mean, f1, hr),
            "min_precision": stat(np.min, p, hp), "min_recall": stat(np.min, r, hr),
            "per_class": {n: {"gt": int(ngt[c]), "pred": int(npred[c]), "tp": int(tp[c]), "precision": r4(p[c]) if hp[c] else None,
                              "recall": r4(r[c]) if hr[c] else None, "f1": r4(f1[c]) if hp[c] or hr[c] else None} for c, n in enumerate(CLASSES)},
            "recall_by": {k: v for k, v in by.items() if v},
            "breaker_recall_by_style": recall_by([g for g, b in zip(sc.anns, brk) if b], found[brk], lambda g: img[g["image_id"]].get("breaker")),
            "false_positives": {"count": len(fp), "at_tile_edge": sum(at_edge(d["bbox"], img[d["image_id"]]) for d in fp)}}


def evaluate(model, proc, a, splits: list, ts: dict | None = None) -> tuple[dict, dict]:
    """metrics per variant and split, each variant at its own operating threshold: ts[variant] if given, otherwise the one that
    maximises mean F1 on val under that variant's post-processing"""
    if not ts and "val" not in splits:
        raise ValueError("the operating thresholds are chosen on val, which is not among the splits")
    runs = {}
    for s in splits:
        ds = Tiles(a.data, s, proc, limit=a.limit); dets, ips = predict(model, proc, ds, a)
        runs[s] = (ds, ips, {v: (d, scored(ds.annotations(), d)) for v, d in dets.items()})
    ts = ts or {v: choose_threshold(runs["val"][2][v][1]) for v in variants(a)}
    out = {v: {} for v in variants(a)}
    for s, (ds, ips, vs) in runs.items():
        for v, (d, sc) in vs.items():
            m = out[v][s] = split_metrics(ds, d, ips, sc, ts[v])
            log.info(f"{s} {LABEL[v]}: {m['images']} tiles, {m['boxes']} boxes; mAP@0.5 {m['map50']}, mAP@0.5:0.95 {m['map']}; at threshold {ts[v]:.2f}: "
                     f"mean P {m['mean_precision']}, mean R {m['mean_recall']}, min P {m['min_precision']}, min R {m['min_recall']}; "
                     f"{m['false_positives']['count']} false positives, {m['false_positives']['at_tile_edge']} at a tile edge; {m['images_per_s']:.1f} images/s")
    return ts, out


# train

def setup_log(path: str):
    """log to the run's file and to stdout, including Python warnings (the MPS fall-backs) and transformers' own messages"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True,
                        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)])
    logging.captureWarnings(True); hf_logging.disable_default_handler(); hf_logging.enable_propagation()
    log.info(f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')}")


def command(argv: list) -> str:
    env = f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ['PYTORCH_ENABLE_MPS_FALLBACK']} " if "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ else ""
    return env + "python3 -m sld2quote.train_detector " + shlex.join(argv)


def versions() -> dict:
    return {"python": platform.python_version(), **{m: md.version(m) for m in ("torch", "torchvision", "transformers", "pycocotools", "numpy", "Pillow")}}


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def train_epoch(model, opt, sched, ds: Tiles, a, epoch: int, hist: dict) -> tuple[float, float]:
    """one pass over the train split; logs the loss and images per second every --log-every steps; returns mean loss, images/s"""
    set_seed(a.seed + epoch); ds.epoch = epoch; model.train()
    dl = loader(ds, a, train=True); n = len(dl); total, win, seen, t_ep = 0.0, [], 0, time.time(); t0 = t_ep
    for i, b in enumerate(dl, 1):
        win.append(step(model, opt, sched, b, a.device, a.grad_clip)); seen += len(b["labels"])
        if i % a.log_every == 0 or i == n:
            ips = seen / (time.time() - t0); total += sum(win)
            rec = {"epoch": epoch + 1, "step": epoch * n + i, "loss": r4(statistics.mean(win)), "lr": opt.param_groups[1]["lr"], "images_per_s": r4(ips)}
            hist["steps"].append(rec); json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
            log.info(f"epoch {epoch + 1}/{a.epochs} step {i}/{n} loss {rec['loss']:.4f} lr {rec['lr']:.2e} {ips:.2f} images/s, "
                     f"epoch done in about {(n - i) * a.batch / ips / 60:.1f} min")
            win, seen, t0 = [], 0, time.time()
    return total / n, n * a.batch / (time.time() - t_ep)


def save_checkpoint(model, proc, opt, sched, epoch: int, hist: dict, out: str):
    """checkpoint-last: model and processor by save_pretrained, optimiser, scheduler and epoch for --resume; written aside, then swapped in"""
    tmp, dst = os.path.join(out, "checkpoint-tmp"), os.path.join(out, "checkpoint-last")
    shutil.rmtree(tmp, ignore_errors=True); model.save_pretrained(tmp); proc.save_pretrained(tmp)
    torch.save({"epoch": epoch, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "history": hist}, os.path.join(tmp, "trainer_state.pt"))
    shutil.rmtree(dst, ignore_errors=True); os.rename(tmp, dst)


def over_budget(a, hist: dict, epoch: int) -> str | None:
    """why to stop before this epoch: training time so far (summed over resumes) plus one more epoch, timed by the last, would
    pass --time-budget-min"""
    done = hist["epochs"]
    if not a.time_budget_min or not done:
        return None
    used, last = sum(e["seconds"] for e in done), done[-1]["seconds"]
    if used + last > a.time_budget_min * 60:
        return (f"time budget of {a.time_budget_min:g} min: stopped cleanly after epoch {epoch} of {a.epochs}, {used / 60:.1f} min "
                f"of training used and an epoch takes {last / 60:.1f} min")
    return None


def cmd_train(a, argv: list):
    setup_log(os.path.join(a.out, "train.log")); t_run = time.time(); cmd = command(argv); set_seed(a.seed)
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    if man.get("classes") != CLASSES:
        raise ValueError(f"{a.data}/manifest.json: classes differ from sld2quote/classes.py")
    if man.get("min_visible", a.min_visible) != a.min_visible:
        log.warning(f"--min-visible {a.min_visible} differs from the export's min_visible {man['min_visible']}: zoomed boxes follow another rule")
    ck = os.path.join(a.out, "checkpoint-last"); hist, start = {"command": cmd, "args": vars(a), "steps": [], "epochs": []}, 0
    if a.resume:
        if not os.path.isdir(ck):
            raise SystemExit(f"--resume: no checkpoint at {ck}")
        st = torch.load(os.path.join(ck, "trainer_state.pt"), map_location="cpu", weights_only=True)
        hist, start = st["history"], st["epoch"] + 1; a.model = hist["args"]["model"]; hist.setdefault("resumed", []).append(cmd)
    src = ck if a.resume else a.model
    model, proc = load_model(src).to(a.device), load_processor(src)
    ds = Tiles(a.data, "train", proc, aug=(a.brightness, a.contrast, a.scale_jitter, a.min_visible), limit=a.limit, seed=a.seed)
    steps = len(ds) // a.batch
    if not steps:
        raise SystemExit(f"{len(ds)} training tiles make no full batch of {a.batch}")
    opt = optimiser(model, a.lr, a.backbone_lr, a.weight_decay); sched = get_cosine_schedule_with_warmup(opt, a.warmup_steps, a.epochs * steps)
    if a.resume:
        opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"]); log.info(f"resumed {a.model} from {ck}: {start} epochs done")
    log.info(f"{cmd}\n{src} on {a.device}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters; {len(ds)} train tiles, "
             f"{steps} steps per epoch, {a.epochs * steps} in all; versions {versions()}")
    hist["stopped"] = None
    for epoch in range(start, a.epochs):
        if stop := over_budget(a, hist, epoch):
            hist["stopped"] = stop; log.info(stop); break
        t_ep = time.time(); loss, ips = train_epoch(model, opt, sched, ds, a, epoch, hist)
        rec = {"epoch": epoch + 1, "train_loss": r4(loss), "images_per_s": r4(ips)}
        if a.eval_every and (epoch + 1) % a.eval_every == 0:
            ts, m = evaluate(model, proc, a, ["val"])
            rec["val"] = {v: {k: m[v]["val"][k] for k in VAL_KEYS} | {"threshold": ts[v]} for v in m}
        rec["seconds"] = round(time.time() - t_ep, 1); hist["epochs"].append(rec)
        save_checkpoint(model, proc, opt, sched, epoch, hist, a.out); json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
        log.info(f"epoch {epoch + 1} done in {rec['seconds'] / 60:.1f} min: train loss {rec['train_loss']:.4f}, {ips:.2f} images/s, val {rec.get('val')}")
    t_ev = time.time(); vs = variants(a)
    ts, out = evaluate(model, proc, a, [s for s in EVAL_SPLITS if os.path.isdir(os.path.join(a.data, s))])
    model.config.update({"score_threshold": ts[vs[0]], "score_threshold_raw": ts["raw"], "nms_iou": a.nms_iou})  # they travel with the weights
    model.save_pretrained(a.out); proc.save_pretrained(a.out)
    metrics = {"model": a.model, "weights": a.out, "data": a.data, "manifest_sha256": sha256(os.path.join(a.data, "manifest.json")), "device": a.device,
               "nms_iou": a.nms_iou, "threshold": ts[vs[0]], "threshold_chosen_on": "val", "match_iou": MATCH_IOU, "epochs_run": len(hist["epochs"]),
               "stopped": hist["stopped"], "train_images_per_s": r4(statistics.mean(e["images_per_s"] for e in hist["epochs"])) if hist["epochs"] else None,
               "train_minutes": r4(sum(e["seconds"] for e in hist["epochs"]) / 60), "final_eval_minutes": r4((time.time() - t_ev) / 60),
               "variants": {v: {"threshold": ts[v], "splits": out[v]} for v in vs}}
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1); json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
    open(os.path.join(a.out, "card.md"), "w").write(card(a, man, metrics, hist))
    log.info(f"wrote {a.out}: weights, metrics.json, card.md, history.json; final evaluation {metrics['final_eval_minutes']:.1f} min, "
             f"{(time.time() - t_run) / 60:.1f} min in all")


def card(a, man: dict, metrics: dict, hist: dict) -> str:
    """the model card: what was trained on what, how, and how well, with NMS and raw"""
    V = metrics["variants"]; run = f"{metrics['epochs_run']} of {a.epochs}" + (f" ({hist['stopped']})" if hist["stopped"] else "")
    counts = lambda c: ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in c.items() if isinstance(v, (int, float)))
    post = (f"class-agnostic NMS at IoU {metrics['nms_iou']} on each tile's detections, the operating post-processing, and raw" if metrics["nms_iou"] else "raw")
    L = [f"# sld2quote symbol detector: {a.model} fine-tuned on synthetic tiles", "",
         "The data is synthetic: every tile was cut from a sheet drawn by `sld2quote/generate.py` and every label comes from the "
         "generator's answer key. No real drawing was used for training or testing.", "",
         f"- Model: `{a.model}` (RT-DETR, Hugging Face transformers), {len(CLASSES)} classes as in `sld2quote/classes.py`: {', '.join(CLASSES)}",
         f"- Dataset: `{a.data}`, manifest.json sha256 `{metrics['manifest_sha256']}`, exported {man.get('created', '-')}; tiles {man.get('tile')} px, "
         f"overlap {man.get('overlap')}, a box kept in a tile from {man.get('min_visible')} of its symbol; held out of training: {json.dumps(man.get('held_out'))}",
         *[f"  - {s}: {counts(c)}" for s, c in man.get("splits", {}).items()],
         f"- Command: `{hist['command']}`" + "".join(f"; resumed with `{c}`" for c in hist.get("resumed", [])),
         f"- Seed {a.seed}; device {a.device}; epochs run {run}; {metrics['train_images_per_s']} training images/s; {metrics['train_minutes']:.0f} min of training",
         "- Libraries: " + ", ".join(f"{k} {v}" for k, v in versions().items()),
         f"- Post-processing: {post}; each at its own threshold, chosen on val to maximise the mean per-class F1 ("
         + ", ".join(f"{LABEL[v]} {r['threshold']:.2f}" for v, r in V.items()) + f"). A detection counts when it has the box's class and IoU >= {MATCH_IOU}, "
         "matched greedily by score; scores are per tile, so a symbol in two tiles counts twice, and a symbol cut below the export's "
         "min_visible is unlabelled, so a detection of it at a tile edge counts against precision (the reader merges tiles)"]
    for v, r in V.items():
        S = r["splits"]
        L += ["", f"## Every split {LABEL[v]}, threshold {r['threshold']:.2f}", "",
              "| split | tiles | boxes | mAP@0.5 | mAP@0.5:0.95 | mean P | mean R | min P | min R | images/s |", "|---|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {s} | {m['images']} | {m['boxes']} | {fmt(m['map50'])} | {fmt(m['map'])} | {fmt(m['mean_precision'])} | {fmt(m['mean_recall'])} | "
              f"{fmt(m['min_precision'])} | {fmt(m['min_recall'])} | {m['images_per_s']} |" for s, m in S.items()]
        L += ["", "Precision / recall per class (- where there is nothing to count):", "", "| class | " + " | ".join(S) + " |", "|---|" + "---|" * len(S)]
        L += [f"| {c} | " + " | ".join(f"{fmt(m['per_class'][c]['precision'])} / {fmt(m['per_class'][c]['recall'])}" for m in S.values()) + " |" for c in CLASSES]
        L += [""] + [f"- {s}: recall by copy {json.dumps(m['recall_by'].get('copy'))}; breaker recall by style {json.dumps(m['breaker_recall_by_style'])}; "
                     f"recall by visibility {json.dumps(m['recall_by'].get('visible'))}; {m['false_positives']['count']} false positives, "
                     f"{m['false_positives']['at_tile_edge']} at a tile edge" for s, m in S.items()]
    L += ["", "Full metrics in `metrics.json`, training curve in `history.json`.", "", "Licence: Apache-2.0, as the pretrained RT-DETR weights."]
    return "\n".join(L) + "\n"


# eval

def cmd_eval(a, argv: list):
    if not os.path.isfile(os.path.join(a.weights, "config.json")):
        raise SystemExit(f"--weights {a.weights}: not a trained weights directory (no config.json)")
    setup_log(os.path.join(a.weights, f"eval-{a.split}.log")); log.info(command(argv))
    model, proc = load_model(a.weights).to(a.device), load_processor(a.weights); c, vs = model.config, variants(a)
    if a.threshold is not None:
        ts = {v: a.threshold for v in vs}
    elif getattr(c, "nms_iou", None) == a.nms_iou:  # chosen on val at training time under this post-processing
        ts = {v: c.score_threshold if v == vs[0] else c.score_threshold_raw for v in vs}
    else:
        ts = None
    ts, out = evaluate(model, proc, a, [a.split] if ts else list(dict.fromkeys(["val", a.split])), ts)
    res = {"weights": a.weights, "data": a.data, "manifest_sha256": sha256(os.path.join(a.data, "manifest.json")), "device": a.device,
           "nms_iou": a.nms_iou, "match_iou": MATCH_IOU, "variants": {v: {"threshold": ts[v], "splits": {a.split: out[v][a.split]}} for v in vs}}
    path = a.out or os.path.join(a.weights, f"eval-{a.split}.json"); json.dump(res, open(path, "w"), indent=1); log.info(f"wrote {path}")


# summary

def size_rule(val: dict, unseen: dict, target: float) -> dict:
    """per class: pass when precision and recall reach target on val and on test_unseen, fail otherwise, n/a without boxes in either"""
    rule = {}
    for c in CLASSES:
        x, y = val["per_class"][c], unseen["per_class"][c]
        rule[c] = ("n/a" if not (x["gt"] and y["gt"]) else
                   "pass" if all((m["precision"] or 0) >= target and m["recall"] >= target for m in (x, y)) else "fail")
    return rule


def worst(m: dict, k: str) -> str:
    vals = [(c[k], n) for n, c in m["per_class"].items() if c[k] is not None]
    return f"{min(vals)[0]} ({min(vals)[1]})" if vals else "-"


def reproduced(d: str, V: dict) -> str:
    """whether eval, run on every split after training, gives the training-time metrics (timings aside)"""
    same = lambda x, y: {k: v for k, v in x.items() if k != "images_per_s"} == {k: v for k, v in y.items() if k != "images_per_s"}
    got = {}
    for s in next(iter(V.values()))["splits"]:
        p = os.path.join(d, f"eval-{s}.json")
        if os.path.isfile(p):
            e = json.load(open(p))["variants"]
            got[s] = all(s in e.get(v, {}).get("splits", {}) and same(e[v]["splits"][s], r["splits"][s]) for v, r in V.items())
    bad = [s for s, ok in got.items() if not ok]
    return ("eval was not run after training" if not got else f"eval of {', '.join(got)} after training " +
            ("reproduces the training-time metrics exactly" if not bad else f"differs from the training-time metrics on {', '.join(bad)}"))


def plain_words(name: str, V: dict) -> str:
    """what the model still gets wrong, in plain words for the project owner, on test_unseen (val if absent) as it would run"""
    op = next(iter(V)); S = V[op]["splits"]; s = "test_unseen" if "test_unseen" in S else "val"; m = S[s]; pc = m["per_class"]
    tp, gt, pred = (sum(c[k] for c in pc.values()) for k in ("tp", "gt", "pred"))
    pct, fp = (lambda x: f"{100 * x:.0f}%"), m["false_positives"]
    words = [f"On the {'unseen-style' if s == 'test_unseen' else 'validation'} sheets {name} finds {pct(tp / max(gt, 1))} of the symbols, and " +
             (f"{pct(1 - tp / pred)} of what it reports is a false alarm ({fp['count']} on {m['images']} tiles, {fp['at_tile_edge']} of them at a tile edge)."
              if pred else "it reports nothing at its threshold.")]
    weak = sorted((c for c in CLASSES if pc[c]["recall"] is not None and pc[c]["recall"] < 0.95), key=lambda c: pc[c]["recall"])
    if weak:
        words.append("It misses too many " + ", ".join(f"{PLAIN[c]} (finds {pct(pc[c]['recall'])})" for c in weak[:4]) + ".")
    noisy = sorted((c for c in CLASSES if pc[c]["precision"] is not None and pc[c]["precision"] < 0.95), key=lambda c: pc[c]["precision"])
    if noisy:
        words.append("It reports some symbols that are not there, mostly " + ", ".join(
            f"{PLAIN[c]} ({pct(1 - pc[c]['precision'])} of those reported are wrong)" for c in noisy[:3]) + ".")
    copy, vis, style = m["recall_by"].get("copy", {}), m["recall_by"].get("visible", {}), m["breaker_recall_by_style"]
    if copy.get("clean", 0) - copy.get("scan", 1) >= 0.02:
        words.append(f"Scan-like copies are harder: it finds {pct(copy['scan'])} of their symbols against {pct(copy['clean'])} on clean copies.")
    if "box" in style and "iec" in style and abs(style["box"] - style["iec"]) >= 0.02:
        words.append(f"It finds {pct(style['box'])} of the breakers drawn as a box with a diagonal and {pct(style['iec'])} of the IEC breakers.")
    if "partial" in vis:
        words.append(f"It finds {pct(vis['partial'])} of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it.")
    raw_fp = V["raw"]["splits"][s]["false_positives"]["count"]
    if op != "raw" and raw_fp != fp["count"]:
        words.append(f"Without duplicate suppression, at its own best threshold, it would report {raw_fp} false alarms on the same tiles instead of {fp['count']}.")
    return " ".join(words)


def summarise(d: str, target: float) -> list:
    """one model's section of the overnight summary"""
    if not os.path.isfile(os.path.join(d, "metrics.json")):
        return [f"## `{d}`", "", f"No metrics.json: the run did not finish; see `{d}/train.log`.", ""]
    m, h = json.load(open(os.path.join(d, "metrics.json"))), json.load(open(os.path.join(d, "history.json")))
    V, g = m["variants"], h["args"]; op = next(iter(V)); S = V[op]["splits"]; name = os.path.basename(m["model"])
    L = [f"## {name} (`{d}`)", "",
         f"- {m['epochs_run']} of {g['epochs']} epochs at batch {g['batch']}, lr {g['lr']:.0e} (backbone {g['backbone_lr']:.0e}), warm-up {g['warmup_steps']} steps, "
         f"seed {g['seed']}: {m['train_minutes']:.0f} min of training" + (f" ({m['stopped']})" if m["stopped"] else "") +
         f" and {m['final_eval_minutes']:.1f} min of final evaluation, {m['train_images_per_s']} training images/s",
         f"- Thresholds chosen on val: " + ", ".join(f"{LABEL[v]} {r['threshold']:.2f}" for v, r in V.items()) + f"; NMS is class-agnostic at IoU {m['nms_iou']}",
         f"- {reproduced(d, V)}", "",
         "| split | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |", "|---|---|---|---|---|---|---|"]
    L += [f"| {s} | {LABEL[v]} | {fmt(r['splits'][s]['map50'])} | {fmt(r['splits'][s]['mean_precision'])} | {worst(r['splits'][s], 'precision')} | "
          f"{fmt(r['splits'][s]['mean_recall'])} | {worst(r['splits'][s], 'recall')} |" for s in S for v, r in V.items()]
    if "val" in S and "test_unseen" in S:
        rule = size_rule(S["val"], S["test_unseen"], target); fails = [c for c, x in rule.items() if x == "fail"]
        L += ["", f"Per class {LABEL[op]}, and the size rule (precision and recall at least {target} on val and on test_unseen):", "",
              "| class | val P | val R | test_unseen P | test_unseen R | rule |", "|---|---|---|---|---|---|"]
        L += [f"| {c} | " + " | ".join(fmt(S[s]["per_class"][c][k]) for s in ("val", "test_unseen") for k in ("precision", "recall")) + f" | {rule[c]} |" for c in CLASSES]
        na = sum(x == "n/a" for x in rule.values())
        L += ["", f"Size rule: {sum(x == 'pass' for x in rule.values())} of {len(CLASSES)} classes pass" +
              (f", {na} cannot be judged (no boxes in val or test_unseen)" if na else "") + "; " + (f"fails on {', '.join(fails)}." if fails else "no class fails.")]
    L += ["", "Training curve, val after each epoch:", "", "| epoch | train loss | val mAP@0.5 with NMS | val mAP@0.5 raw | images/s | minutes |", "|---|---|---|---|---|---|"]
    L += [f"| {e['epoch']} | {e['train_loss']} | {fmt(e.get('val', {}).get('nms', {}).get('map50'))} | {fmt(e.get('val', {}).get('raw', {}).get('map50'))} | "
          f"{e['images_per_s']} | {e['seconds'] / 60:.1f} |" for e in h["epochs"]]
    return L + ["", "What it still gets wrong: " + plain_words(name, V), ""]


def cmd_summary(a, argv: list):
    L = ["# Overnight detector runs", "",
         "Synthetic data only: every tile was cut from a sheet drawn by `sld2quote/generate.py`. Scores are per 640 px tile, so a symbol in two "
         "tiles counts twice. \"With NMS\" is the operating post-processing (class-agnostic duplicate suppression); \"raw\" is without it; each "
         "uses its own threshold chosen on val.", ""]
    for d in a.runs.split(","):
        L += summarise(d, a.target)
    open(a.out, "w").write("\n".join(L) + "\n"); print(f"wrote {a.out}")


# bench

@torch.no_grad()
def time_sheet(model, proc, sheet, device: str, batch: int, tile: int, overlap: float, floor: float, nms_iou: float) -> tuple[float, int]:
    """seconds to read one grey sheet: cut the tiles as the export does (tiles.py), run them through detect() in batches; and the tile count"""
    t0 = time.perf_counter(); rgb = sheet.convert("RGB")
    tiles = [cut_tile(rgb, x, y, tile) for x, y in tile_grid(*rgb.size, tile, overlap)]
    for i in range(0, len(tiles), batch):
        pv = proc(images=tiles[i:i + batch], return_tensors="pt")["pixel_values"].to(device)
        [r["boxes"].cpu() for r in detect(model, proc, pv, [(tile, tile)] * len(pv), floor, nms_iou)]  # waits for the device to finish
    return time.perf_counter() - t0, len(tiles)


def machine() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    except OSError:  # no sysctl, as on some Linux images
        return platform.processor() or platform.machine()


def cmd_bench(a, argv: list):
    setup_log(os.path.join(a.out, "bench.log")); log.info(command(argv))
    sheets = [tuple(map(int, s.split("x"))) for s in a.sheets.split(",")]
    runs = [(d, int(n)) for d in a.device.split(",") for n in (a.threads.split(",") if d == "cpu" else [torch.get_num_threads()])]
    rows = []
    for name in a.models.split(","):
        model, proc = load_model(name).eval(), load_processor(name)
        for dev, threads in runs:
            torch.set_num_threads(threads); model.to(dev)
            for w, h in sheets:  # a white sheet: RT-DETR does the same work on any content, a fixed 300 queries per tile
                times = [time_sheet(model, proc, Image.new("L", (w, h), 255), dev, a.batch, a.tile, a.overlap, a.score_floor, a.nms_iou)
                         for _ in range(a.warmup + a.repeats)]
                secs = [round(s, 3) for s, _ in times[a.warmup:]]
                rows.append({"model": name, "device": dev, "threads": threads, "sheet": f"{w}x{h}", "tiles": times[0][1], "median_s": statistics.median(secs), "runs_s": secs})
                log.info(json.dumps(rows[-1]))
    L = [f"Detector speed per sheet on {machine()}, torch {torch.__version__}; tiles {a.tile} px at stride {round(a.tile * (1 - a.overlap))} cut by "
         f"sld2quote/tiles.py, batches of up to {a.batch}, " + (f"class-agnostic NMS at IoU {a.nms_iou}" if a.nms_iou else "no NMS") +
         f"; median of {a.repeats} runs after {a.warmup} warm-up; untrained 14-class heads (speed only)", "",
         "| model | device | sheet | tiles | median s | runs s |", "|---|---|---|---|---|---|"]
    L += [f"| {os.path.basename(r['model'])} | {r['device']}" + (f", {r['threads']} threads" if r["device"] == "cpu" else "") +
          f" | {r['sheet']} | {r['tiles']} | {r['median_s']:.2f} | {', '.join(f'{s:.2f}' for s in r['runs_s'])} |" for r in rows]
    json.dump({"machine": machine(), "torch": torch.__version__, "command": command(argv), "rows": rows}, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
    open(os.path.join(a.out, "bench.md"), "w").write("\n".join(L) + "\n"); log.info("\n" + "\n".join(L))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python3 -m sld2quote.train_detector", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    nms_help = "class-agnostic NMS IoU on each tile's detections, 0 disables it; metrics are reported with it and raw"
    run = argparse.ArgumentParser(add_help=False)  # shared by train and eval
    run.add_argument("--data", default="data/coco"); run.add_argument("--device", default="mps"); run.add_argument("--batch", type=int, default=8)
    run.add_argument("--workers", type=int, default=4, help="data-loading processes"); run.add_argument("--limit", type=int, help="first N tiles per split, for smoke runs")
    run.add_argument("--score-floor", type=float, default=0.01, help="detections below this score are dropped before NMS, mAP and matching")
    run.add_argument("--nms-iou", type=float, default=0.5, help=nms_help)
    t = sub.add_parser("train", parents=[run], help="fine-tune on train; per-epoch val; metrics.json and card.md at the end")
    t.add_argument("--out", default="weights/rtdetr_r18"); t.add_argument("--model", default="PekingU/rtdetr_r18vd")
    t.add_argument("--epochs", type=int, default=3); t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--backbone-lr", type=float, default=1e-5)
    t.add_argument("--weight-decay", type=float, default=1e-4); t.add_argument("--warmup-steps", type=int, default=200); t.add_argument("--grad-clip", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint-last")
    t.add_argument("--time-budget-min", type=float, help="training minutes, summed over resumes: stop at the first epoch boundary where one more "
                   "epoch would pass it; the final evaluation runs after")
    t.add_argument("--eval-every", type=int, default=1, help="val metrics every N epochs, 0 for none"); t.add_argument("--log-every", type=int, default=20, help="steps")
    t.add_argument("--brightness", type=float, default=0.2); t.add_argument("--contrast", type=float, default=0.2)
    t.add_argument("--scale-jitter", type=float, default=0.1, help="zoom by up to this fraction either way")
    t.add_argument("--min-visible", type=float, default=0.6, help="keep a box the zoom cuts while this fraction of its symbol shows, as the export's tiling")
    e = sub.add_parser("eval", parents=[run], help="the same metrics for one split, with NMS and raw")
    e.add_argument("--weights", required=True); e.add_argument("--split", required=True)
    e.add_argument("--threshold", type=float, help="one threshold for every variant; default: those chosen on val at training time")
    e.add_argument("--out", help="default: <weights>/eval-<split>.json")
    s = sub.add_parser("summary", help="per model: every split with NMS and raw, per-class size rule, training curve, plain words")
    s.add_argument("--runs", required=True, help="comma-separated weights directories"); s.add_argument("--out", required=True)
    s.add_argument("--target", type=float, default=0.95, help="precision and recall every class must reach on val and test_unseen")
    b = sub.add_parser("bench", help="seconds per whole sheet for each model on each device")
    b.add_argument("--models", default="PekingU/rtdetr_r18vd,PekingU/rtdetr_r50vd"); b.add_argument("--device", default="cpu,mps")
    b.add_argument("--threads", default=f"{torch.get_num_threads()},2", help="cpu thread counts to time (a free Hugging Face CPU Space has 2 vCPUs)")
    b.add_argument("--sheets", default="1600x1000,3200x1000"); b.add_argument("--tile", type=int, default=640); b.add_argument("--overlap", type=float, default=0.25)
    b.add_argument("--batch", type=int, default=16, help="tiles per forward pass"); b.add_argument("--warmup", type=int, default=1); b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--score-floor", type=float, default=0.01); b.add_argument("--nms-iou", type=float, default=0.5, help=nms_help); b.add_argument("--out", default="weights/bench")
    a = ap.parse_args(argv)
    {"train": cmd_train, "eval": cmd_eval, "summary": cmd_summary, "bench": cmd_bench}[a.cmd](a, argv)


if __name__ == "__main__":
    main()
