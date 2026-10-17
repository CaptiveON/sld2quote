"""RT-DETR symbol detector: fine-tune on the synthetic COCO tiles, evaluate any split, time whole sheets on cpu and mps.

  python3 -m sld2quote.train_detector train --data data/coco --out weights/rtdetr_r18 --model PekingU/rtdetr_r18vd --epochs 3
  python3 -m sld2quote.train_detector eval --weights weights/rtdetr_r18 --data data/coco --split test_unseen
  python3 -m sld2quote.train_detector bench --models PekingU/rtdetr_r18vd,PekingU/rtdetr_r50vd --device cpu,mps

The export at data/coco holds manifest.json and <split>/annotations.json with <split>/images/*.png: grey 640 px tiles, COCO boxes,
0-based category ids in classes.CLASSES order. Precision and recall match greedily by score, as evaluate.py does, at IoU >= 0.5;
mAP comes from pycocotools. The data is synthetic: every tile is cut from a sheet drawn by generate.py."""
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


def r4(v):
    return None if v is None else round(float(v), 4)


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


@torch.no_grad()
def predict(model, proc, ds: Tiles, a) -> tuple[list, float]:
    """detections at or above the score floor in COCO form (tile pixels), and images per second including loading"""
    model.eval(); dets = []; t0 = time.time()
    for b in loader(ds, a, train=False):
        out = model(pixel_values=b["pixel_values"].to(a.device))
        sizes = [tuple(l["orig_size"].tolist()) for l in b["labels"]]
        for l, r in zip(b["labels"], proc.post_process_object_detection(out, threshold=a.score_floor, target_sizes=sizes)):
            for s, c, (x0, y0, x1, y1) in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
                dets.append({"image_id": int(l["image_id"]), "category_id": c, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s})
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
    with contextlib.redirect_stdout(io.StringIO()) as out:
        gt = COCO(); gt.dataset = {"images": ds.images, "annotations": anns, "categories": ds.categories}; gt.createIndex()
        ev = COCOeval(gt, gt.loadRes([dict(d) for d in dets]), "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    log.info("pycocotools\n" + "\n".join(l for l in out.getvalue().splitlines() if l.lstrip().startswith("Average")))
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


def evaluate(model, proc, a, splits: list, t: float | None = None) -> tuple[float, dict]:
    """metrics per split at one operating threshold: t if given, otherwise the one that maximises mean F1 on val"""
    if t is None and "val" not in splits:
        raise ValueError("the operating threshold is chosen on val, which is not among the splits")
    runs = {}
    for s in splits:
        ds = Tiles(a.data, s, proc, limit=a.limit); dets, ips = predict(model, proc, ds, a)
        runs[s] = (ds, dets, ips, scored(ds.annotations(), dets))
    if t is None:
        t = choose_threshold(runs["val"][3])
    out = {s: split_metrics(*run, t) for s, run in runs.items()}
    for s, m in out.items():
        log.info(f"{s}: {m['images']} tiles, {m['boxes']} boxes; mAP@0.5 {m['map50']}, mAP@0.5:0.95 {m['map']}; at threshold {t:.2f}: "
                 f"mean P {m['mean_precision']}, mean R {m['mean_recall']}, min P {m['min_precision']}, min R {m['min_recall']}; "
                 f"{m['false_positives']['count']} false positives, {m['false_positives']['at_tile_edge']} at a tile edge; {m['images_per_s']:.1f} images/s")
    return t, out


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
            t, m = evaluate(model, proc, a, ["val"]); v = m["val"]
            rec["val"] = {k: v[k] for k in ("map50", "map", "mean_precision", "mean_recall", "min_precision", "min_recall")} | {"threshold": t}
        rec["seconds"] = round(time.time() - t_ep, 1); hist["epochs"].append(rec)
        save_checkpoint(model, proc, opt, sched, epoch, hist, a.out); json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
        log.info(f"epoch {epoch + 1} done in {rec['seconds'] / 60:.1f} min: train loss {rec['train_loss']:.4f}, {ips:.2f} images/s, val {rec.get('val')}")
    t_ev = time.time(); t, splits = evaluate(model, proc, a, [s for s in EVAL_SPLITS if os.path.isdir(os.path.join(a.data, s))])
    model.config.score_threshold = t; model.save_pretrained(a.out); proc.save_pretrained(a.out)  # the operating threshold travels with the weights
    metrics = {"model": a.model, "weights": a.out, "data": a.data, "manifest_sha256": sha256(os.path.join(a.data, "manifest.json")), "device": a.device,
               "threshold": t, "threshold_chosen_on": "val", "match_iou": MATCH_IOU, "epochs_run": len(hist["epochs"]), "stopped": hist["stopped"],
               "train_images_per_s": r4(statistics.mean(e["images_per_s"] for e in hist["epochs"])) if hist["epochs"] else None, "splits": splits}
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1); json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
    open(os.path.join(a.out, "card.md"), "w").write(card(a, man, metrics, hist))
    log.info(f"wrote {a.out}: weights, metrics.json, card.md, history.json; final evaluation {(time.time() - t_ev) / 60:.1f} min, "
             f"{(time.time() - t_run) / 60:.1f} min in all")


def card(a, man: dict, metrics: dict, hist: dict) -> str:
    """the model card: what was trained on what, how, and how well"""
    S, f = metrics["splits"], lambda v: "-" if v is None else str(v)
    run = f"{metrics['epochs_run']} of {a.epochs}" + (f" ({hist['stopped']})" if hist["stopped"] else "")
    counts = lambda c: ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in c.items() if isinstance(v, (int, float)))
    L = [f"# sld2quote symbol detector: {a.model} fine-tuned on synthetic tiles", "",
         "The data is synthetic: every tile was cut from a sheet drawn by `sld2quote/generate.py` and every label comes from the "
         "generator's answer key. No real drawing was used for training or testing.", "",
         f"- Model: `{a.model}` (RT-DETR, Hugging Face transformers), {len(CLASSES)} classes as in `sld2quote/classes.py`: {', '.join(CLASSES)}",
         f"- Dataset: `{a.data}`, manifest.json sha256 `{metrics['manifest_sha256']}`, exported {man.get('created', '-')}; tiles {man.get('tile')} px, "
         f"overlap {man.get('overlap')}, a box kept in a tile from {man.get('min_visible')} of its symbol; held out of training: {json.dumps(man.get('held_out'))}",
         *[f"  - {s}: {counts(c)}" for s, c in man.get("splits", {}).items()],
         f"- Command: `{hist['command']}`" + "".join(f"; resumed with `{c}`" for c in hist.get("resumed", [])),
         f"- Seed {a.seed}; device {a.device}; epochs run {run}; {metrics['train_images_per_s']} training images/s",
         "- Libraries: " + ", ".join(f"{k} {v}" for k, v in versions().items()),
         f"- Operating threshold {metrics['threshold']:.2f}, chosen on val to maximise the mean per-class F1; a detection counts when it has "
         f"the box's class and IoU >= {MATCH_IOU}, matched greedily by score; scores are per tile, so a symbol in two tiles counts twice", "",
         "| split | tiles | boxes | mAP@0.5 | mAP@0.5:0.95 | mean P | mean R | min P | min R | images/s |", "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {s} | {m['images']} | {m['boxes']} | {f(m['map50'])} | {f(m['map'])} | {f(m['mean_precision'])} | {f(m['mean_recall'])} | "
          f"{f(m['min_precision'])} | {f(m['min_recall'])} | {m['images_per_s']} |" for s, m in S.items()]
    L += ["", "Precision / recall per class at the operating threshold (- where there is nothing to count):", "",
          "| class | " + " | ".join(S) + " |", "|---|" + "---|" * len(S)]
    L += [f"| {c} | " + " | ".join(f"{f(m['per_class'][c]['precision'])} / {f(m['per_class'][c]['recall'])}" for m in S.values()) + " |" for c in CLASSES]
    L += ["", "Recall by copy and breaker style, and false positives (a symbol cut below the export's min_visible is unlabelled, so a "
          "detection of it at a tile edge counts against precision here; the reader merges tiles):", ""]
    L += [f"- {s}: recall by copy {json.dumps(m['recall_by'].get('copy'))}; breaker recall by style {json.dumps(m['breaker_recall_by_style'])}; "
          f"recall by visibility {json.dumps(m['recall_by'].get('visible'))}; {m['false_positives']['count']} false positives, "
          f"{m['false_positives']['at_tile_edge']} at a tile edge" for s, m in S.items()]
    L += ["", "Full metrics in `metrics.json`, training curve in `history.json`.", "", "Licence: Apache-2.0, as the pretrained RT-DETR weights."]
    return "\n".join(L) + "\n"


# eval

def cmd_eval(a, argv: list):
    if not os.path.isfile(os.path.join(a.weights, "config.json")):
        raise SystemExit(f"--weights {a.weights}: not a trained weights directory (no config.json)")
    setup_log(os.path.join(a.weights, f"eval-{a.split}.log")); log.info(command(argv))
    model, proc = load_model(a.weights).to(a.device), load_processor(a.weights)
    t = a.threshold if a.threshold is not None else getattr(model.config, "score_threshold", None)
    t, splits = evaluate(model, proc, a, [a.split] if t is not None else list(dict.fromkeys(["val", a.split])), t)
    out = {"weights": a.weights, "data": a.data, "manifest_sha256": sha256(os.path.join(a.data, "manifest.json")), "device": a.device,
           "threshold": t, "match_iou": MATCH_IOU, "splits": {a.split: splits[a.split]}}
    path = a.out or os.path.join(a.weights, f"eval-{a.split}.json"); json.dump(out, open(path, "w"), indent=1); log.info(f"wrote {path}")


# bench

@torch.no_grad()
def time_sheet(model, proc, sheet, device: str, batch: int, tile: int, overlap: float, floor: float) -> tuple[float, int]:
    """seconds to read one grey sheet: cut the tiles as the export does (tiles.py), run them through the detector in batches,
    post-process; and the tile count"""
    t0 = time.perf_counter(); rgb = sheet.convert("RGB")
    tiles = [cut_tile(rgb, x, y, tile) for x, y in tile_grid(*rgb.size, tile, overlap)]
    for i in range(0, len(tiles), batch):
        pv = proc(images=tiles[i:i + batch], return_tensors="pt")["pixel_values"].to(device)
        res = proc.post_process_object_detection(model(pixel_values=pv), threshold=floor, target_sizes=[(tile, tile)] * len(pv))
        [r["boxes"].cpu() for r in res]  # waits for the device to finish
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
                times = [time_sheet(model, proc, Image.new("L", (w, h), 255), dev, a.batch, a.tile, a.overlap, a.score_floor) for _ in range(a.warmup + a.repeats)]
                secs = [round(s, 3) for s, _ in times[a.warmup:]]
                rows.append({"model": name, "device": dev, "threads": threads, "sheet": f"{w}x{h}", "tiles": times[0][1], "median_s": statistics.median(secs), "runs_s": secs})
                log.info(json.dumps(rows[-1]))
    L = [f"Detector speed per sheet on {machine()}, torch {torch.__version__}; tiles {a.tile} px at stride {round(a.tile * (1 - a.overlap))} cut by "
         f"sld2quote/tiles.py, batches of up to {a.batch}; median of {a.repeats} runs after {a.warmup} warm-up; untrained 14-class heads (speed only)", "",
         "| model | device | sheet | tiles | median s | runs s |", "|---|---|---|---|---|---|"]
    L += [f"| {os.path.basename(r['model'])} | {r['device']}" + (f", {r['threads']} threads" if r["device"] == "cpu" else "") +
          f" | {r['sheet']} | {r['tiles']} | {r['median_s']:.2f} | {', '.join(f'{s:.2f}' for s in r['runs_s'])} |" for r in rows]
    json.dump({"machine": machine(), "torch": torch.__version__, "command": command(argv), "rows": rows}, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
    open(os.path.join(a.out, "bench.md"), "w").write("\n".join(L) + "\n"); log.info("\n" + "\n".join(L))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python3 -m sld2quote.train_detector", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = argparse.ArgumentParser(add_help=False)  # shared by train and eval
    run.add_argument("--data", default="data/coco"); run.add_argument("--device", default="mps"); run.add_argument("--batch", type=int, default=8)
    run.add_argument("--workers", type=int, default=4, help="data-loading processes"); run.add_argument("--limit", type=int, help="first N tiles per split, for smoke runs")
    run.add_argument("--score-floor", type=float, default=0.01, help="detections below this score are dropped before mAP and matching")
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
    e = sub.add_parser("eval", parents=[run], help="the same metrics for one split")
    e.add_argument("--weights", required=True); e.add_argument("--split", required=True)
    e.add_argument("--threshold", type=float, help="default: the one chosen on val at training time"); e.add_argument("--out", help="default: <weights>/eval-<split>.json")
    b = sub.add_parser("bench", help="seconds per whole sheet for each model on each device")
    b.add_argument("--models", default="PekingU/rtdetr_r18vd,PekingU/rtdetr_r50vd"); b.add_argument("--device", default="cpu,mps")
    b.add_argument("--threads", default=f"{torch.get_num_threads()},2", help="cpu thread counts to time (a free Hugging Face CPU Space has 2 vCPUs)")
    b.add_argument("--sheets", default="1600x1000,3200x1000"); b.add_argument("--tile", type=int, default=640); b.add_argument("--overlap", type=float, default=0.25)
    b.add_argument("--batch", type=int, default=16, help="tiles per forward pass"); b.add_argument("--warmup", type=int, default=1); b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--score-floor", type=float, default=0.01); b.add_argument("--out", default="weights/bench")
    a = ap.parse_args(argv)
    {"train": cmd_train, "eval": cmd_eval, "bench": cmd_bench}[a.cmd](a, argv)


if __name__ == "__main__":
    main()
