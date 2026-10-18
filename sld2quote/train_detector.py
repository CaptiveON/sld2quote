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
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import ops
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, get_cosine_schedule_with_warmup, set_seed
from transformers.utils import logging as hf_logging
from .classes import CLASSES, CLASS_ID
from .tiles import cut_tile, tile_grid

log = logging.getLogger("sld2quote.train_detector")
MATCH_IOU = 0.5                                                # a detection finds a box of its class when they overlap this much
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)         # operating thresholds tried on val
EVAL_SPLITS = ("val", "val_unseen", "test_clean", "test_scan", "test_unseen")
SELECT = ("val", "val_unseen")  # the splits a pass and the thresholds are chosen on, where present; test_unseen stays for the final numbers
STYLE_KEYS = ("copy", "breaker", "level", "font", "label_pos")  # image keys the export adds, for recall by style
EDGE_PX = 2                                                    # a detection this close to the tile border touches the edge
SLOWED = 0.85                                                  # a pass below this share of its run's best rate was slowed by other work
VAL_KEYS = ("map50", "map", "mean_precision", "mean_recall", "min_precision", "min_recall", "mean_f1")  # kept per epoch in history.json
SCORE = {"map50": ("mean_map50", "mAP@0.5"), "f1": ("pooled_mean_f1", "mean per-class F1 at the operating threshold")}  # --select: key, wording
LABEL = {"nms": "with NMS", "raw": "raw"}
PLAIN = {"breaker": "breakers", "fuse_switch": "fuse switches", "isolator": "isolators", "contactor": "contactors", "transformer": "transformers",
         "generator": "generators", "motor": "motors", "board": "downstream boards", "ct": "current transformers", "meter": "meters",
         "relay": "protection relays", "spd": "surge arresters", "capacitor": "capacitors", "earth": "earth symbols"}  # for the owner's summary


def r4(v):
    return None if v is None else round(float(v), 4)


def iou(a, b) -> float:
    """IoU of two [x, y, w, h] boxes, as evaluate.py computes it; kept here because the readers that evaluate.py runs import this module"""
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy; return inter / float(aw * ah + bw * bh - inter + 1e-9)


def fmt(v) -> str:
    return "-" if v is None else str(v)


# data

class Tiles(Dataset):
    """One split of the COCO export, read one tile at a time. aug = (brightness, contrast, scale, min_visible) for training."""

    def __init__(self, root: str, split: str, proc, aug: tuple | None = None, limit: int | None = None, seed: int = 0):
        js = json.load(open(os.path.join(root, split, "annotations.json")))
        if {c["id"]: c["name"] for c in js["categories"]} != dict(enumerate(CLASSES)):
            raise ValueError(f"{root}/{split}: categories do not match sld2quote/classes.py")
        self.dir, self.proc, self.aug, self.seed, self.epoch, self.factors = os.path.join(root, split, "images"), proc, aug, seed, 0, None
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
        i, rep = i % len(self.images), i // len(self.images)  # repeat k of tile i (repeat-factor sampling) comes as i + k * len
        im = self.images[i]; anns = self.anns[im["id"]]
        img = Image.open(os.path.join(self.dir, os.path.basename(im["file_name"]))).convert("RGB")
        if self.aug:  # seeded by tile, repeat and epoch, so a run repeats whatever the number of workers and survives a resume
            img, anns = jitter(img, anns, random.Random(f"{self.seed}:{self.epoch}:{i}" + (f":{rep}" if rep else "")), *self.aug)
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


def repeat_factors(ds: Tiles, t: float) -> tuple[list, dict]:
    """repeat-factor sampling, the LVIS rule: per class r(c) = max(1, sqrt(t / f(c))), f(c) the fraction of the split's tiles holding
    class c; per tile the largest factor among its classes, 1 for a tile with none. Returns the tile factors and the class factors."""
    held = [{a["category_id"] for a in ds.anns[im["id"]]} for im in ds.images]
    f = Counter(c for h in held for c in h)
    rc = {c: max(1.0, math.sqrt(t * len(held) / f[c])) for c in f}
    return [max((rc[c] for c in h), default=1.0) for h in held], rc


def repeat_indices(factors: list, seed: int, epoch: int) -> list:
    """one epoch's draw: tile i appears floor(r) times and once more with probability r - floor(r), so its expected count is its
    factor r; seeded by seed and epoch, so a run repeats and survives a resume. Repeat k of tile i is index i + k * len(factors),
    so each repeat gets its own augmentation; the epoch is a little longer than the split."""
    rng, n, out = random.Random(f"{seed}:repeat:{epoch}"), len(factors), []
    for i, r in enumerate(factors):
        out += [i + k * n for k in range(int(r) + (rng.random() < r - int(r)))]
    return out


def collate(batch):
    return {"pixel_values": torch.stack([b[0] for b in batch]), "labels": [b[1] for b in batch]}


def loader(ds: Tiles, a, train: bool) -> DataLoader:
    """workers come from a fork server: under spawn, on this Mac (torch 2.5.1, macOS 27), every pass ended with a 5 s join
    timeout per worker although the workers had exited, 20 s per split or epoch with 4 workers. With repeat factors, a training
    pass shuffles that epoch's repeated draw instead of the plain split."""
    g = torch.Generator().manual_seed(ds.seed + ds.epoch)
    rep = train and ds.factors is not None
    return DataLoader(ds, batch_size=a.batch, shuffle=train and not rep, drop_last=train, num_workers=a.workers, collate_fn=collate,
                      sampler=SubsetRandomSampler(repeat_indices(ds.factors, ds.seed, ds.epoch), generator=g) if rep else None,
                      multiprocessing_context="forkserver" if a.workers else None, generator=g)


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


def pool(scs: list):
    """several splits' scored detections as one, for choosing a threshold: each split was matched on its own tiles, so only the
    per-box and per-detection arrays are joined"""
    return SimpleNamespace(**{k: np.concatenate([getattr(s, k) for s in scs]) for k in ("gt_cls", "got", "det_cls", "det_score")})


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
    maximises mean F1 under that variant's post-processing on val, pooled with val_unseen where it is among the splits"""
    if not ts and not set(SELECT) & set(splits):
        raise ValueError("the operating thresholds are chosen on val and val_unseen, and neither is among the splits")
    runs = {}
    for s in splits:
        ds = Tiles(a.data, s, proc, limit=a.limit); dets, ips = predict(model, proc, ds, a)
        runs[s] = (ds, ips, {v: (d, scored(ds.annotations(), d)) for v, d in dets.items()})
    ts = ts or {v: choose_threshold(pool([runs[s][2][v][1] for s in SELECT if s in runs])) for v in variants(a)}
    out, sel = {v: {} for v in variants(a)}, [s for s in SELECT if s in runs]
    for s, (ds, ips, vs) in runs.items():
        for v, (d, sc) in vs.items():
            m = out[v][s] = split_metrics(ds, d, ips, sc, ts[v])
            log.info(f"{s} {LABEL[v]}: {m['images']} tiles, {m['boxes']} boxes; mAP@0.5 {m['map50']}, mAP@0.5:0.95 {m['map']}; at threshold {ts[v]:.2f}: "
                     f"mean P {m['mean_precision']}, mean R {m['mean_recall']}, min P {m['min_precision']}, min R {m['min_recall']}; "
                     f"{m['false_positives']['count']} false positives, {m['false_positives']['at_tile_edge']} at a tile edge; {m['images_per_s']:.1f} images/s")
    return ts, out, ({v: selection(pool([runs[s][2][v][1] for s in sel]), ts[v], out[v]) for v in out} if sel else {})


def selection(sc, t: float, out: dict) -> dict:
    """the scores a pass can be chosen by: the mean per-class F1 at the operating threshold on val and val_unseen pooled, the mean
    of their mAP@0.5, and val_unseen's worst class, the smaller of its precision and recall"""
    f1, ngt = pr(sc, [t])[2][:, 0], np.bincount(sc.gt_cls, minlength=len(CLASSES)); vu = out.get("val_unseen")
    worst = min((min(c["precision"] or 0.0, c["recall"]) for c in vu["per_class"].values() if c["gt"]), default=None) if vu else None
    return {"pooled_mean_f1": r4(f1[ngt > 0].mean()) if ngt.any() else None, "mean_map50": r4(statistics.mean(out[s]["map50"] or 0.0 for s in SELECT if s in out)),
            "val_unseen_worst_min_pr": r4(worst)}


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


def save_dir(dst: str, model, proc, state: dict | None = None):
    """model and processor by save_pretrained, and the trainer state if given, written aside, then swapped in"""
    tmp = dst + "-tmp"
    shutil.rmtree(tmp, ignore_errors=True); model.save_pretrained(tmp); proc.save_pretrained(tmp)
    if state:
        torch.save(state, os.path.join(tmp, "trainer_state.pt"))
    shutil.rmtree(dst, ignore_errors=True); os.rename(tmp, dst)


def save_checkpoint(model, proc, opt, sched, epoch: int, hist: dict, out: str):
    """checkpoint-last: model and processor, optimiser, scheduler and epoch for --resume"""
    save_dir(os.path.join(out, "checkpoint-last"), model, proc, {"epoch": epoch, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "history": hist})


def curve(a, hist: dict) -> list:
    """(epoch, selection score) for every pass that ran val: mAP@0.5 under the operating post-processing on val, averaged with
    val_unseen where the pass scored it; the score early stopping and the best pass watch"""
    v, k = variants(a)[0], SCORE[getattr(a, "select", "map50")][0]
    old = "map50" if k == "mean_map50" else "mean_f1"  # a history from before the selection scores were recorded
    score = lambda e: (e["selection"][v][k] if "selection" in e else statistics.mean(e[s][v].get(old) or 0.0 for s in SELECT if s in e)) or 0.0
    return [(e["epoch"], score(e)) for e in hist["epochs"] if "val" in e]


def watched(a, hist: dict) -> str:
    e, f1 = next((e for e in reversed(hist["epochs"]) if "val" in e), {}), getattr(a, "select", "map50") == "f1"
    both = ("val and val_unseen pooled: " if f1 else "the mean of val and val_unseen ") if "val_unseen" in e else "val "
    return both + SCORE["f1" if f1 else "map50"][1] + " " + LABEL[variants(a)[0]]


def early_stop(scores: list, min_delta: float) -> tuple:
    """the index of the best pass in a curve of val scores, a pass counting as better only when it beats the best so far by at
    least min_delta, and how many passes have run since it; (None, 0) for an empty curve"""
    best = None
    for i, s in enumerate(scores):
        if best is None or s >= scores[best] + min_delta - 1e-9:
            best = i
    return best, 0 if best is None else len(scores) - 1 - best


def patience_stop(a, hist: dict, epoch: int) -> str | None:
    """why to stop before this epoch: --patience val passes in a row without an improvement of --min-delta (0 never stops)"""
    c = curve(a, hist); best, since = early_stop([s for _, s in c], a.min_delta)
    if a.patience and c and since >= a.patience:
        return (f"early stop after epoch {epoch} of {a.epochs}: {since} val pass{'es' if since > 1 else ''} in a row without an improvement of {a.min_delta:g} in "
                f"{watched(a, hist)}; the best pass was epoch {c[best][0]} at {r4(c[best][1])}")
    return None


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


def presentations(ds: Tiles, rc: dict, a) -> dict:
    """what repeat-factor sampling does to the first epoch: per class its factor, the tiles holding it and how many of the first
    epoch's draws hold it"""
    held = [{x["category_id"] for x in ds.anns[im["id"]]} for im in ds.images]; idx = repeat_indices(ds.factors, a.seed, 0)
    plain, shown = Counter(c for h in held for c in h), Counter(c for i in idx for c in held[i % len(held)])
    return {"t": a.rare_repeat, "epoch_tiles": len(idx), "split_tiles": len(held),
            "per_class": {CLASSES[c]: {"factor": r4(rc[c]), "tiles": plain[c], "epoch1": shown[c]} for c in sorted(rc, key=lambda c: -rc[c])}}


def base_of(model: str) -> str:
    """the hub checkpoint a run descends from: the model itself, or for a run from trained weights, the base those weights record"""
    p = os.path.join(model, "metrics.json")
    if not os.path.isfile(p):
        return model
    m = json.load(open(p))
    return m.get("base_model") or m.get("model") or model


def more_command(weights: str, g: dict) -> str:
    """the one line that gives a trained model N more passes: a new run from its weights with its own fresh, shorter and gentler
    schedule (half the learning rates, a 100-step warm-up, the same early stopping), into a new directory"""
    weights = weights.rstrip("/"); out = (weights[:-5] if weights.endswith("_full") else weights) + "_more"
    return (f"PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data {g['data']} --model {weights} --out {out} "
            f"--epochs N --batch {g['batch']} --lr {g['lr'] / 2:.0e} --backbone-lr {g['backbone_lr'] / 2:.0e} --warmup-steps 100 --patience {g.get('patience') or 3} "
            f"--min-delta {g.get('min_delta', 0.002):g} --nms-iou {g.get('nms_iou', 0.5):g} --device {g['device']} --seed {g['seed']}")


def cmd_train(a, argv: list):
    setup_log(os.path.join(a.out, "train.log")); t_run = time.time(); cmd = command(argv); set_seed(a.seed)
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    if man.get("classes") != CLASSES:
        raise ValueError(f"{a.data}/manifest.json: classes differ from sld2quote/classes.py")
    if man.get("min_visible", a.min_visible) != a.min_visible:
        log.warning(f"--min-visible {a.min_visible} differs from the export's min_visible {man['min_visible']}: zoomed boxes follow another rule")
    if a.patience and not a.eval_every:
        raise SystemExit("--patience watches the val score, so it needs --eval-every")
    best_dir = os.path.join(a.out, "checkpoint-best")
    if not a.resume:
        shutil.rmtree(best_dir, ignore_errors=True)  # a fresh run never inherits another run's best pass
    ck = os.path.join(a.out, "checkpoint-last"); hist, start = {"command": cmd, "args": vars(a), "steps": [], "epochs": []}, 0
    if a.resume:
        if not os.path.isdir(ck):
            raise SystemExit(f"--resume: no checkpoint at {ck}")
        st = torch.load(os.path.join(ck, "trainer_state.pt"), map_location="cpu", weights_only=True)
        hist, start = st["history"], st["epoch"] + 1; a.model = hist["args"]["model"]; hist.setdefault("resumed", []).append(cmd)
    src = ck if a.resume else a.model
    model, proc = load_model(src).to(a.device), load_processor(src)
    ds = Tiles(a.data, "train", proc, aug=(a.brightness, a.contrast, a.scale_jitter, a.min_visible), limit=a.limit, seed=a.seed)
    if len(ds) < a.batch:
        raise SystemExit(f"{len(ds)} training tiles make no full batch of {a.batch}")
    if a.rare_repeat:
        ds.factors, rc = repeat_factors(ds, a.rare_repeat); hist["rare_repeat"] = presentations(ds, rc, a)
        log.info(f"repeat-factor sampling at T {a.rare_repeat:g}: an epoch shows about {hist['rare_repeat']['epoch_tiles']} tiles instead of {len(ds)}; "
                 "per class factor, tiles holding it and presentations in the first epoch: " +
                 ", ".join(f"{n} x{v['factor']} {v['tiles']} -> {v['epoch1']}" for n, v in hist["rare_repeat"]["per_class"].items() if v["factor"] > 1))
    steps = [(len(repeat_indices(ds.factors, a.seed, e)) if ds.factors else len(ds)) // a.batch for e in range(a.epochs)]  # each draw is known
    opt = optimiser(model, a.lr, a.backbone_lr, a.weight_decay); sched = get_cosine_schedule_with_warmup(opt, a.warmup_steps, sum(steps))
    if a.resume:
        opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"]); log.info(f"resumed {a.model} from {ck}: {start} epochs done")
    log.info(f"{cmd}\n{src} on {a.device}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters; {len(ds)} train tiles, "
             f"{steps[0]} steps in the first epoch, {sum(steps)} in all; versions {versions()}")
    hist["stopped"] = None
    for epoch in range(start, a.epochs):
        if stop := over_budget(a, hist, epoch) or patience_stop(a, hist, epoch):
            hist["stopped"] = stop; log.info(stop); break
        t_ep = time.time(); loss, ips = train_epoch(model, opt, sched, ds, a, epoch, hist)
        rec = {"epoch": epoch + 1, "train_loss": r4(loss), "images_per_s": r4(ips)}
        if a.eval_every and (epoch + 1) % a.eval_every == 0:
            sel = [s for s in SELECT if os.path.isdir(os.path.join(a.data, s))]; ts, m, chosen = evaluate(model, proc, a, sel)
            rec |= {s: {v: {k: m[v][s][k] for k in VAL_KEYS} | {"threshold": ts[v]} for v in m} for s in sel} | {"selection": chosen}
        rec["seconds"] = round(time.time() - t_ep, 1); hist["epochs"].append(rec)
        c = curve(a, hist); best, since = early_stop([s for _, s in c], a.min_delta)
        if c and not since and c[best][0] == epoch + 1:  # this pass is the best so far: keep its weights
            hist["best_epoch"], hist["best_score"] = c[best][0], r4(c[best][1]); save_dir(best_dir, model, proc)
        if a.keep_passes:  # every pass's weights, so another selection rule can be applied without training again
            save_dir(os.path.join(a.out, f"pass-{epoch + 1}"), model, proc)
        save_checkpoint(model, proc, opt, sched, epoch, hist, a.out); json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
        log.info(f"epoch {epoch + 1} done in {rec['seconds'] / 60:.1f} min: train loss {rec['train_loss']:.4f}, {ips:.2f} images/s, val {rec.get('val')}; "
                 f"best pass so far: epoch {hist.get('best_epoch')}")
    if hist.get("best_epoch") and os.path.isdir(best_dir):  # the final weights are the best pass's, not the last
        model = load_model(best_dir).to(a.device)
        log.info(f"final weights: the best pass, epoch {hist['best_epoch']} ({watched(a, hist)} {hist['best_score']}), from {best_dir}")
    t_ev = time.time(); vs = variants(a)
    ts, out, chosen = evaluate(model, proc, a, [s for s in EVAL_SPLITS if os.path.isdir(os.path.join(a.data, s))])
    model.config.update({"score_threshold": ts[vs[0]], "score_threshold_raw": ts["raw"], "nms_iou": a.nms_iou})  # they travel with the weights
    model.save_pretrained(a.out); proc.save_pretrained(a.out)
    metrics = {"model": a.model, "base_model": base_of(a.model), "weights": a.out, "data": a.data, "manifest_sha256": sha256(os.path.join(a.data, "manifest.json")),
               "device": a.device, "nms_iou": a.nms_iou, "threshold": ts[vs[0]], "match_iou": MATCH_IOU, "selection": watched(a, hist), "selection_scores": chosen,
               "threshold_chosen_on": " and ".join(s for s in SELECT if s in out[vs[0]]) + (" pooled" if "val_unseen" in out[vs[0]] else ""),
               "epochs_cap": a.epochs, "epochs_run": len(hist["epochs"]), "best_epoch": hist.get("best_epoch"), "best_score": hist.get("best_score"),
               "patience": a.patience, "min_delta": a.min_delta, "stopped": hist["stopped"], "rare_repeat": a.rare_repeat, "repeat_sampling": hist.get("rare_repeat"),
               "train_images_per_s": r4(statistics.mean(e["images_per_s"] for e in hist["epochs"])) if hist["epochs"] else None,
               "train_minutes": r4(sum(e["seconds"] for e in hist["epochs"]) / 60), "final_eval_minutes": r4((time.time() - t_ev) / 60),
               "variants": {v: {"threshold": ts[v], "splits": out[v]} for v in vs}}
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1); json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
    open(os.path.join(a.out, "card.md"), "w").write(card(a, man, metrics, hist))
    log.info(f"wrote {a.out}: weights, metrics.json, card.md, history.json; final evaluation {metrics['final_eval_minutes']:.1f} min, "
             f"{(time.time() - t_run) / 60:.1f} min in all")


def card(a, man: dict, metrics: dict, hist: dict) -> str:
    """the model card: what was trained on what, how, and how well, with NMS and raw"""
    V = metrics["variants"]; run = f"{metrics['epochs_run']} of a cap of {a.epochs}" + (f" ({hist['stopped']})" if hist["stopped"] else ", ended by the cap")
    best = (f"final weights from the best pass, epoch {metrics['best_epoch']} ({metrics['selection']} {metrics['best_score']})"
            if metrics["best_epoch"] else "final weights from the last pass")
    per = statistics.mean(e["seconds"] for e in hist["epochs"]) / 60 if hist["epochs"] else 0
    counts = lambda c: ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in c.items() if isinstance(v, (int, float)))
    post = (f"class-agnostic NMS at IoU {metrics['nms_iou']} on each tile's detections, the operating post-processing, and raw" if metrics["nms_iou"] else "raw")
    L = [f"# sld2quote symbol detector: {a.model} fine-tuned on synthetic tiles", "",
         "The data is synthetic: every tile was cut from a sheet drawn by `sld2quote/generate.py` and every label comes from the "
         "generator's answer key. No real drawing was used for training or testing.", "",
         f"- Model: `{a.model}`" + (f", trained weights whose base is `{metrics['base_model']}`" if metrics["base_model"] != a.model else "") +
         f" (RT-DETR, Hugging Face transformers), {len(CLASSES)} classes as in `sld2quote/classes.py`: {', '.join(CLASSES)}",
         f"- Dataset: `{a.data}`, manifest.json sha256 `{metrics['manifest_sha256']}`, exported {man.get('created', '-')}; tiles {man.get('tile')} px, "
         f"overlap {man.get('overlap')}, a box kept in a tile from {man.get('min_visible')} of its symbol; held out of training: {json.dumps(man.get('held_out'))}",
         *[f"  - {s}: {counts(c)}" for s, c in man.get("splits", {}).items()],
         f"- Command: `{hist['command']}`" + "".join(f"; resumed with `{c}`" for c in hist.get("resumed", [])),
         f"- Seed {a.seed}; device {a.device}; epochs run {run}; {best}; early stopping with patience {a.patience} (0 is off) and min delta {a.min_delta:g}; "
         f"{metrics['train_images_per_s']} training images/s; {metrics['train_minutes']:.0f} min of training, about {per:.0f} min per pass with its val",
         *([f"- Rare-class sampling: repeat factors by the LVIS rule at T {a.rare_repeat:g}, r(c) = max(1, sqrt(T / f(c))) with f(c) the share of "
            f"training tiles holding class c, a tile drawn by its largest factor; an epoch shows about {rs['epoch_tiles']} tiles instead of {rs['split_tiles']}; "
            "per class factor, tiles holding it, presentations in the first epoch: " + ", ".join(f"{n} x{v['factor']} ({v['tiles']} -> {v['epoch1']})"
                                                                                        for n, v in rs["per_class"].items() if v["factor"] > 1)]
           if (rs := metrics.get("repeat_sampling")) else []),
         "- Libraries: " + ", ".join(f"{k} {v}" for k, v in versions().items()),
         f"- Post-processing: {post}; each at its own threshold, chosen on {metrics['threshold_chosen_on']} to maximise the mean per-class F1 ("
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
    L += ["", "Full metrics in `metrics.json`, training curve in `history.json`.", "",
          f"More passes: a new run from these weights with its own fresh, shorter schedule, N passes of about {per:.0f} min each here, "
          f"into a new directory (this one stays as it is): `{more_command(a.out, vars(a))}`", "",
          "Licence: Apache-2.0, as the pretrained RT-DETR weights."]
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
    sel = [s for s in SELECT if os.path.isdir(os.path.join(a.data, s))]
    ts, out, _ = evaluate(model, proc, a, [a.split] if ts else list(dict.fromkeys(sel + [a.split])), ts)
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


def latest(d: str, m: dict) -> tuple[dict, str]:
    """the run's metrics per variant and split, each split from its latest eval (eval-<split>.json) where there is one, and a line on
    where they come from: an eval on the export the run trained on must reproduce the training-time metrics (timings aside); one on
    a newer export, with another manifest, replaces them"""
    same = lambda x, y: {k: v for k, v in x.items() if k != "images_per_s"} == {k: v for k, v in y.items() if k != "images_per_s"}
    V = {v: {"threshold": r["threshold"], "splits": dict(r["splits"])} for v, r in m["variants"].items()}
    kept, differs, newer = [], [], []
    for s in dict.fromkeys(list(EVAL_SPLITS) + list(next(iter(V.values()))["splits"])):
        p = os.path.join(d, f"eval-{s}.json")
        e = json.load(open(p)) if os.path.isfile(p) else {}
        if not all(s in e.get("variants", {}).get(v, {}).get("splits", {}) for v in V):
            continue
        ev = {v: e["variants"][v]["splits"][s] for v in V}
        if e.get("manifest_sha256") != m.get("manifest_sha256") or s not in next(iter(V.values()))["splits"]:
            newer.append(f"{s} ({next(iter(ev.values()))['images']} tiles, export {e['manifest_sha256'][:8]})")
        else:
            (kept if all(same(ev[v], V[v]["splits"][s]) for v in V) else differs).append(s)
        for v in V:
            V[v]["splits"][s] = ev[v]
    parts = ([f"eval of {', '.join(kept)} after training reproduces the training-time metrics exactly"] if kept else []) + \
            ([f"eval of {', '.join(differs)} differs from the training-time metrics on the same export"] if differs else []) + \
            ([f"evaluated after training on a newer export, at the thresholds chosen then, and shown so: {', '.join(newer)}"] if newer else [])
    return V, "; ".join(parts) or "eval was not run after training"


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
    (V, note), g = latest(d, m), h["args"]; op = next(iter(V)); S = V[op]["splits"]; base = m.get("base_model") or m["model"]
    name = os.path.basename(base) + (f" continued from {os.path.basename(m['model'])}" if base != m["model"] else "")
    parent = os.path.join(m["model"], "metrics.json")  # a run from trained weights is shown beside the run it continues
    pm = json.load(open(parent)) if base != m["model"] and os.path.isfile(parent) else None
    L = [f"## {name} (`{d}`)", ""] + ([f"- Continues `{m['model']}`, whose best pass scored val mAP@0.5 {LABEL[op]} {pm['variants'][op]['splits']['val']['map50']} "
                                       f"and test_unseen {pm['variants'][op]['splits']['test_unseen']['map50']}"] if pm and op in pm.get("variants", {}) else []) + [
         f"- {m['epochs_run']} pass{'es' if m['epochs_run'] != 1 else ''} of a cap of {g['epochs']} at batch {g['batch']}, lr {g['lr']:.0e} (backbone {g['backbone_lr']:.0e}), warm-up "
         f"{g['warmup_steps']} steps, seed {g['seed']}, early stopping patience {g.get('patience', 0)}" +
         (f", rare-class sampling at T {g['rare_repeat']:g}" if g.get("rare_repeat") else "") + ": " + (m["stopped"] if m["stopped"] else "ended by the cap"),
         "- Scored with " + (f"the best pass's weights, epoch {m['best_epoch']} ({m.get('selection', 'val mAP@0.5 ' + LABEL[op])} {m.get('best_score', m.get('best_val_map50'))})" if m.get("best_epoch")
                             else "the last pass's weights") +
         f"; {m['train_minutes']:.0f} min of training and {m['final_eval_minutes']:.1f} min of final evaluation, {m['train_images_per_s']} training images/s",
         f"- Thresholds chosen on {m.get('threshold_chosen_on', 'val')} at training time: " + ", ".join(f"{LABEL[v]} {r['threshold']:.2f}" for v, r in V.items()) +
         f"; NMS is class-agnostic at IoU {m['nms_iou']}",
         f"- {note}", "",
         "| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {s} | {r['splits'][s]['images']} | {LABEL[v]} | {fmt(r['splits'][s]['map50'])} | {fmt(r['splits'][s]['mean_precision'])} | {worst(r['splits'][s], 'precision')} | "
          f"{fmt(r['splits'][s]['mean_recall'])} | {worst(r['splits'][s], 'recall')} |" for s in S for v, r in V.items()]
    if "val" in S and "test_unseen" in S:
        rule = size_rule(S["val"], S["test_unseen"], target); fails = [c for c, x in rule.items() if x == "fail"]
        L += ["", f"Per class {LABEL[op]}, and the size rule (precision and recall at least {target} on val and on test_unseen):", "",
              "| class | val P | val R | test_unseen P | test_unseen R | rule |", "|---|---|---|---|---|---|"]
        L += [f"| {c} | " + " | ".join(fmt(S[s]["per_class"][c][k]) for s in ("val", "test_unseen") for k in ("precision", "recall")) + f" | {rule[c]} |" for c in CLASSES]
        na = sum(x == "n/a" for x in rule.values())
        L += ["", f"Size rule: {sum(x == 'pass' for x in rule.values())} of {len(CLASSES)} classes pass" +
              (f", {na} cannot be judged (no boxes in val or test_unseen)" if na else "") + "; " + (f"fails on {', '.join(fails)}." if fails else "no class fails.")]
    vu, sc = any("val_unseen" in e for e in h["epochs"]), any("selection" in e for e in h["epochs"])
    at = lambda e, s, v: fmt(e.get(s, {}).get(v, {}).get("map50")); pick = lambda e, k: fmt(e.get("selection", {}).get(op, {}).get(k))
    L += ["", "Training curve, val after each epoch:", "", "| epoch | train loss | val mAP@0.5 with NMS |" + (" val_unseen mAP@0.5 with NMS |" if vu else "") +
          (" pooled mean F1 | val_unseen worst min(P, R) |" if sc else "") + " val mAP@0.5 raw | images/s | minutes |",
          "|---|---|---|" + ("---|" if vu else "") + ("---|---|" if sc else "") + "---|---|---|"]
    L += [f"| {e['epoch']}{' (best)' if e['epoch'] == m.get('best_epoch') else ''} | {e['train_loss']} | {at(e, 'val', 'nms')} |" + (f" {at(e, 'val_unseen', 'nms')} |" if vu else "") +
          (f" {pick(e, 'pooled_mean_f1')} | {pick(e, 'val_unseen_worst_min_pr')} |" if sc else "") +
          f" {at(e, 'val', 'raw')} | {e['images_per_s']} | {e['seconds'] / 60:.1f} |" for e in h["epochs"]]
    if slow := slowed(h):
        L += ["", f"Pass{'es' if len(slow) > 1 else ''} {', '.join(map(str, slow))} ran below {SLOWED:.0%} of the run's best rate "
              f"({max(e['images_per_s'] for e in h['epochs'])} images/s), slowed by other work on the machine: those minutes are contended."]
    return L + ["", "What it still gets wrong: " + plain_words(name, V), ""]


def slowed(h: dict) -> list:
    """the passes that ran below SLOWED of the run's best training rate: the machine was shared, so their minutes are no measure"""
    best = max((e["images_per_s"] for e in h["epochs"]), default=0)
    return [e["epoch"] for e in h["epochs"] if e["images_per_s"] < SLOWED * best]


def more_passes(runs: list) -> list:
    """the summary's last section: per model, the time per pass measured in its run and the one line that orders more passes"""
    L = ["## More passes", "",
         "To give a model more passes than its cap, start a new run from its trained weights with its own fresh, shorter and gentler schedule "
         "(half the learning rates, a 100-step warm-up, the same early stopping). It writes a new directory with its own card and metrics, scored "
         "with its best pass, and leaves the overnight run as it is. Replace N with the number of passes:", ""]
    for d in runs:
        if os.path.isfile(os.path.join(d, "metrics.json")):
            m, h = json.load(open(os.path.join(d, "metrics.json"))), json.load(open(os.path.join(d, "history.json")))
            per = (min if slowed(h) else statistics.mean)(e["seconds"] for e in h["epochs"]) / 60
            L += [f"- `{d}` ({os.path.basename(m.get('base_model') or m['model'])}): one pass with its val took about {per:.0f} min tonight" +
                  (" at its fastest (other passes were slowed by other work)" if slowed(h) else "") + f", so N passes take about N times {per:.0f} min, "
                  f"then about {m['final_eval_minutes']:.0f} min of final evaluation:", "", f"  `{more_command(d, h['args'])}`", ""]
    return L


def cmd_summary(a, argv: list):
    runs = a.runs.split(",")
    L = ["# Overnight detector runs", "",
         "Synthetic data only: every tile was cut from a sheet drawn by `sld2quote/generate.py`. Scores are per 640 px tile, so a symbol in two "
         "tiles counts twice. \"With NMS\" is the operating post-processing (class-agnostic duplicate suppression); \"raw\" is without it; each "
         "uses its own threshold, chosen at training time on val (and val_unseen, pooled, for runs that had it), which also picks the best "
         "pass; test_unseen is never used for choosing, only for the final numbers and the size rule.", ""]
    for d in runs:
        L += summarise(d, a.target)
    open(a.out, "w").write("\n".join(L + more_passes(runs)) + "\n"); print(f"wrote {a.out}")


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
    t.add_argument("--out", default="weights/rtdetr_r18")
    t.add_argument("--model", default="PekingU/rtdetr_r18vd", help="a hub checkpoint, or a trained weights directory to give a model more passes "
                   "with a fresh schedule of its own (its 14-class head is kept); the card's More passes line has the command")
    t.add_argument("--epochs", type=int, default=3); t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--backbone-lr", type=float, default=1e-5)
    t.add_argument("--weight-decay", type=float, default=1e-4); t.add_argument("--warmup-steps", type=int, default=200); t.add_argument("--grad-clip", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint-last")
    t.add_argument("--time-budget-min", type=float, help="training minutes, summed over resumes: stop at the first epoch boundary where one more "
                   "epoch would pass it; the final evaluation runs after")
    t.add_argument("--eval-every", type=int, default=1, help="val metrics every N epochs, 0 for none"); t.add_argument("--log-every", type=int, default=20, help="steps")
    t.add_argument("--patience", type=int, default=0, help="stop after N val passes in a row that do not raise val mAP@0.5 (with NMS, or raw "
                   "when NMS is off) by --min-delta; 0 never stops early. Whenever val runs, checkpoint-best keeps the best pass and the final "
                   "weights, evaluation, metrics and card use it")
    t.add_argument("--min-delta", type=float, default=0.002, help="the smallest rise in the watched val score that counts as an improvement")
    t.add_argument("--select", choices=sorted(SCORE), default="map50", help="the score that picks the best pass, averaged over val and val_unseen: "
                   "map50 is mAP@0.5, f1 the mean per-class F1 at the operating threshold")
    t.add_argument("--keep-passes", action="store_true", help="also keep every pass's weights in <out>/pass-<n>")
    t.add_argument("--rare-repeat", type=float, default=0, help="repeat-factor sampling threshold T, the LVIS rule: a class in a share f of the "
                   "training tiles gets factor max(1, sqrt(T / f)), a tile its largest class factor, and each epoch draws every tile its factor "
                   "times in expectation (seeded, so the epoch is a little longer than the split); 0 is off")
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
