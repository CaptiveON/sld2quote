"""The trained reader's operating threshold, chosen on whole validation drawings, never per tile and never on a test set: the
detector and reader_v2's cross-tile merge only (no OCR, no roles) over the val and val_unseen sheets (evaluate --make-sets),
class-level precision and recall at IoU >= 0.3 (evaluate's rule) for every threshold from 0.05 to 0.60 in steps of 0.01,
and the threshold with the best mean per-class F1 pooled over the sets (the trainer's rule per tile, applied to drawings).
It goes into the weights' config.json as score_threshold_drawing, beside the per-tile score_threshold, with the sweep in
threshold_drawing.json; reader_v2 prefers it. A false positive whose box holds almost no ink is counted as on blank paper.
The same sweep gives score_calibration: for score bins from the threshold up, how often a box was right on those drawings,
which reader_v2 shows as a box's confidence; --calibrate-only rebuilds it from a stored threshold_drawing.json.

  python3 -m sld2quote.select_threshold --weights weights/rtdetr_r18_full --sets data/eval/val,data/eval/val_unseen --device cpu
  python3 -m sld2quote.select_threshold --weights weights/rtdetr_r18_full --calibrate-only
"""
from __future__ import annotations
import argparse, glob, json, os, time
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from . import reader, reader_v2 as rv2
from .classes import DEVICE_TO_CLASS
from .evaluate import iou

THRESHOLDS = [round(float(t), 2) for t in np.arange(0.05, 0.6001, 0.01)]
MATCH_IOU = 0.3   # a detection finds an answer of its class at this overlap, as evaluate.py matches devices
BLANK_INK = 0.01  # a false positive whose box is less than this share ink sits on blank paper
MIN_BOXES = 100   # a calibration bin holds at least this many boxes


def sheet(png: str, model, proc, device: str, nms_iou: float) -> tuple:
    """one sheet as the sweep needs it: the prepared page's tile detections at the lowest threshold, the answer key by class
    (drawing frame, which the deskewed page shares) and the page's ink"""
    g, ink, _ = reader.prepare(png)
    grid, results, size = rv2.tile_detections(Image.fromarray(g.astype(np.uint8)), model, proc, device, THRESHOLDS[0], nms_iou)
    gt = json.load(open(png[:-4] + ".json"))
    return grid, [{k: v.cpu() for k, v in r.items()} for r in results], size, [(DEVICE_TO_CLASS[d["type"]], d["box"]) for d in gt["devices"]], ink


def score(symbols: list, gts: list, ink) -> tuple:
    """per class tp, pred and gt (best score first, a detection takes the unmatched answer of its class it overlaps most at
    IoU >= MATCH_IOU), and how many of the unmatched detections sit on blank paper"""
    c, used, blank = defaultdict(Counter), set(), 0
    for k, _ in gts:
        c[k]["gt"] += 1
    for s in sorted(symbols, key=lambda s: -s["shape_score"]):
        c[s["kind"]]["pred"] += 1
        best, bj = MATCH_IOU, None
        for j, (k, box) in enumerate(gts):
            if j not in used and k == s["kind"] and (v := iou(s["box"], box)) >= best:
                best, bj = v, j
        if bj is not None:
            used.add(bj); c[s["kind"]]["tp"] += 1
        else:
            x, y, w, h = s["box"]; region = ink[max(0, y): y + h, max(0, x): x + w]
            blank += int(bool(region.size) and region.mean() < BLANK_INK)
    return c, blank


def sweep(sheets: list, thresholds: list = THRESHOLDS) -> dict:
    """per threshold: the per-class counts pooled over the sheets, the mean per-class F1 over the classes in the answer key,
    and the false positives on blank paper; each sheet's tiles are cut at the threshold and merged as reader_v2 merges them"""
    table = {}
    for t in thresholds:
        tot, blank = defaultdict(Counter), 0
        for grid, results, size, gts, ink in sheets:
            c, b = score(rv2.to_symbols(rv2.merge(grid, [{k: v[r["scores"] >= t] for k, v in r.items()} for r in results], size)), gts, ink)
            blank += b
            for k, n in c.items():
                tot[k].update(n)
        f1 = [2 * n["tp"] / (n["pred"] + n["gt"]) for n in tot.values() if n["gt"]]
        table[t] = {"mean_f1": round(sum(f1) / len(f1), 4) if f1 else 0.0, "blank_fp": blank, "classes": {k: dict(n) for k, n in sorted(tot.items())}}
    return table


def choose(table: dict) -> float:
    """the threshold with the best mean per-class F1; a tie goes to the middle of the tied thresholds, as the trainer chooses"""
    ts = sorted(table); m = max(table[t]["mean_f1"] for t in ts)
    best = [t for t in ts if table[t]["mean_f1"] >= m - 1e-9]
    return best[len(best) // 2]


def calibration(table: dict, start: float) -> list:
    """[[lower_edge, boxes, right], ...]: score bins from start upward, one between each pair of sweep thresholds and the last
    holding every box above the top one; a bin's boxes and right boxes are the differences of the pooled counts (a box is
    only ever removed by a better-scoring one, so the sets nest); bins merged upward to hold MIN_BOXES or more, then pooled
    where precision would fall (pool-adjacent-violators), so a higher score never means a lower confidence"""
    ts = [t for t in sorted(table) if t >= start - 1e-9]
    tot = [(sum(c.get("pred", 0) for c in table[t]["classes"].values()), sum(c.get("tp", 0) for c in table[t]["classes"].values())) for t in ts] + [(0, 0)]
    bins = []
    for t, (n, r), (n2, r2) in zip(ts, tot, tot[1:]):
        if bins and bins[-1][1] < MIN_BOXES:
            bins[-1][1] += n - n2; bins[-1][2] += r - r2
        else:
            bins.append([t, n - n2, r - r2])
    if len(bins) > 1 and bins[-1][1] < MIN_BOXES:  # a short top bin joins the one below
        e, n, r = bins.pop(); bins[-1][1] += n; bins[-1][2] += r
    pooled = []
    for b in bins:
        pooled.append(b)
        while len(pooled) > 1 and pooled[-2][2] * pooled[-1][1] > pooled[-1][2] * pooled[-2][1]:  # precision falls: pool
            e, n, r = pooled.pop(); pooled[-1][1] += n; pooled[-1][2] += r
    return [b for b in pooled if b[1]]


def write(weights: str, chosen: float, table: dict, record: dict) -> list:
    """the sweep and its calibration bins into threshold_drawing.json, then score_threshold_drawing and score_calibration
    ([[lower_edge, precision], ...]) into config.json; each is written aside and swapped in, so the weights never hold a
    half-written file nor a threshold without the sweep behind it"""
    bins = calibration(table, chosen)
    path = os.path.join(weights, "config.json"); cfg = json.load(open(path))
    cfg.update(score_threshold_drawing=chosen, score_calibration=[[e, round(r / n, 4)] for e, n, r in bins])
    for dst, text in ((os.path.join(weights, "threshold_drawing.json"), json.dumps({**record, "chosen": chosen, "calibration": bins, "sweep": {str(t): v for t, v in table.items()}}, indent=1)),
                      (path, json.dumps(cfg, indent=2, sort_keys=True) + "\n")):
        with open(dst + ".tmp", "w") as fh:
            fh.write(text)
        os.replace(dst + ".tmp", dst)
    return bins


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m sld2quote.select_threshold", description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sets", default="data/eval/val,data/eval/val_unseen", help="comma-separated validation sets; each folder name must start with val")
    ap.add_argument("--device", help="cpu or mps (default mps when available, else cpu)")
    ap.add_argument("--calibrate-only", action="store_true", help="rebuild score_calibration from the stored threshold_drawing.json, no new sweep")
    a = ap.parse_args(argv)
    if a.calibrate_only:
        d = json.load(open(os.path.join(a.weights, "threshold_drawing.json"))); d.pop("calibration", None)
        bins = write(a.weights, d.pop("chosen"), {float(t): v for t, v in d.pop("sweep").items()}, d)
        print(f"score_calibration for {a.weights} (lower edge, boxes, right, precision):", [[e, n, r, round(r / n, 4)] for e, n, r in bins])
        return
    sets = [s for s in a.sets.split(",") if s]
    if not sets or any(not os.path.basename(os.path.normpath(s)).startswith("val") for s in sets):
        raise SystemExit(f"--sets {a.sets}: a threshold is chosen on validation sets only (folders named val...), never on a test set")
    device = a.device or rv2.default_device(); model, proc = rv2.load(a.weights, device); t0 = time.time()
    pngs = [p for s in sets for p in sorted(glob.glob(os.path.join(s, "sheet-*.png")))]
    if not pngs:
        raise SystemExit(f"--sets {a.sets}: no sheets; build them with python3 -m sld2quote.evaluate --make-sets")
    table = sweep([sheet(p, model, proc, device, model.config.nms_iou) for p in pngs])
    chosen, tile = choose(table), getattr(model.config, "score_threshold", None)
    near = min(table, key=lambda t: abs(t - tile)) if tile is not None else None
    record = {"weights": a.weights, "sets": sets, "sheets": len(pngs), "device": device, "match_iou": MATCH_IOU, "blank_ink": BLANK_INK,
              "per_tile_threshold": tile, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    bins = write(a.weights, chosen, table, record)
    for label, t in (("chosen on drawings", chosen), ("per tile", near)):
        if t is not None:
            print(f"{label}: threshold {t}, mean class F1 {table[t]['mean_f1']}, false positives on blank paper {table[t]['blank_fp']}")
    print(f"wrote score_threshold_drawing {chosen} and score_calibration {[[e, round(r / n, 4)] for e, n, r in bins]} into {a.weights}/config.json "
          f"and the sweep into {a.weights}/threshold_drawing.json ({len(pngs)} sheets, {record['seconds']} s)")


if __name__ == "__main__":
    main()
