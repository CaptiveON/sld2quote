"""The site's chart data: eval/charts.json, built from what exists on this Mac (the scorecards in eval/RESULTS.md, the
training histories and the threshold sweep under weights/, eval/OCR.md, the precomputed samples, the COCO export).

  python3 scripts/build_charts.py [--model rtdetr_r18_plain_ema_fix_pass4] [--out eval/charts.json]

--model is the deployed directory under weights/: a run, or a pass of one copied as <run>_passN with its own config.json and
threshold_drawing.json (the training record stays with the run, which is then the live one, with live_pass N).
Keys stay stable; a section is left out when its source is missing. Finished training runs only (metrics.json exists).
validate(doc) checks the shape and is what tests/test_site.py runs on the written file."""
import argparse, glob, json, os, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from sld2quote.classes import CLASSES  # noqa: E402
from sld2quote.evaluate import TYPES  # noqa: E402
from sld2quote.result import TYPE_PLAIN  # noqa: E402

SETS = {"test": "data/eval/test", "unseen": "data/eval/unseen", "baseline": "data/baseline"}
COPIES = ("clean", "scan")
METRICS = {"sym_recall": "symbol_recall", "sym_precision": "symbol_precision", "rating_acc": "rating_accuracy", "bom_recall": "bom_recall",
           "price_error_mean": "price_error", "sheets": "n_sheets", "link_precision": "link_precision", "link_recall": "link_recall"}
TITLE = re.compile(r"^(data/\S+), (trained|classical) reader(?: \(`weights/([^`]+)`\))?(?:, (\S+) OCR, crops (on|off))?")
CLASS_PLAIN = {"breaker": "circuit breaker", "fuse_switch": "fuse switch", "isolator": "isolator", "contactor": "contactor (motor starter)",
               "transformer": "transformer", "generator": "generator", "motor": "motor", "board": "downstream board", "ct": "current transformer",
               "meter": "panel meter", "relay": "protection relay", "spd": "surge protector", "capacitor": "capacitor bank", "earth": "earth"}
RUN_LABELS = {"rtdetr_r18_full": "r18, full set", "rtdetr_r50_full": "r50, full set", "rtdetr_r18_more": "r18, continued from r18 full",
              "rtdetr_r18_rare": "r18, rare classes repeated", "rtdetr_r18_plain": "r18, plain control", "rtdetr_r18_plain_ema": "r18, plain control with EMA",
              "rtdetr_r18_fonts": "r18, wide font set", "rtdetr_r18_fonts_ema": "r18, wide font set with EMA", "rtdetr_r18_plain_ema_fix": "r18, plain control with EMA and the gradient fix"}
OCR_LABELS = {"vision": "Apple Vision", "rapidocr": "RapidOCR (PP-OCRv4, English recogniser)", "rapidocr-ch": "RapidOCR (bundled Chinese recogniser)",
              "tesseract": "Tesseract", "easyocr": "EasyOCR", "doctr": "docTR"}
KINDS = {(True, "clean"): "synthetic clean", (True, "scan"): "synthetic scan", (False, "clean"): "real drawing", (False, "scan"): "real drawing"}


def r3(v):
    return round(float(v), 3)


def tables(text: str) -> list:
    """every markdown table in text as (header cells, rows of cells)"""
    out, cur = [], None
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = (cells, [])
            elif not all(set(c) <= set("-: ") for c in cells):
                cur[1].append(cells)
        else:
            if cur:
                out.append(cur)
            cur = None
    return out + ([cur] if cur else [])


def sections(path: str) -> dict:
    """RESULTS.md by section title (without the ##)"""
    head, *parts = re.split(r"(?m)^(?=## )", open(path).read())
    return {p.split("\n", 1)[0][3:].strip(): p for p in parts}


def scorecard(text: str) -> tuple:
    """a RESULTS.md section: metrics {key: (clean, scan)}, per-type rows, per-sheet rows"""
    metrics, per_type, sheets = {}, [], []
    for header, rows in tables(text):
        for cells in rows:
            if cells[0] in METRICS and len(cells) == 3:
                metrics[cells[0]] = (float(cells[1]), float(cells[2]))
            elif cells[0] in TYPES and len(cells) == 9:
                per_type.append(cells)
            elif cells[0].startswith("sheet-") and len(cells) >= 14:  # 16 since the link columns (19 Oct), added at the end
                sheets.append(cells)
    return metrics, per_type, sheets


def run_of(model: str) -> tuple:
    """the training run a deployed directory comes from and the pass copied, if any: rtdetr_r18_plain_ema_fix_pass4 -> (run, 4)"""
    m = re.fullmatch(r"(.+)_pass(\d+)", model)
    return (m.group(1), int(m.group(2))) if m else (model, None)


def sets_section(model: str) -> tuple:
    """sets, per_type and price_error from the scorecards: per set and reader the latest section in file order, the trained
    reader's among the site engine's (rapidocr) sections of the deployed run, so a rerun of a set replaces the earlier one"""
    run = run_of(model)[0]; dirs = {d: s for s, d in SETS.items()}; latest = {}
    for title, text in sections("eval/RESULTS.md").items():
        m = TITLE.match(title)
        if not m or m.group(1) not in dirs:
            continue
        reader, weights, engine, crops = m.group(2), m.group(3), m.group(4) or "vision", (m.group(5) or "on") == "on"  # sections before the engine naming ran with Vision, crops on
        if reader == "trained" and (weights != run or engine != "rapidocr"):
            continue
        latest[(dirs[m.group(1)], reader)] = (text, engine, crops)
    sets, per_type, price = [], [], []
    for name in SETS:
        for reader in ("trained", "classical"):
            if (name, reader) not in latest:
                continue
            text, engine, crops = latest[(name, reader)]
            metrics, types, sheets = scorecard(text)
            for i, copy in enumerate(COPIES):
                sets.append({"set": name, "copy": copy, "reader": reader, "engine": engine, "crops": crops,
                             **{out: (None if k not in metrics else int(metrics[k][i]) if k == "sheets" else r3(metrics[k][i])) for k, out in METRICS.items()}})
                if reader == "trained":
                    for c in types:  # P, R, n, rating per copy; found and reported recovered from the rounded ratios (exact at these counts)
                        p, rc, n = c[1 + 4 * i], c[2 + 4 * i], int(c[3 + 4 * i])
                        found = round(float(rc) * n) if rc != "-" else 0
                        per_type.append({"set": name, "copy": copy, "type": c[0], "plain": TYPE_PLAIN.get(c[0], c[0]), "total": n, "found": found,
                                         "reported": round(found / float(p)) if p != "-" and float(p) > 0 else 0})
                    price.append({"set": name, "copy": copy, "values": [r3(s[10]) for s in sheets if s[0].endswith("-scan.png") == (copy == "scan")]})
    return sets, per_type, price


def training(model: str) -> list:
    """one entry per finished run (metrics.json and a best pass), from its history.json; live is the deployed run"""
    live, live_pass = run_of(model); out = []
    for d in sorted(glob.glob("weights/*/")):
        run = os.path.basename(d.rstrip("/"))
        if run.startswith(("smoke", "aborted", "bench")) or not os.path.isfile(d + "metrics.json") or not os.path.isfile(d + "history.json"):
            continue
        h = json.load(open(d + "history.json")); ep = h["epochs"]
        if not ep or not h.get("best_epoch"):
            continue
        pooled = all(e.get("selection") for e in ep)
        out.append({"run": run, "label": RUN_LABELS.get(run, run), "live": run == live, "live_pass": live_pass if run == live else None, "ema": bool(h.get("args", {}).get("ema")),
                    "passes": [e["epoch"] for e in ep],
                    "score": [r3(e["selection"]["nms"]["pooled_mean_f1"] if pooled else e["val"]["nms"]["map50"]) for e in ep],
                    "metric": "pooled per-class F1 at the operating threshold on val and val_unseen" if pooled else "val mAP@0.5",
                    "train_loss": [r3(e["train_loss"]) for e in ep], "best_pass": h["best_epoch"]})
    return out


def threshold(model: str) -> tuple:
    """the drawing-level sweep and the calibration bins from threshold_drawing.json"""
    p = f"weights/{model}/threshold_drawing.json"
    if not os.path.isfile(p):
        return None, None
    t = json.load(open(p)); sw = {float(k): v for k, v in t["sweep"].items()}; ts = sorted(sw)
    cfg = f"weights/{model}/config.json"  # the deployed threshold, when the directory carries one
    chosen = json.load(open(cfg)).get("score_threshold_drawing", t["chosen"]) if os.path.isfile(cfg) else t["chosen"]
    tot = lambda th, k: sum(c.get(k, 0) for c in sw[th]["classes"].values())  # a count the sweep left out is zero
    sweep = {"thresholds": ts, "precision": [r3(tot(th, "tp") / max(1, tot(th, "pred"))) for th in ts], "recall": [r3(tot(th, "tp") / max(1, tot(th, "gt"))) for th in ts],
             "f1": [r3(sw[th]["mean_f1"]) for th in ts], "blank_paper_boxes": [sw[th]["blank_fp"] for th in ts], "chosen": chosen, "sheets": t["sheets"]}
    bins = t["calibration"]
    cal = [{"from": e, "to": bins[i + 1][0] if i + 1 < len(bins) else 1, "boxes": n, "precision": r3(right / n)} for i, (e, n, right) in enumerate(bins)]
    return sweep, cal


def ocr() -> list:
    """per engine, the page-pass table of eval/OCR.md (per copy: character accuracy on matched boxes, label strings, seconds)"""
    if not os.path.isfile("eval/OCR.md"):
        return None
    for header, rows in tables(open("eval/OCR.md").read()):
        if "copy" in header and "char acc (matched)" in header and "crops per sheet" not in header:
            col = {h: i for i, h in enumerate(header)}
            by = {}
            for c in rows:
                by.setdefault(c[0], {})[c[1]] = c
            return [{"engine": e, "label": OCR_LABELS.get(e, e), "clean_char": r3(v["clean"][col["char acc (matched)"]]), "scan_char": r3(v["scan"][col["char acc (matched)"]]),
                     "clean_label_char": r3(v["clean"][col["label char"]]), "scan_label_char": r3(v["scan"][col["label char"]]),
                     "seconds_per_sheet": r3((float(v["clean"][col["s per sheet"]]) + float(v["scan"][col["s per sheet"]])) / 2), "chosen": e == "rapidocr", "linux": e != "vision"}
                    for e, v in by.items() if "clean" in v and "scan" in v]
    return None


def timing() -> dict:
    """the five stages and the whole run, averaged over the precomputed samples"""
    rs = [json.load(open(p)) for p in sorted(glob.glob("site/samples/*/result.json"))]
    if not rs:
        return None
    stages = [{"step": s["key"], "label": s["title"], "mean_seconds": r3(sum(r["steps"][i]["seconds"] for r in rs) / len(rs))} for i, s in enumerate(rs[0]["steps"])]
    kinds = {}
    for r in rs:
        kinds.setdefault(KINDS[(bool(r["source"]["has_answer_key"]), r["source"].get("copy") or "clean")], []).append(r["timing"]["total"])
    return {"n": len(rs), "stages": stages, "by_kind": [{"kind": k, "n": len(v), "mean_total_seconds": r3(sum(v) / len(v))} for k, v in kinds.items()]}


def data() -> dict:
    """the COCO export: tiles and boxes per split, boxes per class in the training and held-out splits"""
    splits, counts = [], {c: {} for c in CLASSES}
    for p in sorted(glob.glob("data/coco/*/annotations.json")):
        split = os.path.basename(os.path.dirname(p)); d = json.load(open(p))
        splits.append({"split": split, "tiles": len(d["images"]), "boxes": len(d["annotations"])})
        for a in d["annotations"]:
            c = counts[CLASSES[a["category_id"]]]; c[split] = c.get(split, 0) + 1
    if not splits:
        return None
    classes = [{"type": c, "plain": CLASS_PLAIN[c], **{s: counts[c].get(s, 0) for s in ("train", "val", "val_unseen", "test_unseen")}} for c in CLASSES]
    return {"splits": splits, "classes": sorted(classes, key=lambda c: -c["train"])}


def build(model: str) -> dict:
    doc = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "engine": "rapidocr"}
    if os.path.isfile("eval/RESULTS.md"):
        doc["sets"], doc["per_type"], doc["price_error"] = sets_section(model)
    doc["training"] = training(model)
    doc["threshold"], doc["calibration"] = threshold(model)
    doc["ocr"], doc["timing"], doc["data"] = ocr(), timing(), data()
    return {k: v for k, v in doc.items() if v not in (None, [])}


def validate(doc: dict):
    """the shape the page relies on; raises AssertionError with the failing fact"""
    assert doc["model"] and doc["engine"] == "rapidocr"
    keys = {(s["set"], s["copy"], s["reader"]) for s in doc["sets"]}
    assert keys == {(n, c, r) for n in SETS for c in COPIES for r in ("trained", "classical")}, f"sets incomplete: {sorted(keys)}"
    assert all(s["n_sheets"] > 0 and 0 <= s["symbol_recall"] <= 1 and s["price_error"] >= 0 for s in doc["sets"])
    assert all(s[k] is None or 0 <= s[k] <= 1 for s in doc["sets"] for k in ("link_precision", "link_recall"))
    assert all(0 <= t["found"] <= min(t["total"], t["reported"]) for t in doc["per_type"]) and doc["per_type"]
    assert all(len(p["values"]) > 0 for p in doc["price_error"])
    for run in doc["training"]:
        assert run["passes"] == list(range(1, len(run["passes"]) + 1)) and len(run["score"]) == len(run["train_loss"]) == len(run["passes"]), run["run"]
        assert run["best_pass"] in run["passes"] and run["metric"]
    assert sum(r["live"] for r in doc["training"]) == 1 and all(r["live_pass"] is None or (r["live"] and r["live_pass"] in r["passes"]) for r in doc["training"])
    t = doc["threshold"]
    assert t["thresholds"] == sorted(t["thresholds"]) and t["chosen"] in t["thresholds"]
    assert all(len(t[k]) == len(t["thresholds"]) for k in ("precision", "recall", "f1", "blank_paper_boxes"))
    cal = doc["calibration"]
    assert all(a["from"] < a["to"] and a["to"] == b["from"] for a, b in zip(cal, cal[1:])) and cal[-1]["to"] == 1 and cal[0]["from"] == t["chosen"]
    assert all(0 <= b["precision"] <= 1 and b["boxes"] > 0 for b in cal) and all(a["precision"] <= b["precision"] + 0.03 for a, b in zip(cal, cal[1:]))
    assert sum(o["chosen"] for o in doc["ocr"]) == 1 and all(o["linux"] == (o["engine"] != "vision") for o in doc["ocr"])
    assert [s["step"] for s in doc["timing"]["stages"]] == ["text", "symbols", "labels", "spec", "price"] and doc["timing"]["n"] == sum(k["n"] for k in doc["timing"]["by_kind"])
    assert [c["train"] for c in doc["data"]["classes"]] == sorted((c["train"] for c in doc["data"]["classes"]), reverse=True) and doc["data"]["splits"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="rtdetr_r18_plain_ema_fix_pass4", help="the site's detector: weights/<model>")
    ap.add_argument("--out", default="eval/charts.json")
    a = ap.parse_args(argv)
    doc = build(a.model)
    validate(doc)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"{a.out}: {os.path.getsize(a.out) / 1024:.0f} kB; sections {', '.join(k for k in doc if k not in ('generated', 'model', 'engine'))}; "
          f"training runs {[r['run'] for r in doc.get('training', [])]}")


if __name__ == "__main__":
    main()
