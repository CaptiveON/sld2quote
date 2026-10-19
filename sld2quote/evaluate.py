"""Evaluation against ground truth: symbol detection, label reading, notes, conflicts, bill-of-materials lines and price.
The reference bill of materials is built by running the same rules on the ground truth, so the metrics measure reading, not pricing.

  python3 -m sld2quote.evaluate --data data/baseline --out eval                     # the classical reader
  python3 -m sld2quote.evaluate --data data/eval/test --out eval --reader trained   # the trained reader (--weights, --device)
  python3 -m sld2quote.evaluate --make-sets                                         # data/eval/test and data/eval/unseen

A run writes <out>/results.json (its rows, one per sheet) and its own section of <out>/RESULTS.md, one section per set and
reader: a rerun replaces only its own section."""
from __future__ import annotations
import argparse, glob, json, os, re, shutil, time
from collections import Counter, defaultdict
from .classes import DEVICE_TO_CLASS
from .ocr import default_engine
from .reader import read_sheet, add_reader_args, choose
from .spec import read_spec, cross_check
from .rules import build_bom

TYPES = list(DEVICE_TO_CLASS)  # device types, in the order the tables list them
TARGET = 0.95                  # the symbol rule: precision and recall per type
HEAD = ["# sld2quote evaluation", "",
        "Synthetic sheets only. One section per evaluation set and reader, each written by `python3 -m sld2quote.evaluate` (a rerun replaces "
        "its own section and keeps the others). The reference bill of materials is built by the same rules from ground truth, so the numbers "
        "measure reading, not pricing. A predicted device matches a ground-truth device of the same type at IoU >= 0.3. Link precision and "
        "recall score the symbol-to-label link alone: only links between a device matched to the answer key and a text the reader located "
        "(an OCR box lying over a text the generator drew, whether or not its characters were read right) count, so a device the answer key "
        "does not match drops out of both, and a note or title line linked to a device counts as a wrong link; a link near 1.0 beside a "
        "misrated device means the right text was attached but read or parsed wrong."]


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy; return inter / float(aw * ah + bw * bh - inter + 1e-9)


def gt_extraction(gt: dict) -> dict:
    """ground truth in the reader's output shape"""
    devs = []
    for i, d in enumerate(gt["devices"]):
        devs.append({"id": f"G{i + 1}", "symbol": f"G{i + 1}", "box": d["box"], "type": d["type"], "rating": d.get("rating"), "poles": d.get("poles"), "kA": d.get("kA"),
                     "dest": d.get("dest", ""), "label": d.get("label", ""), "label_texts": [], "conf": 1.0, "on_bus": d.get("on_bus")})
    bus = [{"id": b["id"], "rating": b["rating"], "kA": gt["notes"]["fault_kA"], "label_text": "gt", "x1": b["x1"], "x2": b["x2"], "y1": b["y"] - 3, "y2": b["y"] + 3, "y": b["y"]} for b in gt["busbars"]]
    return {"image": gt["image"], "size": gt["size"], "texts": [], "busbars": bus, "devices": devs,
            "notes": {"form": gt["notes"]["form"], "ip": gt["notes"]["ip"], "fault_kA": gt["notes"]["fault_kA"], "poles_default": gt["notes"]["poles_default"], "evidence": ["gt"]}}


def drawn(box, gtx) -> int | None:
    """the answer-key text an OCR box reads: the one it overlaps most, if by half the smaller box or more"""
    best, bi = 0.0, None
    for j, t in enumerate(gtx):
        ix = max(0, min(box[0] + box[2], t["box"][0] + t["box"][2]) - max(box[0], t["box"][0]))
        iy = max(0, min(box[1] + box[3], t["box"][1] + t["box"][3]) - max(box[1], t["box"][1]))
        if ix * iy > best:
            best, bi = ix * iy, j
    return bi if bi is not None and best >= 0.5 * min(box[2] * box[3], gtx[bi]["box"][2] * gtx[bi]["box"][3]) else None


def link(ex: dict, gt: dict, pairs: dict) -> dict:
    """the symbol-to-label link apart from symbol finding and text reading. Made: the reader's text-to-device assignments
    between a device matched to the answer key and a text read where the key drew one; right when the key drew that text
    for that device. Due: the key's assignments whose device was found and whose text was read; found when the reader
    made it. Empty for answer keys written before texts recorded their device."""
    gtx, ids = gt["texts"], [d["id"] for d in gt["devices"]]
    if not gtx or any("device" not in t for t in gtx):
        return {}
    read = {t["id"]: drawn(t["box"], gtx) for t in ex["texts"]}
    made = [(read[tid], pairs[d["id"]]) for d in ex["devices"] if d["id"] in pairs for tid in d["label_texts"] if read.get(tid) is not None]
    found = set(pairs.values()); seen = {g for g in read.values() if g is not None}
    due = {(g, ids.index(t["device"])) for g, t in enumerate(gtx) if g in seen and t["device"] in ids and ids.index(t["device"]) in found}
    return {"link_made": len(made), "link_right": sum(gtx[g]["device"] == ids[j] for g, j in made), "link_due": len(due), "link_found": len(due & set(made))}


def eval_sheet(png: str, gt_path: str, spec_path: str, read=read_sheet) -> dict:
    gt = json.load(open(gt_path))
    ex = read(png)
    secs = ex["seconds"]  # the reader's own clock, so a trained model's one-off loading is not charged to a sheet
    sp = read_spec(spec_path)
    conflicts = cross_check(ex["notes"], sp)
    bom = build_bom(ex, sp, conflicts)
    gex = gt_extraction(gt)
    gconf = cross_check(gex["notes"], sp)
    gbom = build_bom(gex, sp, gconf)
    # symbols: match predicted devices to GT by IoU >= 0.3 and same type
    gt_devs = gt["devices"]; matched = set(); pairs = {}; tp = 0; rating_ok = poles_ok = ka_ok = 0; hit, hit_rating = Counter(), Counter()
    for pd in ex["devices"]:
        best, bi = 0, None
        for j, gd in enumerate(gt_devs):
            if j in matched or gd["type"] != pd["type"]:
                continue
            v = iou(pd["box"], gd["box"])
            if v > best:
                best, bi = v, j
        if bi is not None and best >= 0.3:
            matched.add(bi); pairs[pd["id"]] = bi; tp += 1; gd = gt_devs[bi]; ok = int(pd.get("rating") == gd.get("rating"))
            rating_ok += ok; hit[gd["type"]] += 1; hit_rating[gd["type"]] += ok
            poles_ok += int((pd.get("poles") or ex["notes"].get("poles_default")) == (gd.get("poles") or gt["notes"]["poles_default"]))
            ka_ok += int((pd.get("kA") or ex["notes"].get("fault_kA")) == (gd.get("kA") or gt["notes"]["fault_kA"]))
    prec = tp / max(1, len(ex["devices"])); rec = tp / max(1, len(gt_devs)); lk = link(ex, gt, pairs)
    npred, ngt = Counter(d["type"] for d in ex["devices"]), Counter(d["type"] for d in gt_devs)
    notes_fields = ["form", "ip", "fault_kA", "poles_default"]
    notes_ok = sum(int(str(ex["notes"].get(k)).lower() == str(gt["notes"].get(k)).lower()) for k in notes_fields)
    # conflicts: set of fields
    pc = {c["field"] for c in conflicts}; gc = {c["field"] for c in gt.get("conflicts", [])}
    # BoM line-item multiset on (part, qty)
    pm = Counter((l["part"], l["qty"]) for l in bom["lines"]); gm = Counter((l["part"], l["qty"]) for l in gbom["lines"])
    inter = sum((pm & gm).values()); bprec = inter / max(1, sum(pm.values())); brec = inter / max(1, sum(gm.values()))
    price_err = abs(bom["total_gbp"] - gbom["total_gbp"]) / max(1.0, gbom["total_gbp"])
    return {"sheet": os.path.basename(png), "scan": "-scan" in os.path.basename(png), "gt_devices": len(gt_devs), "pred_devices": len(ex["devices"]), "tp": tp,
            "sym_precision": round(prec, 3), "sym_recall": round(rec, 3), **lk, "link_precision": round(lk["link_right"] / lk["link_made"], 3) if lk.get("link_made") else None,
            "link_recall": round(lk["link_found"] / lk["link_due"], 3) if lk.get("link_due") else None,
            "rating_acc": round(rating_ok / max(1, tp), 3), "poles_acc": round(poles_ok / max(1, tp), 3), "ka_acc": round(ka_ok / max(1, tp), 3),
            "notes_acc": round(notes_ok / len(notes_fields), 3), "conflict_tp": len(pc & gc), "conflict_fp": len(pc - gc), "conflict_fn": len(gc - pc),
            "bom_precision": round(bprec, 3), "bom_recall": round(brec, 3), "price_error": round(price_err, 4), "pred_total": bom["total_gbp"], "gt_total": gbom["total_gbp"], "seconds": round(secs["total"], 1),
            "seconds_ocr": secs["ocr"], "seconds_detector": secs["detector"], "reader": ex["reader"], "detector": ex.get("detector"), "crops": ex.get("crops"), "engine": engine(),
            "per_type": {t: {"tp": hit[t], "pred": npred[t], "gt": ngt[t], "rating_ok": hit_rating[t]} for t in TYPES if npred[t] or ngt[t]}}


def engine() -> str:
    """the OCR engine the reader used: SLD2QUOTE_OCR, else the platform default"""
    return os.environ.get("SLD2QUOTE_OCR") or default_engine()


def agg(sub):
    n = max(1, len(sub)); tp = sum(r["tp"] for r in sub); gtn = sum(r["gt_devices"] for r in sub); pdn = sum(r["pred_devices"] for r in sub)
    lk = [r for r in sub if "link_made" in r]; tot = lambda k: sum(r[k] for r in lk)
    return {"sheets": len(sub), "sym_precision": tp / max(1, pdn), "sym_recall": tp / max(1, gtn),
            "link_precision": tot("link_right") / max(1, tot("link_made")) if lk else None, "link_recall": tot("link_found") / max(1, tot("link_due")) if lk else None,
            "rating_acc": sum(r["rating_acc"] * r["tp"] for r in sub) / max(1, tp),
            "poles_acc": sum(r["poles_acc"] * r["tp"] for r in sub) / max(1, tp), "ka_acc": sum(r["ka_acc"] * r["tp"] for r in sub) / max(1, tp),
            "notes_acc": sum(r["notes_acc"] for r in sub) / n, "conflicts": f"{sum(r['conflict_tp'] for r in sub)} found of {sum(r['conflict_tp'] + r['conflict_fn'] for r in sub)}, {sum(r['conflict_fp'] for r in sub)} false",
            "bom_precision": sum(r["bom_precision"] for r in sub) / n, "bom_recall": sum(r["bom_recall"] for r in sub) / n,
            "price_error_mean": sum(r["price_error"] for r in sub) / n, "price_error_max": max((r["price_error"] for r in sub), default=0), "seconds_mean": sum(r["seconds"] for r in sub) / n,
            "seconds_ocr_mean": sum(r["seconds_ocr"] for r in sub) / n, "seconds_detector_mean": sum(r["seconds_detector"] for r in sub) / n}


def per_type(sub) -> dict:
    """tp, pred, gt and rating_ok per device type, summed over the sheets"""
    tot = defaultdict(Counter)
    for r in sub:
        for t, c in r["per_type"].items():
            tot[t].update(c)
    return {t: tot[t] for t in TYPES if t in tot}


def ratio(a, b):
    return a / b if b else None


def f3(v) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def rule_line(name: str, n: int, pt: dict) -> str:
    """the symbol rule at drawing level: every type's precision and recall over all n sheets, and which reach TARGET"""
    ok, bad = [], []
    for t, c in pt.items():
        p, r = ratio(c["tp"], c["pred"]), ratio(c["tp"], c["gt"])
        (ok if p is not None and r is not None and min(p, r) >= TARGET else bad).append(f"{t} {f3(p)} / {f3(r)}")
    return f"- {name} ({n} sheets): {len(ok)} of {len(pt)} types pass ({', '.join(ok) or 'none'}); fail: {', '.join(bad) or 'none'}"


def describe(rows: list) -> str:
    d = rows[0]["detector"] if rows else None
    on = d and (d["device"] + (f" ({d['threads']} threads)" if d["device"] == "cpu" and d.get("threads") else ""))
    src = {"drawing": " (chosen on whole validation drawings)", "tile": " (chosen per tile at training)", "argument": " (given)"}.get(d.get("score_threshold_source"), "") if d else ""
    crops = "" if not rows or rows[0].get("crops") is None else f"; OCR engine {rows[0].get('engine', '?')}, crop pass around every symbol {'on' if rows[0]['crops'] else 'off'}"
    return ("classical reader (shape tests)" if not d else f"trained reader: detector `{d['weights']}` on {on}, score threshold {d['score_threshold']}{src}, "
            f"NMS IoU {d['nms_iou']} per tile, tiles merged at IoU {d['merge_iou']}" +
            (f", boxes within {d['edge_px']} px of an inner tile edge dropped" if d.get("edge_px") is not None else "") +
            (f", a box holding {d['contain']} of a better box's area dropped" if d.get("contain") is not None else "")) + crops


def section(rows: list) -> list[str]:
    clean = [r for r in rows if not r["scan"]]; scan = [r for r in rows if r["scan"]]; splits = [("clean", clean), ("scan-like", scan)]
    L = [f"Run on {len(rows)} synthetic sheets ({len(clean)} clean, {len(scan)} scan-like) on {time.strftime('%Y-%m-%d %H:%M')}; {describe(rows)}. "
         "Seconds are the reader's own, split into OCR and the symbol finder (\"detector\").", "",
         "| Metric | Clean | Scan-like |", "|---|---|---|"]
    ca, sa = agg(clean), agg(scan)
    L += [f"| {k} | {f3(ca[k])} | {f3(sa[k])} |" for k in ca]
    pts = [per_type(sub) for _, sub in splits]
    L += ["", "### Symbols per device type", "",
          "Precision and recall over all sheets of the split; n is the ground-truth count, rating the rating accuracy over that type's matches; - where there is nothing to count.", "",
          "| type | " + " | ".join(f"{s} {k}" for s, _ in splits for k in ("P", "R", "n", "rating")) + " |", "|---|" + "---|" * 8]
    for t in [t for t in TYPES if any(t in pt for pt in pts)]:
        cells = []
        for pt in pts:
            c = pt.get(t, Counter())
            cells += [f3(ratio(c["tp"], c["pred"])), f3(ratio(c["tp"], c["gt"])), str(c["gt"]), f3(ratio(c["rating_ok"], c["tp"]))]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", f"Symbol rule at drawing level (precision and recall >= {TARGET} for each type, over all sheets of the split; type precision / recall):"]
    L += [rule_line(s, len(sub), pt) for (s, sub), pt in zip(splits, pts)]
    L += ["", f"<details><summary>Per sheet ({len(rows)})</summary>", "",
          "| sheet | gt | pred | tp | sym P | sym R | rating | notes | BoM P | BoM R | price err | s | OCR s | detector s | link P | link R |", "|" + "---|" * 16]
    for r in rows:
        L.append(f"| {r['sheet']} | {r['gt_devices']} | {r['pred_devices']} | {r['tp']} | {r['sym_precision']} | {r['sym_recall']} | {r['rating_acc']} | {r['notes_acc']} | {r['bom_precision']} | "
                 f"{r['bom_recall']} | {r['price_error']:.3f} | {r['seconds']} | {r['seconds_ocr']} | {r['seconds_detector']} | {f3(r.get('link_precision'))} | {f3(r.get('link_recall'))} |")
    return L + ["", "</details>"]


def write_section(path: str, title: str, lines: list):
    """this run's section of RESULTS.md: it replaces the section with the same heading, or is added at the end"""
    text = open(path).read() if os.path.exists(path) else "\n".join(HEAD) + "\n"
    head, *parts = re.split(r"(?m)^(?=## )", text)
    sections = {p.split("\n", 1)[0]: p.rstrip("\n") for p in parts}
    sections[title] = "\n".join([title, ""] + lines)
    open(path, "w").write(head.rstrip("\n") + "\n\n" + "\n\n".join(sections.values()) + "\n")


SETS = {"test": ("test_clean", "test_scan"), "unseen": ("test_unseen",), "val": ("val",), "val_unseen": ("val_unseen",)}  # set: the export's splits


def sets_dir(coco: str) -> str:
    """where an export's evaluation sets go: data/coco's in data/eval, data/coco_fonts's in data/eval_fonts"""
    coco = os.path.normpath(coco); name = os.path.basename(coco)
    return os.path.join(os.path.dirname(coco), "eval" + (name[4:] if name.startswith("coco") else "_" + name))


def make_sets(coco: str = "data/coco", out: str | None = None) -> dict:
    """<out>/test, unseen, val and val_unseen: the sheets of the export's splits (SETS), both copies with their answer keys
    and specifications, hard-linked from the pool its manifest names, so drawing-level and tile-level numbers come from the
    same sheets and a threshold can be chosen on whole validation drawings. Each set is rebuilt from scratch, so it follows
    the manifest when a split is widened; a split the manifest lacks gives no set."""
    man = json.load(open(os.path.join(coco, "manifest.json"))); sp, pool = man["splits"], man["pool"]["path"]; out = out or sets_dir(coco)
    sets = {name: dict.fromkeys(i for k in splits for i in sp[k]["sheet_ids"]) for name, splits in SETS.items() if all(k in sp for k in splits)}
    for name, ids in sets.items():
        d = os.path.join(out, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        for f in (s + c + e for s in ids for c in ("", "-scan") for e in (".png", ".json", "-spec.txt")):
            try:
                os.link(os.path.join(pool, f), os.path.join(d, f))
            except OSError as e:
                if not os.path.exists(os.path.join(pool, f)):
                    raise FileNotFoundError(f"{pool}/{f}: named by {coco}/manifest.json but not in the pool") from e
                shutil.copy2(os.path.join(pool, f), os.path.join(d, f))  # another file system: copy instead
    return {os.path.join(out, name): len(ids) for name, ids in sets.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m sld2quote.evaluate", description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="eval")
    add_reader_args(ap)
    ap.add_argument("--make-sets", action="store_true", help="build the test, unseen, val and val_unseen sets from the manifest in --coco, then stop")
    ap.add_argument("--coco", default="data/coco", help="the COCO export whose manifest names the sheets of each split")
    ap.add_argument("--sets", help="where --make-sets puts the sets (default data/eval for data/coco, data/eval_fonts for data/coco_fonts)")
    ap.add_argument("--tag", help="added to the section title, so runs of different code or weights copies on the same settings sit side by side")
    a = ap.parse_args(argv)
    if a.make_sets:
        for d, n in make_sets(a.coco, a.sets).items():
            print(f"{d}: {n} sheets, clean and scan copies")
        return
    read = choose(a.reader, a.weights, a.device, a.crops)
    rows = []
    for png in sorted(glob.glob(os.path.join(a.data, "sheet-*.png"))):
        stem = png[:-4]; gt = stem + ".json"; spec = os.path.join(a.data, os.path.basename(stem).replace("-scan", "") + "-spec.txt")  # the clean copy's
        if os.path.exists(gt) and os.path.exists(spec):
            r = eval_sheet(png, gt, spec, read); rows.append(r); print(json.dumps({k: v for k, v in r.items() if k != "per_type"}))
    if not rows:
        raise SystemExit(f"{a.data}: no sheet-*.png with its answer key (.json) and specification (-spec.txt)")
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    # the engine and crop setting are in the title, so a run with another setting sits beside the earlier sections instead of replacing them
    title = f"## {os.path.normpath(a.data)}, {a.reader} reader" + (f" (`{a.weights}`)" if a.reader == "trained" else "") + f", {rows[0]['engine']} OCR, crops {'on' if rows[0]['crops'] else 'off'}" + (f", {a.tag}" if a.tag else "")
    L = section(rows)
    write_section(os.path.join(a.out, "RESULTS.md"), title, L)
    print("\n".join([title, ""] + L[:next(i for i, l in enumerate(L) if l.startswith("<details>"))]))


if __name__ == "__main__":
    main()
