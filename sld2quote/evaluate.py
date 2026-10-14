"""Evaluation against ground truth: symbol detection, label reading, notes, conflicts, bill-of-materials lines and price.
The reference bill of materials is built by running the same rules on the ground truth, so the metrics measure reading, not pricing."""
from __future__ import annotations
import argparse, glob, json, os, time
from collections import Counter
from .reader import read_sheet
from .spec import read_spec, cross_check
from .rules import build_bom


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


def eval_sheet(png: str, gt_path: str, spec_path: str) -> dict:
    gt = json.load(open(gt_path))
    t0 = time.time()
    ex = read_sheet(png)
    secs = time.time() - t0
    sp = read_spec(spec_path)
    conflicts = cross_check(ex["notes"], sp)
    bom = build_bom(ex, sp, conflicts)
    gex = gt_extraction(gt)
    gconf = cross_check(gex["notes"], sp)
    gbom = build_bom(gex, sp, gconf)
    # symbols: match predicted devices to GT by IoU >= 0.3 and same type
    gt_devs = gt["devices"]; matched = set(); tp = 0; rating_ok = poles_ok = ka_ok = 0
    for pd in ex["devices"]:
        best, bi = 0, None
        for j, gd in enumerate(gt_devs):
            if j in matched or gd["type"] != pd["type"]:
                continue
            v = iou(pd["box"], gd["box"])
            if v > best:
                best, bi = v, j
        if bi is not None and best >= 0.3:
            matched.add(bi); tp += 1; gd = gt_devs[bi]
            rating_ok += int(pd.get("rating") == gd.get("rating"))
            poles_ok += int((pd.get("poles") or ex["notes"].get("poles_default")) == (gd.get("poles") or gt["notes"]["poles_default"]))
            ka_ok += int((pd.get("kA") or ex["notes"].get("fault_kA")) == (gd.get("kA") or gt["notes"]["fault_kA"]))
    prec = tp / max(1, len(ex["devices"])); rec = tp / max(1, len(gt_devs))
    notes_fields = ["form", "ip", "fault_kA", "poles_default"]
    notes_ok = sum(int(str(ex["notes"].get(k)).lower() == str(gt["notes"].get(k)).lower()) for k in notes_fields)
    # conflicts: set of fields
    pc = {c["field"] for c in conflicts}; gc = {c["field"] for c in gt.get("conflicts", [])}
    # BoM line-item multiset on (part, qty)
    pm = Counter((l["part"], l["qty"]) for l in bom["lines"]); gm = Counter((l["part"], l["qty"]) for l in gbom["lines"])
    inter = sum((pm & gm).values()); bprec = inter / max(1, sum(pm.values())); brec = inter / max(1, sum(gm.values()))
    price_err = abs(bom["total_gbp"] - gbom["total_gbp"]) / max(1.0, gbom["total_gbp"])
    return {"sheet": os.path.basename(png), "scan": "-scan" in png, "gt_devices": len(gt_devs), "pred_devices": len(ex["devices"]), "tp": tp,
            "sym_precision": round(prec, 3), "sym_recall": round(rec, 3), "rating_acc": round(rating_ok / max(1, tp), 3), "poles_acc": round(poles_ok / max(1, tp), 3), "ka_acc": round(ka_ok / max(1, tp), 3),
            "notes_acc": round(notes_ok / len(notes_fields), 3), "conflict_tp": len(pc & gc), "conflict_fp": len(pc - gc), "conflict_fn": len(gc - pc),
            "bom_precision": round(bprec, 3), "bom_recall": round(brec, 3), "price_error": round(price_err, 4), "pred_total": bom["total_gbp"], "gt_total": gbom["total_gbp"], "seconds": round(secs, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="eval"); a = ap.parse_args(argv)
    rows = []
    for png in sorted(glob.glob(os.path.join(a.data, "sheet-*.png"))):
        stem = png[:-4]; gt = stem + ".json"; spec = stem.replace("-scan", "") + "-spec.txt"
        if os.path.exists(gt) and os.path.exists(spec):
            r = eval_sheet(png, gt, spec); rows.append(r); print(json.dumps(r))
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    def agg(sub):
        n = max(1, len(sub)); tp = sum(r["tp"] for r in sub); gtn = sum(r["gt_devices"] for r in sub); pdn = sum(r["pred_devices"] for r in sub)
        return {"sheets": len(sub), "sym_precision": tp / max(1, pdn), "sym_recall": tp / max(1, gtn), "rating_acc": sum(r["rating_acc"] * r["tp"] for r in sub) / max(1, tp),
                "poles_acc": sum(r["poles_acc"] * r["tp"] for r in sub) / max(1, tp), "ka_acc": sum(r["ka_acc"] * r["tp"] for r in sub) / max(1, tp),
                "notes_acc": sum(r["notes_acc"] for r in sub) / n, "conflicts": f"{sum(r['conflict_tp'] for r in sub)} found of {sum(r['conflict_tp'] + r['conflict_fn'] for r in sub)}, {sum(r['conflict_fp'] for r in sub)} false",
                "bom_precision": sum(r["bom_precision"] for r in sub) / n, "bom_recall": sum(r["bom_recall"] for r in sub) / n,
                "price_error_mean": sum(r["price_error"] for r in sub) / n, "price_error_max": max((r["price_error"] for r in sub), default=0), "seconds_mean": sum(r["seconds"] for r in sub) / n}
    clean = [r for r in rows if not r["scan"]]; scan = [r for r in rows if r["scan"]]
    L = ["# sld2quote evaluation", "", f"Run on {len(rows)} synthetic sheets ({len(clean)} clean, {len(scan)} scan-like). Reference bill of materials built by the same rules from ground truth, so the numbers measure reading, not pricing.", "",
         "| Metric | Clean | Scan-like |", "|---|---|---|"]
    ca, sa = agg(clean), agg(scan)
    for k in ["sheets", "sym_precision", "sym_recall", "rating_acc", "poles_acc", "ka_acc", "notes_acc", "conflicts", "bom_precision", "bom_recall", "price_error_mean", "price_error_max", "seconds_mean"]:
        f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        L.append(f"| {k} | {f(ca[k])} | {f(sa[k])} |")
    L += ["", "## Per sheet", "", "| sheet | gt | pred | tp | sym P | sym R | rating | notes | BoM P | BoM R | price err | s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['sheet']} | {r['gt_devices']} | {r['pred_devices']} | {r['tp']} | {r['sym_precision']} | {r['sym_recall']} | {r['rating_acc']} | {r['notes_acc']} | {r['bom_precision']} | {r['bom_recall']} | {r['price_error']:.3f} | {r['seconds']} |")
    open(os.path.join(a.out, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
