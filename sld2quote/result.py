"""One run of the pipeline as the site's Result: the files cli.run writes, plus a sheet copy, the review overlay as a PNG,
result.json, plain-language sentences for the five steps and, with an answer key, the score by evaluate.py's own matching.

  python3 -m sld2quote.result <image> [--spec sheet-spec.txt] [--gt sheet.json] --out <dir> [--reader trained|classical]

Image and file entries of the Result are bare file names inside out_dir; the server maps them to URLs."""
from __future__ import annotations
import argparse, base64, json, os, shutil, time
from collections import Counter
from PIL import Image
from . import __version__
from .evaluate import eval_sheet
from .ocr import default_engine
from .quote import _overlay, write_outputs
from .reader import AMPS, SOURCES, WEIGHTS, add_reader_args, choose
from .rules import CAT, build_bom
from .spec import cross_check, read_spec

TYPE_PLAIN = {"ACB": "incoming air circuit breaker", "MCCB": "outgoing moulded-case breaker", "BUS_SECTION": "bus-section switch",
              "FUSE_SWITCH": "fuse switch", "ISOLATOR": "isolator", "CONTACTOR": "motor starter", "TRANSFORMER": "transformer",
              "GENERATOR": "generator", "MOTOR": "motor", "BOARD": "downstream board", "CT": "current transformer", "METER": "panel meter",
              "RELAY": "protection relay", "SPD": "surge protector", "CAPACITOR": "power-factor correction bank", "EARTH": "earth"}
UNIT = {t: "A" for t in AMPS} | {t: "kVA" for t in SOURCES} | {"CONTACTOR": "kW", "MOTOR": "kW", "CAPACITOR": "kVAr"}
FILES = {"quotation_md": "quotation.md", "bom_csv": "bom.csv", "review_html": "review.html", "extraction_json": "extraction.json"}
NOTE = {"generated": "Drawn by the generator from seed {seed} ({level}, {copy} copy) and scored against the generator's own answer key.",
        "sample": "A synthetic drawing with its answer key, so the run is scored.",
        "upload": "Your drawing: no answer key, so there is no score; check the overlay against the drawing."}


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}{'es' if word.endswith(('s', 'sh', 'ch', 'x')) else 's'}"


def steps(ex: dict, sp: dict | None, conflicts: list, bom: dict, spec_s: float, price_s: float) -> list[dict]:
    """the five steps in plain language, with the counts a reader can check against the tables"""
    secs, devs = ex["seconds"], ex["devices"]
    n_text = sum(t.get("src") == "page" for t in ex["texts"])
    kinds = Counter(TYPE_PLAIN.get(d["type"], d["type"].lower()) for d in devs)
    found = ", ".join(plural(n, k) for k, n in kinds.most_common())
    n_lab, n_rated = sum(bool(d.get("label_texts")) for d in devs), sum(d.get("rating") is not None for d in devs)
    b = bom["board"]
    board = f"{b['form']}, {b['ip']}, {b['fault_kA']} kA" if b["fault_kA"] else f"{b['form']}, {b['ip']}, fault level not found"
    rest = max(0.0, secs["total"] - secs["ocr"] - secs["detector"])
    spec_plain = (f"Read {plural(len(sp['clauses']), 'clause')} of the specification and compared them with the drawing notes ({board}): "
                  f"{plural(len(conflicts), 'conflict')}. An estimator checks the drawing against the specification before pricing, because a "
                  "conflict is a question for the consultant, not a guess." if sp else
                  f"No specification was supplied, so the board's form, protection and fault level come from the drawing notes alone ({board}). "
                  "An estimator would check these against the specification before pricing.")
    return [
        {"key": "text", "title": "Read the words", "count": n_text, "seconds": secs["ocr"],
         "plain": f"Read {plural(n_text, 'piece')} of text in {secs['ocr']} s. An estimator starts by reading every rating, label and note on "
                  "the drawing; here the computer did it, and read again around each symbol for the small print."},
        {"key": "symbols", "title": "Find the symbols", "count": len(ex["symbols"]), "seconds": secs["detector"],
         "plain": f"Found {plural(len(ex['symbols']), 'symbol')} in {secs['detector']} s" + (f": {found}." if found else ".") +
                  " Each symbol is a device the switchboard must contain, so a missed symbol is a missing line on the quotation."},
        {"key": "labels", "title": "Pair labels with symbols", "count": n_lab, "seconds": round(rest, 2),
         "plain": f"Paired {n_lab} of {plural(len(devs), 'symbol')} with the label text beside them; {n_rated} carry a rating the pricing rules "
                  "can use. A breaker is priced by its amps, poles and fault rating, so the right label on the right symbol is what makes the price right."},
        {"key": "spec", "title": "Read the specification and compare", "count": len(sp["clauses"]) if sp else 0, "seconds": round(spec_s, 2), "plain": spec_plain},
        {"key": "price", "title": "Pick parts and price", "count": len(bom["lines"]), "seconds": round(price_s, 2),
         "plain": f"Chose {plural(len(bom['lines']), 'catalogue line')} and priced the board at £{bom['total_gbp']:,.0f} (illustrative prices, "
                  f"{CAT['margin'] * 100:.0f}% margin included); {plural(len(bom['issues']), 'item')} flagged for the estimator. Every line names the "
                  "symbol, text or clause it rests on, so a reviewer can check it on the drawing."}]


def score(row: dict) -> dict:
    """evaluate.py's per-sheet row as the Result's score"""
    return {"symbols_total": row["gt_devices"], "symbols_found": row["tp"], "symbols_reported": row["pred_devices"],
            "precision": row["sym_precision"], "recall": row["sym_recall"],
            "per_type": [{"type": t, "total": c["gt"], "found": c["tp"], "reported": c["pred"]} for t, c in row["per_type"].items()],
            "reference_total_gbp": row["gt_total"], "price_error": row["price_error"], "rating_accuracy": row["rating_acc"], "notes_accuracy": row["notes_acc"]}


def source_of(image_path: str, gt: dict | None, given: dict | None) -> dict:
    st = (gt or {}).get("style", {}); sid = (gt or {}).get("id", "")
    s = {"kind": "sample" if gt else "upload", "title": os.path.splitext(os.path.basename(image_path))[0],
         "seed": int(sid.rsplit("-", 1)[-1]) if sid.rsplit("-", 1)[-1].isdigit() else None, "level": st.get("level"),
         "copy": None if gt is None else ("scan" if st.get("scan") else "clean"), "has_answer_key": gt is not None, "note": ""}
    s.update(given or {})
    s["note"] = s["note"] or NOTE.get(s["kind"], "").format(**s)
    return s


def build_result(image_path: str, spec_path: str | None = None, out_dir: str | None = None, gt_path: str | None = None,
                 reader: str = "trained", weights: str = WEIGHTS, device: str = "cpu", source: dict | None = None) -> dict:
    """run the pipeline as cli.run does, write its files plus sheet.png, overlay.png and result.json into out_dir, return the Result"""
    t0 = time.perf_counter()
    out_dir = out_dir or os.path.join("out", os.path.splitext(os.path.basename(image_path))[0])
    os.makedirs(out_dir, exist_ok=True)
    ex = choose(reader, weights, device)(image_path)
    t1 = time.perf_counter()
    sp = read_spec(spec_path) if spec_path and os.path.exists(spec_path) else None
    conflicts = cross_check(ex["notes"], sp) if sp else []
    t2 = time.perf_counter()
    bom = build_bom(ex, sp, conflicts)
    t3 = time.perf_counter()
    write_outputs(out_dir, image_path, ex, sp, conflicts, bom, {"version": __version__, "seconds": round(t3 - t0, 1), "reader": ex["reader"]})
    if image_path.lower().endswith(".png"):
        shutil.copyfile(image_path, os.path.join(out_dir, "sheet.png"))
    else:
        Image.open(image_path).save(os.path.join(out_dir, "sheet.png"))
    with open(os.path.join(out_dir, "overlay.png"), "wb") as fh:
        fh.write(base64.b64decode(_overlay(image_path, ex, bom)))
    gt = json.load(open(gt_path)) if gt_path else None
    # the score by evaluate.py's own matching, on the extraction already read (os.devnull: a specification with no clauses)
    sc = score(eval_sheet(image_path, gt_path, spec_path if sp else os.devnull, read=lambda _: ex)) if gt else None
    det = ex.get("detector") or {}
    res = {"id": os.path.basename(os.path.normpath(out_dir)), "source": source_of(image_path, gt, source),
           "images": {"sheet": "sheet.png", "overlay": "overlay.png"}, "size": ex["size"],
           "steps": steps(ex, sp, conflicts, bom, t2 - t1, t3 - t2),
           "devices": [{"id": d["id"], "type": d["type"], "type_plain": TYPE_PLAIN.get(d["type"], d["type"]), "rating": d.get("rating"), "unit": UNIT.get(d["type"]),
                        "poles": d.get("poles"), "kA": d.get("kA"), "dest": d.get("dest", ""), "conf": d["conf"], "box": d["box"], "label": d.get("label", "")} for d in ex["devices"]],
           "board": bom["board"],
           "bom": {"lines": [{k: l[k] for k in ("line", "part", "description", "qty", "unit_gbp", "total_gbp", "conf", "evidence", "rule")} for l in bom["lines"]],
                   "subtotal_gbp": bom["subtotal_gbp"], "margin_gbp": bom["margin_gbp"], "total_gbp": bom["total_gbp"]},
           "issues": bom["issues"], "conflicts": [{k: c[k] for k in ("field", "drawing", "spec", "spec_clause")} for c in conflicts],
           "spec": {"clauses": sp["clauses"]} if sp else None, "score": sc,
           "timing": {"total": round(time.perf_counter() - t0, 2), "ocr": ex["seconds"]["ocr"], "detector": ex["seconds"]["detector"]},
           "reader": ex["reader"], "engine": os.environ.get("SLD2QUOTE_OCR") or default_engine(),
           "model": {"weights": det.get("weights"), "threshold": det.get("score_threshold")}, "files": dict(FILES)}
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m sld2quote.result", description=__doc__.splitlines()[0])
    ap.add_argument("image"); ap.add_argument("--spec"); ap.add_argument("--gt", help="answer key (the generator's .json): scores the run"); ap.add_argument("--out", required=True)
    add_reader_args(ap); ap.set_defaults(reader="trained", device="cpu")
    a = ap.parse_args(argv)
    r = build_result(a.image, a.spec, a.out, a.gt, a.reader, a.weights, a.device)
    print(json.dumps({"out": a.out, "devices": len(r["devices"]), "total_gbp": r["bom"]["total_gbp"], "score": r["score"], "timing": r["timing"]}))


if __name__ == "__main__":
    main()
