"""OCR measured against the generator's ground truth, so the text reader is chosen by numbers, not by reputation.

Each engine reads every page as the reader does (reader._ocr_image at 2x) and, with --crops, a window around every
ground-truth device (reader.ocr_region at 3x, with its same-line merge). Recognised boxes are matched one to one to the
ground-truth text boxes (IoU >= 0.3, or either centre inside the other box), strings are compared with whitespace
collapsed, and the strings the quote depends on are scored on their own, a missed one scoring 0: device labels (a number
with A, kA, kVA, kW or kVAr) and the notes lines the reader parses (kA, Form, IP, -pole). Pages are read as saved: scan
copies keep their tilt and the ground truth goes through image_box. Speed is wall time on the CPU at 2 threads.

python3 -m sld2quote.evaluate_ocr --data data/baseline --engines vision,rapidocr --crops
Records go to out/ocr_eval/<data>-<engine>.json and eval/OCR.md is rebuilt from every record there, so engines can be
measured one at a time (--engines none only rebuilds the report).
"""
from __future__ import annotations
import argparse, glob, inspect, json, os, platform, re, time
for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")  # every engine at 2 threads (ocr.THREADS, the Space's vCPUs), set before numpy and torch load
from collections import Counter
from PIL import Image
from . import ocr, reader
from .evaluate import iou
from .generate import image_box

LABEL = re.compile(r"\d\s*(?:A|kA|kVA|kW|kVAr)\b")       # device labels: ratings in amps, kA, kVA, kW, kVAr
NOTE = re.compile(r"^\d\.\s.*(?:kA|Form|IP\d|-pole)")     # the notes lines the reader parses
STANDARD = [1600, 1000]                                   # a standard sheet, the size the time rule is set on
LIMIT_S = 5.0                                             # the rule: under about 5 s per standard sheet on 2 threads
OUT = os.path.join("out", "ocr_eval")
INSTALL = {  # what each engine needs on a Linux Space, lightest first
    "tesseract": "apt tesseract-ocr (about 30 MB with the English model), pip pytesseract",
    "rapidocr": "pip rapidocr-onnxruntime (onnxruntime, OpenCV, 15 MB of bundled models), a 9 MB English recogniser fetched once, apt libgl1 for OpenCV",
    "rapidocr-ch": "pip rapidocr-onnxruntime (onnxruntime, OpenCV, 15 MB of bundled models), apt libgl1 for OpenCV",
    "doctr": "pip python-doctr (torch, OpenCV, onnx, h5py and more), 124 MB of models fetched once",
    "easyocr": "pip easyocr (torch, OpenCV, scikit-image), 94 MB of models fetched once",
    "vision": "macOS only (Apple Vision), not available on Linux"}


def norm(s: str) -> str:
    return " ".join(s.replace("\u2019", "'").split())


def lev(a: str, b: str) -> int:
    """edit distance (insertions, deletions, substitutions)"""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def char_acc(pred: str, gt: str) -> float:
    """1 minus the edit distance over the longer string, on whitespace-collapsed text"""
    a, b = norm(pred), norm(gt)
    return 1 - lev(a, b) / max(len(a), len(b), 1)


def _centre_in(a, b) -> bool:
    cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
    return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]


def _inside(a, b) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]


def match(gt: list, pred: list) -> list[tuple[int, int]]:
    """one-to-one (ground truth, prediction) index pairs: IoU >= 0.3 or either centre inside the other box, best IoU first"""
    cand = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            v = iou(g, p)
            if v >= 0.3 or _centre_in(p, g) or _centre_in(g, p):
                cand.append((v, i, j))
    used_g, used_p, pairs = set(), set(), []
    for _, i, j in sorted(cand, reverse=True):
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); pairs.append((i, j))
    return pairs


def score(gt: list[dict], pred: list[dict]) -> Counter:
    """counts for one pass: boxes found and strings read, plus the label and notes strings on their own"""
    read = {i: pred[j]["text"] for i, j in match([g["box"] for g in gt], [p["box"] for p in pred])}
    c = Counter(gt=len(gt), pred=len(pred), matched=len(read))
    for i, g in enumerate(gt):
        acc = char_acc(read[i], g["text"]) if i in read else 0.0
        c["exact"] += i in read and acc == 1.0; c["char"] += acc
        key = "notes" if NOTE.search(g["text"]) else "label" if LABEL.search(g["text"]) else None
        if key:
            c[key + "_n"] += 1; c[key + "_exact"] += acc == 1.0; c[key + "_char"] += acc
    return c


def measure_sheet(png: str, gt: dict, crops: bool) -> dict:
    img = Image.open(png).convert("L")
    rot = gt["style"].get("rotate_deg", 0.0)
    texts = [dict(t, box=image_box(t["box"], gt["size"], rot)) for t in gt["texts"]]
    t = time.time(); page = reader._ocr_image(img, 2.0, "page")
    row = {"sheet": os.path.basename(png), "copy": "scan" if gt["style"].get("scan") else "clean", "size": gt["size"],
           "page": dict(score(texts, page), s=time.time() - t)}
    if crops:
        pad = inspect.signature(reader.ocr_region).parameters["pad"].default
        tot, secs = Counter(), 0.0
        for d in gt["devices"]:
            x, y, w, h = image_box(d["box"], gt["size"], rot)
            win = [x - pad[0], y - pad[1], w + pad[0] + pad[2], h + pad[1] + pad[3]]
            t = time.time(); got = reader.ocr_region(img, [x, y, w, h]); secs += time.time() - t
            full = [g for g in texts if _inside(g["box"], win)]
            cut = [g["box"] for g in texts if not _inside(g["box"], win) and iou(g["box"], win) > 0]
            got = [p for p in got if not any(_centre_in(p["box"], b) for b in cut)]  # text the window cuts is no engine's fault
            tot += score(full, got)
        row["crops"] = dict(tot, s=secs, n=len(gt["devices"]))
    return row


def run_engine(data: str, engine: str, crops: bool, max_width: int | None) -> None:
    os.environ["SLD2QUOTE_OCR"] = engine  # the reader's own OCR calls follow it
    t = time.time(); reader._ocr_image(Image.new("L", (240, 60), 255), 1.0, "warm-up"); load = time.time() - t
    rec = {"engine": engine, "data": data, "threads": ocr.THREADS, "omp": os.environ["OMP_NUM_THREADS"], "load_s": round(load, 2), "crops": crops, "start": time.strftime("%Y-%m-%d %H:%M"),
           "machine": f"{platform.platform()}, {os.cpu_count()} cores", "loadavg": [round(os.getloadavg()[0], 1)], "sheets": []}
    for png in sorted(glob.glob(os.path.join(data, "sheet-*.png"))):
        with open(png[:-4] + ".json") as fh:
            gt = json.load(fh)
        if max_width and gt["size"][0] > max_width:
            continue
        rec["sheets"].append(measure_sheet(png, gt, crops))
        p = rec["sheets"][-1]["page"]
        print(engine, rec["sheets"][-1]["sheet"], f"recall {p['matched'] / max(1, p['gt']):.2f}", f"{p['s']:.1f} s", flush=True)
    rec["loadavg"].append(round(os.getloadavg()[0], 1)); rec["end"] = time.strftime("%Y-%m-%d %H:%M")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{os.path.basename(os.path.normpath(data))}-{engine}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)


def summary(rows: list[dict], part: str) -> dict:
    c = Counter()
    for r in rows:
        c.update(r[part])
    d = lambda a, b: c[a] / c[b] if c[b] else float("nan")
    out = {"sheets": len(rows), "recall": d("matched", "gt"), "precision": d("matched", "pred"), "exact": d("exact", "matched"),
           "char": d("char", "matched"), "s": c["s"] / max(1, len(rows)), "crops": c["n"] / max(1, len(rows))}
    for k in ("label", "notes"):
        out[k + "_exact"], out[k + "_char"] = d(k + "_exact", k + "_n"), d(k + "_char", k + "_n")
    out["key"] = (out["label_char"] + out["notes_char"]) / 2
    return out


THREADS_NOTE = {"vision": "set by macOS, not capped (about 3.5 cores busy); not a Linux candidate", "rapidocr": "2: onnxruntime 2 intra-op and 1 inter-op, OpenCV 2",
                "rapidocr-ch": "2: onnxruntime 2 intra-op and 1 inter-op, OpenCV 2", "tesseract": "1 (this Homebrew build has no OpenMP)",
                "easyocr": "2: torch 2, OpenCV 2, CPU device", "doctr": "2: torch 2, OpenCV 2, CPU device"}


def _f(v: float, nd: int = 3) -> str:
    return "n/a" if v != v else f"{v:.{nd}f}"


def _rows(rec: dict, copy: str | None = None, standard: bool = False) -> list[dict]:
    return [s for s in rec["sheets"] if (copy is None or s["copy"] == copy) and (not standard or s["size"] == STANDARD)]


def _tables(recs: dict) -> list[str]:
    L = ["## Page pass (2x, every text on the sheet)", "", "| engine | copy | sheets | box recall | box precision | exact (matched) | char acc (matched) | label exact | label char | notes exact | notes char | s per sheet |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e, r in recs.items():
        for copy in ("clean", "scan"):
            m = summary(_rows(r, copy), "page")
            L.append(f"| {e} | {copy} | {m['sheets']} | " + " | ".join(_f(m[k]) for k in ("recall", "precision", "exact", "char", "label_exact", "label_char", "notes_exact", "notes_char")) + f" | {_f(m['s'], 2)} |")
    L += ["", "## Crop pass (3x, a window around every ground-truth device, same-line boxes merged)", "",
          "| engine | copy | sheets | crops per sheet | box recall | box precision | exact (matched) | char acc (matched) | label exact | label char | s per sheet |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e, r in recs.items():
        for copy in ("clean", "scan"):
            if r["crops"]:
                m = summary(_rows(r, copy), "crops")
                L.append(f"| {e} | {copy} | {m['sheets']} | {_f(m['crops'], 1)} | " + " | ".join(_f(m[k]) for k in ("recall", "precision", "exact", "char", "label_exact", "label_char")) + f" | {_f(m['s'], 2)} |")
    return L


def _fonts(recs: dict, data: str) -> list[str]:
    """page-pass label and notes accuracy by drawing font, clean / scan: the unseen-style split holds out Courier New"""
    font = {}
    for r in recs.values():
        for s in r["sheets"]:
            if s["sheet"] not in font:
                with open(os.path.join(data, s["sheet"][:-4] + ".json")) as fh:
                    font[s["sheet"]] = json.load(fh)["style"]["font"].replace(".ttf", "")
    fonts = sorted(set(font.values()))
    L = ["## By font (page pass; label char, then notes char; clean / scan)", "", "| engine | " + " | ".join(fonts) + " |", "|---|" + "---|" * len(fonts)]
    for e, r in recs.items():
        cells = []
        for f in fonts:
            m = {c: summary([s for s in _rows(r, c) if font[s["sheet"]] == f], "page") for c in ("clean", "scan")}
            cells.append(f"{_f(m['clean']['label_char'], 2)} / {_f(m['scan']['label_char'], 2)}; {_f(m['clean']['notes_char'], 2)} / {_f(m['scan']['notes_char'], 2)}")
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    return L + ["", "Sheets per font (each clean and scan): " + ", ".join(f"{f} {sum(v == f for k, v in font.items() if '-scan' not in k)}" for f in fonts) + "."]


def _rule(recs: dict) -> tuple[list[str], dict]:
    L = ["## The rule", "", f"Best character accuracy on the label and notes strings (the mean of the two groups, both read in the page pass, clean and scan pooled), "
         f"among the engines that read a standard sheet ({STANDARD[0]} x {STANDARD[1]}) in under about {LIMIT_S:.0f} s on 2 CPU threads; if two are close (within 0.01), "
         "the lighter install wins. Apple Vision is the Mac engine and the reference, not a Linux candidate.", "",
         "| engine | sheets | label char | notes char | label and notes | label char, crop pass | s per standard sheet, page | s per standard sheet, crops | within the time rule | install on Linux |", "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for e, r in recs.items():
        m, st = summary(_rows(r), "page"), summary(_rows(r, standard=True), "page")
        c = summary(_rows(r), "crops") if r["crops"] else None
        cs = summary(_rows(r, standard=True), "crops") if r["crops"] else None
        res[e] = dict(m, page_std=st["s"], crops_std=cs["s"] if cs else float("nan"), label_crop=c["label_char"] if c else float("nan"), ok=st["s"] < LIMIT_S)
        L.append(f"| {e} | {m['sheets']} | {_f(m['label_char'])} | {_f(m['notes_char'])} | {_f(m['key'])} | {_f(res[e]['label_crop'])} | {_f(st['s'], 2)} | "
                 f"{_f(res[e]['crops_std'], 2)} | {'yes' if res[e]['ok'] else 'no'} | {INSTALL.get(e, '')} |")
    return L, res


SPACE = {"rapidocr": [
    "- `requirements.txt`: `rapidocr-onnxruntime==1.4.4`, `onnxruntime==1.31.0` and `opencv-python==5.0.0.93`, the versions measured "
    "(rapidocr-onnxruntime also pulls pyclipper, shapely, PyYAML, six and tqdm); `huggingface_hub` is already there for the detector.",
    "- `packages.txt`: `libgl1` and `libglib2.0-0`, because rapidocr-onnxruntime requires opencv-python by name and its cv2 loads libGL.",
    "- Models: the detector and the direction classifier ship inside the wheel (15 MB). The English recogniser (9 MB) is fetched on first use "
    "from the Hugging Face hub (`SWHL/RapidOCR`, `PP-OCRv3/en_PP-OCRv3_rec_infer.onnx`, pinned revision) into the hub cache "
    "(`~/.cache/huggingface/hub`, or `$HF_HOME/hub`); warm the engine once when the app starts (one `ocr.ocr_image` call on a small blank "
    "image loads it and fetches the recogniser), so no request waits.",
    "- Engine: leave `SLD2QUOTE_OCR` unset (with no Vision tool on Linux the default is rapidocr) or set it to `rapidocr`. "
    "`ocr.THREADS` is 2, the free Space's vCPUs."],
    "tesseract": ["- `packages.txt`: `tesseract-ocr` (brings the English model).", "- `requirements.txt`: `pytesseract==0.3.13`.",
                  "- Models: none to fetch. Set `SLD2QUOTE_OCR=tesseract`; `OMP_THREAD_LIMIT` is set to 2 for Debian's OpenMP build."]}


def _choice(res: dict) -> list[str]:
    cands = sorted((e for e in res if e != "vision" and res[e]["ok"]), key=lambda e: -res[e]["key"])
    if not cands:
        return ["No candidate meets the time rule."]
    best, v, P = cands[0], res.get("vision"), []
    P.append(f"**{best}**: label and notes accuracy {_f(res[best]['key'])} (labels {_f(res[best]['label_char'])}, notes {_f(res[best]['notes_char'])}) "
             f"in {_f(res[best]['page_std'], 2)} s per standard sheet."
             + (f" Apple Vision on the same pages: {_f(v['key'])} (labels {_f(v['label_char'])}, notes {_f(v['notes_char'])}) in {_f(v['page_std'], 2)} s." if v else ""))
    P.append("Ranked under the rule: " + ", ".join(f"{e} {_f(res[e]['key'])} in {_f(res[e]['page_std'], 2)} s" for e in cands) + "."
             + (" The first two are within 0.01, so the lighter install decides." if len(cands) > 1 and res[best]["key"] - res[cands[1]]["key"] <= 0.01 else ""))
    out = [e for e in res if e != "vision" and not res[e]["ok"]]
    if out:
        P.append("Over the time rule: " + ", ".join(f"{e} ({_f(res[e]['page_std'], 1)} s per standard sheet, accuracy {_f(res[e]['key'])})" for e in out) + ".")
    if v:
        P.append(f"For the reader: {best} reads labels at {_f(res[best]['label_char'])} from the 2x page and {_f(res[best]['label_crop'])} from the 3x crops "
                 f"(crops add {_f(res[best]['crops_std'], 1)} s per standard sheet); Vision reads them at {_f(v['label_char'])} from the page and "
                 f"{_f(v['label_crop'])} from the crops, which is why the reader crops today.")
    return ["## The choice", ""] + [x for p in P for x in (p, "")] + ["## On a Hugging Face Space (Linux, 2 shared vCPUs)", ""] + SPACE.get(best, [f"- {INSTALL[best]}"])


def report(data: str, path: str) -> None:
    name = os.path.basename(os.path.normpath(data)); recs = {}
    for p in glob.glob(os.path.join(OUT, name + "-*.json")):
        with open(p) as fh:
            r = json.load(fh)
        recs[r["engine"]] = r
    recs = {e: recs[e] for e in ocr.ENGINES if e in recs}
    if not recs:
        raise SystemExit(f"no records for {name} in {OUT}")
    rule, res = _rule(recs)
    n = max(len(r["sheets"]) for r in recs.values())
    L = ["# OCR engines measured on our sheets", "", f"Written by `python3 -m sld2quote.evaluate_ocr --data {data}` from the records in `{OUT}`; synthetic sheets only, "
         f"nothing here is an R&B drawing. {n} sheets, a clean and a scan-like copy of each drawing; {sum(s['size'] == STANDARD for s in _rows(max(recs.values(), key=lambda r: len(r['sheets']))))} "
         f"of them (both copies) are standard {STANDARD[0]} x {STANDARD[1]} sheets, the rest are wider (medium boards and long rows of ways).", "",
         "Scoring: each engine reads every page as saved (scan copies keep their tilt of up to 1.2 degrees; the ground-truth boxes are mapped into the image) "
         "through the reader's own OCR call, so the boxes and the look-alike letter map are what the reader gets. A recognised box matches a ground-truth "
         "text box one to one when IoU >= 0.3 or either centre lies inside the other. Strings are compared after collapsing whitespace. "
         "Character accuracy is 1 minus the edit distance over the longer string. Label strings are those with a number and A, kA, kVA, kW or kVAr "
         "(device, busbar and CT labels); notes strings are the notes lines with kA, Form, IP or -pole (the four the reader parses). "
         "In the label and notes columns a missed string scores 0, so they combine finding and reading. The crop pass reads a window around "
         "every ground-truth device as the reader does around every symbol; text cut by a window's edge is left out of that window's score.", "",
         "| engine | measured | threads | model load s | load average, start and end | crop pass |", "|---|---|---|---|---|---|"]
    for e, r in recs.items():
        L.append(f"| {e} | {r['start']} to {r['end']} | {THREADS_NOTE.get(e, r['threads'])} | {r['load_s']} | {r['loadavg'][0]} / {r['loadavg'][-1]} | {'yes' if r['crops'] else 'no'} |")
    few = [f"{e} ({len(r['sheets'])} of the {n})" for e, r in recs.items() if len(r["sheets"]) < n]
    if few:
        L += ["", "Run on the standard sheets only (`--max-width 1600`), to spare the CPU for the training run: " + ", ".join(few) + "."]
    L += ["", f"Machine: {next(iter(recs.values()))['machine']}. CPU only: docTR and EasyOCR on torch's CPU device, RapidOCR on onnxruntime's CPU provider, and "
          "Vision's default path gave output bitwise identical to a build pinned to the CPU. Every candidate ran at 2 threads (OMP_NUM_THREADS=2, "
          "torch.set_num_threads(2), onnxruntime and OpenCV at 2), one engine at a time, while the detector's training run held the GPU and about four to five "
          "cores, so the times are a little pessimistic. A free Space's 2 shared vCPUs are slower than 2 M2 Max cores: check the times there.", ""]
    L += _tables(recs) + [""] + _fonts(recs, data) + [""] + rule + [""] + _choice(res)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(rule + [""] + _choice(res)[:5]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="measure OCR engines against the generator's ground truth")
    ap.add_argument("--data", default="data/baseline")
    ap.add_argument("--engines", default=",".join(dict.fromkeys([ocr.default_engine(), "rapidocr"])), help="comma list (default: this machine's engine and rapidocr), or none to only rebuild the report")
    ap.add_argument("--crops", action="store_true", help="also read a 3x window around every ground-truth device, as the reader does")
    ap.add_argument("--max-width", type=int, help="only sheets up to this width (a quick run on the standard sheets: 1600)")
    ap.add_argument("--out", default=os.path.join("eval", "OCR.md"))
    a = ap.parse_args(argv)
    for e in [e for e in a.engines.split(",") if e and e != "none"]:
        run_engine(a.data, e, a.crops, a.max_width)
    report(a.data, a.out)


if __name__ == "__main__":
    main()
