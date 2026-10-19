"""Reader: turns a sheet image plus OCR text boxes into devices, busbars, connections and notes, each with evidence.

Classical computer vision on purpose: busbars and connection lines by run-length analysis, symbols by connected
components after the lines are removed, classification by simple shape tests, labels by proximity to the symbol.
read_sheet is the one pipeline of both readers (prepare, page OCR, busbars and lines, symbols, OCR around each symbol,
label association, roles, notes); only the symbol finder is swapped: find_symbols here, the trained detector in
reader_v2.py. So the two readers are compared like for like, and everything downstream keeps the same interface.
"""
from __future__ import annotations
import json, math, re, os, time
from functools import partial
import numpy as np
from PIL import Image
from scipy import ndimage
from .classes import DEVICE_TO_CLASS

# OCR sometimes returns Cyrillic look-alikes for Latin capitals on isolated words; map them back.
LOOKALIKE = str.maketrans({"А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y"})


def _ocr_image(img: Image.Image, scale: float, tag: str) -> list[dict]:
    """OCR a PIL image at an upscale factor through the one OCR interface (ocr.py: Apple Vision on a Mac, RapidOCR
    elsewhere, or SLD2QUOTE_OCR), with look-alike letters mapped back."""
    from .ocr import ocr_image
    return [dict(b, text=b["text"].translate(LOOKALIKE)) for b in ocr_image(img, scale, tag)]


def run_ocr(image_path: str, scale: float = 2.0) -> list[dict]:
    img = Image.open(image_path).convert("L")
    boxes = _ocr_image(img, scale, "page")
    for i, b in enumerate(boxes):
        b["id"] = f"T{i + 1}"
    return boxes


def ocr_region(img: Image.Image, box: list, pad=(60, 40, 330, 90), scale: float = 3.0, tag: str = "crop") -> list[dict]:
    """OCR a window around a symbol: left, top, right, bottom padding in page pixels; boxes come back in page coordinates,
    and boxes on the same text line are merged left to right."""
    x, y, w, h = box
    x0, y0 = max(0, int(x - pad[0])), max(0, int(y - pad[1])); x1, y1 = min(img.width, int(x + w + pad[2])), min(img.height, int(y + h + pad[3]))
    crop = img.crop((x0, y0, x1, y1))
    res = _ocr_image(crop, scale, tag)
    for b in res:
        b["box"][0] += x0; b["box"][1] += y0
    # merge boxes whose vertical centres are within 8 px, in x order
    res.sort(key=lambda b: (round(b["box"][1] / 8), b["box"][0]))
    merged = []
    for b in res:
        if merged and abs((merged[-1]["box"][1] + merged[-1]["box"][3] / 2) - (b["box"][1] + b["box"][3] / 2)) <= 8 and b["box"][0] - (merged[-1]["box"][0] + merged[-1]["box"][2]) <= 30:
            m = merged[-1]; nx1 = max(m["box"][0] + m["box"][2], b["box"][0] + b["box"][2]); ny0 = min(m["box"][1], b["box"][1]); ny1 = max(m["box"][1] + m["box"][3], b["box"][1] + b["box"][3])
            m["text"] = m["text"] + " " + b["text"]; m["conf"] = min(m["conf"], b["conf"]); m["box"] = [m["box"][0], ny0, nx1 - m["box"][0], ny1 - ny0]
        else:
            merged.append(dict(b))
    return merged


# ---------------------------------------------------------------- image handling
def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def binarise(g: np.ndarray) -> np.ndarray:
    """Ink mask. Adaptive against a local mean so faded scans still threshold."""
    local = ndimage.uniform_filter(g, size=61)
    return (g < local - 28) & (g < 210)


def deskew_angle(b: np.ndarray) -> float:
    """Small-angle deskew by maximising the sharpness of the row projection (horizontal lines align)."""
    small = b[::4, ::4].astype(np.float32)
    best, best_score = 0.0, -1.0
    for a in np.arange(-2.0, 2.01, 0.1):
        r = ndimage.rotate(small, a, reshape=False, order=0)
        score = float(np.var(r.sum(axis=1)))
        if score > best_score:
            best, best_score = a, score
    return best


def prepare(path: str):
    g = load_gray(path)
    b = binarise(g)
    ang = deskew_angle(b)
    if abs(ang) >= 0.15:  # ang is the rotation that levels the page, counter-clockwise positive in ndimage and PIL alike
        im = Image.fromarray(g.astype(np.uint8)).rotate(ang, resample=Image.BICUBIC, fillcolor=255)
        g = np.asarray(im).astype(np.float32)
        b = binarise(g)
    return g, b, ang


# ---------------------------------------------------------------- lines
def _runs(v: np.ndarray):
    """start, end (exclusive) of True runs in a 1-D bool array"""
    d = np.diff(np.concatenate(([0], v.astype(np.int8), [0])))
    return [(int(s), int(e)) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]  # plain ints, so outputs stay JSON


def find_busbars(b: np.ndarray, min_len=220, min_thick=4):
    H, W = b.shape
    rows = {}
    for y in range(H):
        for s, e in _runs(b[y]):
            if e - s >= min_len:
                rows.setdefault(y, []).append((s, e))
    bands = []
    for y in sorted(rows):
        for s, e in rows[y]:
            for band in bands:
                if band["y2"] >= y - 1 and min(band["x2"], e) - max(band["x1"], s) > min_len * 0.6:
                    band["y2"] = y; band["x1"] = min(band["x1"], s); band["x2"] = max(band["x2"], e); break
            else:
                bands.append({"x1": int(s), "x2": int(e), "y1": y, "y2": y})
    out = []
    for i, band in enumerate(b for b in bands if b["y2"] - b["y1"] + 1 >= min_thick):
        band["y"] = (band["y1"] + band["y2"]) / 2; band["id"] = f"BUS-{i + 1}"
        out.append(band)
    return out


def find_vertical_lines(b: np.ndarray, min_len=30, max_thick=5):
    H, W = b.shape
    cols = {}
    for x in range(W):
        for s, e in _runs(b[:, x]):
            if e - s >= min_len:
                cols.setdefault(x, []).append((s, e))
    segs = []
    for x in sorted(cols):
        for s, e in cols[x]:
            for sg in segs:
                if sg["x2"] >= x - 1 and min(sg["y2"], e) - max(sg["y1"], s) > (e - s) * 0.6:
                    sg["x2"] = x; sg["y1"] = min(sg["y1"], s); sg["y2"] = max(sg["y2"], e); break
            else:
                segs.append({"x1": x, "x2": x, "y1": int(s), "y2": int(e)})
    return [s for s in segs if s["x2"] - s["x1"] + 1 <= max_thick]


def remove_lines(b: np.ndarray, busbars, vlines) -> np.ndarray:
    c = b.copy()
    for bb in busbars:
        c[max(0, bb["y1"] - 1): bb["y2"] + 2, bb["x1"]: bb["x2"]] = False
    for v in vlines:
        c[v["y1"]: v["y2"], max(0, v["x1"] - 1): v["x2"] + 2] = False
    return c


# ---------------------------------------------------------------- symbols
def _overlap_frac(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (ix * iy) / max(1.0, aw * ah)


def _line_score(crop: np.ndarray, p0, p1, tol=1) -> float:
    """fraction of samples along the segment p0->p1 that hit ink (with a tolerance band)"""
    n = 40; hits = 0
    H, W = crop.shape
    for t in np.linspace(0, 1, n):
        x = int(round(p0[0] + (p1[0] - p0[0]) * t)); y = int(round(p0[1] + (p1[1] - p0[1]) * t))
        y0, y1 = max(0, y - tol), min(H, y + tol + 1); x0, x1 = max(0, x - tol), min(W, x + tol + 1)
        if crop[y0:y1, x0:x1].any():
            hits += 1
    return hits / n


def _ring_score(crop: np.ndarray) -> float:
    H, W = crop.shape; cx, cy = (W - 1) / 2, (H - 1) / 2; r = min(W, H) / 2 - 1.5
    n = 48; hits = 0
    for k in range(n):
        a = 2 * math.pi * k / n
        x, y = int(round(cx + r * math.cos(a))), int(round(cy + r * math.sin(a)))
        if 0 <= x < W and 0 <= y < H and crop[max(0, y - 1): y + 2, max(0, x - 1): x + 2].any():
            hits += 1
    return hits / n


def find_symbols(b_nolines: np.ndarray, texts: list[dict]):
    lab, n = ndimage.label(b_nolines, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    symbols = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop; x0, x1 = sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if not (14 <= w <= 70 and 14 <= h <= 100):
            continue
        box = [x0, y0, w, h]
        if any(_overlap_frac(box, t["box"]) > 0.5 for t in texts):
            continue  # it is text
        crop = (lab[sl] == i + 1)
        area = int(crop.sum())
        if area < 40:
            continue
        d1 = _line_score(crop, (0, 0), (w - 1, h - 1)); d2 = _line_score(crop, (w - 1, 0), (0, h - 1))
        ring = _ring_score(crop)
        aspect = h / w
        kind, score = None, 0.0
        if max(d1, d2) >= 0.7 and 0.75 <= aspect <= 1.35:
            kind, score = "BREAKER", max(d1, d2)
        elif ring >= 0.6 and aspect >= 1.45:
            kind, score = "TRANSFORMER", ring
        elif ring >= 0.6:
            kind, score = "CIRCLE", ring
        if kind:
            symbols.append({"id": f"S{len(symbols) + 1}", "kind": kind, "box": [int(x0), int(y0), int(w), int(h)], "shape_score": round(float(score), 2)})
    return symbols


# ---------------------------------------------------------------- labels and parsing
RX_TYPE = re.compile(r"\b(ACB|MCCB|MCB|BS)[-\s]?(\d+)?", re.I)
RX_AMPS = re.compile(r"(\d{2,4})\s*A\b")
RX_POLES = re.compile(r"\b(\d)\s*P\b", re.I)
RX_KA = re.compile(r"(\d{2,3})\s*kA", re.I)
RX_KVA = re.compile(r"(\d{3,4})\s*kVA", re.I)
RX_FORM = re.compile(r"Form\s*(\d)\s*(?:Type\s*(\d)|([ab]))?", re.I)
RX_IP = re.compile(r"\bIP\s*(\d{2})\b", re.I)
RX_POLE_DEFAULT = re.compile(r"(\d)\s*-\s*pole", re.I)


def parse_label(s: str) -> dict:
    s = s.translate(LOOKALIKE)
    out = {}
    m = RX_TYPE.search(s)
    if m:
        out["type_word"] = m.group(1).upper()
    m = RX_KVA.search(s)
    if m:
        out["kva"] = int(m.group(1))
    else:
        m = RX_AMPS.search(s)
        if m:
            out["rating"] = int(m.group(1))
    m = RX_POLES.search(s)
    if m:
        out["poles"] = int(m.group(1))
    m = RX_KA.search(s)
    if m:
        out["kA"] = int(m.group(1))
    return out


def _center(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def nearest_texts(sym, texts, used):
    """texts to the right of the symbol (same row) or directly below it, nearest first"""
    sx, sy, sw, sh = sym["box"]; scx, scy = _center(sym["box"])
    cands = []
    for t in texts:
        if t["id"] in used:
            continue
        tx, ty, tw, th = t["box"]; tcx, tcy = _center(t["box"])
        right = tx >= sx + sw - 6 and tx - (sx + sw) <= 40 and abs(tcy - scy) <= sh * 0.9
        left = tx + tw <= sx + 6 and sx - (tx + tw) <= 40 and abs(tcy - scy) <= sh * 0.9
        below = abs(tcx - scx) <= max(60, tw / 2) and 0 <= ty - (sy + sh) <= 70
        if right or left or below:
            dist = math.hypot(tcx - scx, tcy - scy)
            cands.append((dist, t))
    return [t for _, t in sorted(cands, key=lambda c: c[0])]


def bus_for_x(x, busbars):
    for bb in busbars:
        if bb["x1"] - 6 <= x <= bb["x2"] + 6:
            return bb
    return None


def touches_bus(sym, busbars):
    sx, sy, sw, sh = sym["box"]
    return any(not (sx + sw < bb["x1"] or sx > bb["x2"]) and bb["y1"] - 2 <= sy + sh / 2 <= bb["y2"] + 2 for bb in busbars)


def line_from_symbol(sym, direction, vlines):
    """is there a vertical line leaving the symbol upward (-1) or downward (+1) that reaches a busbar?"""
    sx, sy, sw, sh = sym["box"]; cx = sx + sw / 2
    for v in vlines:
        if abs((v["x1"] + v["x2"]) / 2 - cx) <= 6:
            if direction < 0 and abs(v["y2"] - sy) <= 8:
                return v
            if direction > 0 and abs(v["y1"] - (sy + sh)) <= 8:
                return v
    return None


# symbol kind -> device type: the classical finder's shapes and the detector's classes; a breaker's type is its role
KIND_TYPE = {"TRANSFORMER": "TRANSFORMER", "CIRCLE": "GENERATOR"} | {c: t for t, c in DEVICE_TO_CLASS.items() if c != "breaker"}
SOURCES = {"TRANSFORMER", "GENERATOR"}                              # rated in kVA
AMPS = {"ACB", "MCCB", "BUS_SECTION", "FUSE_SWITCH", "ISOLATOR", "CT"}  # rated in amps, as the ground truth rates them
SWITCHES = {"ACB", "MCCB", "BUS_SECTION", "FUSE_SWITCH", "ISOLATOR"}   # poles and kA from the label
WAYS = {"MCCB", "FUSE_SWITCH", "ISOLATOR"}                           # an outgoing way names what it feeds
# kW (contactor, motor) and kVAr (capacitor) are not parsed by parse_label yet, so those ratings stay None


def role(sym, parsed, busbars, main_bus_y) -> str:
    """a breaker's device type from where it sits: in the busbar a bus-section, above it an incomer ACB, below it an
    outgoing MCCB unless its label says ACB"""
    if touches_bus(sym, busbars):
        return "BUS_SECTION"
    if _center(sym["box"])[1] < main_bus_y:
        return "ACB"
    return "MCCB" if parsed.get("type_word") != "ACB" else "ACB"


def make_device(sym, labels, busbars, vlines, main_bus_y) -> dict:
    """one symbol with its label texts as a device: type, ratings parsed from the labels, busbar and connection"""
    parsed = {}
    for t in labels:
        parsed.update({k: v for k, v in parse_label(t["text"]).items() if k not in parsed})
    dev = {"symbol": sym["id"], "box": sym["box"], "label_texts": [t["id"] for t in labels], "label": " | ".join(t["text"] for t in labels),
           "conf": round(float(min([sym["shape_score"]] + [t["conf"] for t in labels])), 2) if labels else round(sym["shape_score"] * 0.5, 2)}
    if "score" in sym:  # the detector's raw score, beside the calibrated shape_score that conf is made from
        dev["score"] = sym["score"]
    typ = role(sym, parsed, busbars, main_bus_y) if sym["kind"].lower() == "breaker" else KIND_TYPE[sym["kind"]]
    dev.update(type=typ, rating=parsed.get("kva") if typ in SOURCES else parsed.get("rating") if typ in AMPS else None)
    if typ in SWITCHES:
        dev.update(poles=parsed.get("poles"), kA=parsed.get("kA"))
    if typ in WAYS:
        # destination: the lower of the two labels when both are "below", else the second label
        dests = [t["text"] for t in labels if not RX_AMPS.search(t["text"]) and not RX_KVA.search(t["text"])]
        dev["dest"] = dests[0] if dests else ""
    cx, cy = _center(sym["box"]); bb = bus_for_x(cx, busbars)
    dev["on_bus"] = bb["id"] if bb else None
    dev["connected"] = bool(line_from_symbol(sym, +1 if cy < main_bus_y else -1, vlines)) or dev["type"] == "BUS_SECTION"
    return dev


def make_devices(symbols, local, texts, busbars, vlines, H) -> list[dict]:
    """label association, in the finder's order: each symbol's own OCR window first, else the nearest unused page texts"""
    devices, used = [], set()
    main_bus_y = np.median([bb["y"] for bb in busbars]) if busbars else H / 2
    for sym in symbols:
        labels = nearest_texts(sym, local.get(sym["id"], []), set())[:2] or nearest_texts(sym, texts, used)[:2]
        devices.append(make_device(sym, labels, busbars, vlines, main_bus_y))
        for t in labels:
            used.add(t["id"])
    for i, dv in enumerate(devices):
        dv["id"] = f"D{i + 1}"
    return devices


def symbol_texts(page_img, symbols) -> dict:
    """second OCR pass: a window around every symbol, upscaled, so small rating text is read reliably"""
    local = {}
    for sym in symbols:
        reg = ocr_region(page_img, sym["box"])
        for k, tb in enumerate(reg):
            tb["id"] = f"{sym['id']}t{k + 1}"
        local[sym["id"]] = reg
    return local


def label_busbars(busbars, texts):
    """busbar label (rating, kA) from the text that says BUSBAR"""
    for bb in busbars:
        bb["rating"] = None; bb["kA"] = None; bb["label_text"] = None
        for t in texts:
            if "BUSBAR" in t["text"].upper() and bb["y1"] - 60 <= t["box"][1] + t["box"][3] <= bb["y1"] + 5:
                p = parse_label(t["text"]); bb["rating"] = p.get("rating"); bb["kA"] = p.get("kA"); bb["label_text"] = t["id"]


def read_notes(texts) -> dict:
    """notes block, anchored on the word NOTES"""
    notes = {"form": None, "ip": None, "fault_kA": None, "poles_default": None, "evidence": []}
    anchor = next((t for t in texts if t["text"].strip().upper() == "NOTES"), None)
    if anchor:
        ax, ay = anchor["box"][0], anchor["box"][1]
        for t in texts:
            tx, ty = t["box"][0], t["box"][1]
            if ax - 10 <= tx <= ax + 600 and ay < ty <= ay + 170:
                s = t["text"].translate(LOOKALIKE)
                m = RX_FORM.search(s)
                if m and notes["form"] is None:
                    notes["form"] = f"Form {m.group(1)}" + (f" Type {m.group(2)}" if m.group(2) else (m.group(3) if m.group(3) else "")); notes["evidence"].append(t["id"])
                m = RX_IP.search(s)
                if m and notes["ip"] is None:
                    notes["ip"] = f"IP{m.group(1)}"; notes["evidence"].append(t["id"])
                m = RX_KA.search(s)
                if m and notes["fault_kA"] is None:
                    notes["fault_kA"] = int(m.group(1)); notes["evidence"].append(t["id"])
                m = RX_POLE_DEFAULT.search(s)
                if m and notes["poles_default"] is None:
                    notes["poles_default"] = int(m.group(1)); notes["evidence"].append(t["id"])
    return notes


def classical(page_img, ink, texts) -> list[dict]:
    """the classical symbol finder: shape tests on the ink that is left once the lines are removed"""
    return find_symbols(ink, texts)


def _timed(clock: dict, key: str, fn, *args):
    t = time.perf_counter(); out = fn(*args); clock[key] += time.perf_counter() - t
    return out


def read_sheet(image_path: str, texts: list[dict] | None = None, finder=classical, reader: str = "classical", crops: bool | None = None,
               progress=None) -> dict:
    """the reading pipeline of both readers. finder(page_img, ink_without_lines, texts) returns the symbols in the pixels
    of the prepared (deskewed) page, the frame of the OCR and the line finders; every OCR call goes through _ocr_image.
    crops: the second OCR pass around every symbol (None: on for vision, which reads small text better enlarged, off for
    the other engines, which read the page pass better and pay about 0.3 s per symbol for the crops; see eval/OCR.md);
    without it the labels come from the page texts. "seconds" splits the time into OCR and the symbol finder ("detector").
    progress, if given, is called with "text", "symbols" and "labels" as each of those steps finishes."""
    from .ocr import default_engine
    t0, clock, engine = time.perf_counter(), {"ocr": 0.0, "detector": 0.0}, os.environ.get("SLD2QUOTE_OCR") or default_engine()
    done = progress or (lambda step: None)
    if crops is None:
        crops = engine == "vision"
    g, b, ang = prepare(image_path)
    H, W = b.shape
    page_img = Image.fromarray(g.astype(np.uint8))
    texts = texts if texts is not None else _timed(clock, "ocr", _ocr_image, page_img, 2.0, "page")
    for i, tb in enumerate(texts):
        tb["id"] = f"T{i + 1}"
    done("text")
    busbars = find_busbars(b)
    vlines = find_vertical_lines(b)
    ink = remove_lines(b, busbars, vlines)
    symbols = _timed(clock, "detector", finder, page_img, ink, texts)
    done("symbols")
    local = _timed(clock, "ocr", symbol_texts, page_img, symbols) if crops else {}
    label_busbars(busbars, texts)
    devices = make_devices(symbols, local, texts, busbars, vlines, H)
    done("labels")
    notes = read_notes(texts)
    all_texts = texts + [tb for reg in local.values() for tb in reg]
    return {"image": os.path.basename(image_path), "size": [int(W), int(H)], "deskew_deg": round(float(ang), 2),
            "texts": all_texts, "busbars": busbars, "vlines_count": len(vlines), "symbols": symbols, "devices": devices, "notes": notes, "crops": bool(crops),
            "ocr_engine": engine, "reader": reader, "seconds": {"total": round(time.perf_counter() - t0, 2), **{k: round(v, 2) for k, v in clock.items()}}}


READERS = ("classical", "trained")
WEIGHTS = "weights/rtdetr_r18_full"  # the trained reader's detector unless --weights says otherwise


def add_reader_args(ap):
    """--reader, --weights and --device, shared by the command line and the evaluation"""
    ap.add_argument("--reader", choices=READERS, default="classical", help="symbol finder: classical shape tests or the trained detector")
    ap.add_argument("--weights", default=WEIGHTS, help="trained reader: detector weights directory")
    ap.add_argument("--device", help="trained reader: cpu or mps (default mps when available, else cpu)")
    ap.add_argument("--crops", type=int, choices=(0, 1), help="the OCR pass around every symbol: 1 on, 0 off (default: on for vision, off for the other engines)")


def choose(name: str = "classical", weights: str = WEIGHTS, device: str | None = None, crops: bool | int | None = None):
    """read_sheet of the named reader; the trained one is imported only when chosen, since it loads torch"""
    crops = None if crops is None else bool(crops)
    if name == "classical":
        return partial(read_sheet, crops=crops)
    from . import reader_v2
    return partial(reader_v2.read_sheet, weights=weights, device=device, crops=crops)


if __name__ == "__main__":
    import sys
    r = read_sheet(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != "texts"}, indent=1))
