"""Synthetic single-line diagram generator with ground truth.

Draws a low-voltage switchboard SLD the way consultants draw them (one line per circuit, symbol plus rating text),
with style variation, two complexity levels and an optional scan-like degradation, and writes a JSON ground truth
beside each sheet: devices with pixel boxes, busbars, connections, the notes block (form, IP, fault level) and a short
specification with clauses that may deliberately conflict with the drawing.

Levels: "simple" is one transformer and 4 to 9 outgoing ways; "medium" adds a second transformer on busbar B, always a
bus-section, 10 to 14 ways and a cable size under every way. The page widens so no label runs into the next way.

Symbols follow IEC 60617. Breakers are drawn either as the standard switch with a cross or as the common box with a
diagonal (one style per sheet); ways may be fuse switches or isolators; some carry a motor starter (contactor and motor),
power-factor correction (capacitor and earth) or surge protection (arrester and earth); incomers carry a current
transformer with a meter and sometimes a protection relay; some feeders end at a downstream board. Seven symbol bodies
come from an open library (assets/symbols, CC BY 3.0, see ATTRIBUTION.md); the rest are drawn here the same way.
Ground-truth coordinates are in the drawing frame: a scan copy is rotated by style["rotate_deg"] about the page centre
after drawing, and image_box() maps a box into the saved image. A sheet's font comes from a named set (--fonts, FONT_SETS):
"classic", the default, is the five fonts every earlier sheet was drawn from, so earlier seeds reproduce byte for byte;
"wide" adds eleven more and is recorded as style["font_set"]. A wide sheet also writes the letters inside its symbols (G, M,
the meter units, I>) in its font's bold face, sizes its notes box to the longest note, and starts a plain feeder's labels
below its cable stub; classic sheets keep the earlier drawing (Arial Bold inside symbols, a 560 px notes box).

Nothing here is an R&B drawing. Ratings, names and prices are plausible, not real.
"""
from __future__ import annotations
import json, logging, math, os, random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_DIRS = ["/System/Library/Fonts/Supplemental", "/System/Library/Fonts"]
FONTS = ["Arial.ttf", "Arial Narrow.ttf", "Verdana.ttf", "Courier New.ttf", "Trebuchet MS.ttf"]  # every sheet drawn before 18 Oct
# a sheet draws its labels, notes and title block in one font from its set (a wide sheet also the letters inside its symbols, in
# that font's bold); "wide" adds faces a consultant's CAD drawing could use (a .ttc collection gives its first face: Avenir
# Next's is its bold); every one covers the text drawn here, "²" included
FONT_SETS = {"classic": FONTS,
             "wide": FONTS + ["Tahoma.ttf", "Georgia.ttf", "Times New Roman.ttf", "Futura.ttc", "Helvetica.ttc", "HelveticaNeue.ttc",
                              "Avenir Next.ttc", "Optima.ttc", "Geneva.ttf", "Menlo.ttc", "Monaco.ttf"]}
W, H = 1600, 1000

ACB_RATINGS = [1600, 2000, 2500, 3200, 4000]
MCCB_RATINGS = [100, 160, 250, 400, 630, 800]
KA_OPTIONS = [36, 50, 65, 80]
FORMS = ["Form 3b", "Form 4 Type 2", "Form 4 Type 4", "Form 4 Type 6", "Form 4 Type 7"]
IPS = ["IP31", "IP41", "IP42", "IP54"]
DESTS = ["DB-{n} wards", "DB-{n} theatres", "MCC-{n} chillers", "MCC-{n} AHU", "lifts", "kitchen", "pumps", "lighting",
         "data hall", "UPS input", "sub-board {n}", "spare"]
CABLES = {100: "35", 160: "70", 250: "120", 400: "240", 630: "2x185", 800: "2x240"}  # mm² XLPE/SWA by way rating, illustrative
MOTOR_KW = [11, 15, 22, 30, 37, 45, 55, 75]
MOTOR_LOADS = ["pump P-{n}", "fan F-{n}", "chiller CH-{n}", "compressor C-{n}"]
PFC_KVAR = [100, 150, 200, 250, 300]
WAY_WORD = {"MCCB": "MCCB", "FUSE_SWITCH": "FSW", "ISOLATOR": "ISOL"}  # how the label names each way device
log = logging.getLogger("sld2quote.generate")
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "symbols")


@lru_cache(maxsize=None)
def font_path(name: str) -> str:
    """the font file in the first macOS font folder that holds it; a missing one is drawn in Arial, with one warning per name"""
    for p in (os.path.join(d, name) for d in FONT_DIRS):
        if os.path.exists(p):
            return p
    log.warning("font %s is not in %s; drawing in Arial instead", name, " or ".join(FONT_DIRS))
    return os.path.join(FONT_DIRS[0], "Arial.ttf")


def font(name: str, size: int, index: int = 0):
    return ImageFont.truetype(font_path(name), size, index=index)  # index: the face in a .ttc collection


@lru_cache(maxsize=None)
def glyph_face(name: str) -> tuple[str, int]:
    """(file, face index) a wide sheet writes inside its symbols: the font's bold, as '<name> Bold.ttf' beside it or the face
    styled Bold in its .ttc collection, else its own face (Geneva and Monaco have no bold on this Mac)"""
    stem, ext = os.path.splitext(name)
    if os.path.exists(os.path.join(os.path.dirname(font_path(name)), stem + " Bold.ttf")):
        return stem + " Bold.ttf", 0
    if ext == ".ttc":
        for i in range(32):
            try:
                if font(name, 10, i).getname()[1] == "Bold":
                    return name, i
            except OSError:  # past the collection's last face
                break
    return name, 0


@dataclass
class Device:
    id: str
    type: str                      # TRANSFORMER | GENERATOR | ACB | MCCB | BUS_SECTION | FUSE_SWITCH | ISOLATOR | CONTACTOR
                                   # | MOTOR | CAPACITOR | SPD | EARTH | CT | METER | RELAY | BOARD
    box: list                      # [x, y, w, h] of the symbol
    rating: int | None = None      # amps for switching devices and CTs, kVA for sources, kW for starters and motors, kVAr for PFC
    poles: int | None = None
    kA: int | None = None
    label: str = ""
    dest: str = ""
    on_bus: str | None = None      # busbar id the device hangs from (outgoing ways) or feeds (incomers)


@dataclass
class Sheet:
    id: str
    style: dict
    devices: list = field(default_factory=list)
    busbars: list = field(default_factory=list)   # {id, rating, x1, x2, y, thickness}
    connections: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)      # form, ip, fault_kA, poles_default, standard
    texts: list = field(default_factory=list)      # every drawn string with its box (for OCR evaluation)
    spec: dict = field(default_factory=dict)       # clauses and the requirements they encode
    conflicts: list = field(default_factory=list)  # deliberate drawing-vs-spec disagreements


def breaker_symbol(d: ImageDraw.ImageDraw, x: int, y: int, s: int, lw: int):
    d.rectangle([x, y, x + s, y + s], outline="black", width=lw)
    d.line([x, y, x + s, y + s], fill="black", width=lw)


def circle(d, cx, cy, r, lw):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=lw)


def dashed(d, p0, p1, on: int = 5, off: int = 4):
    """a thin dashed line, as drawn for CT and control wiring"""
    (x0, y0), (x1, y1) = p0, p1
    n, t = max(1.0, math.hypot(x1 - x0, y1 - y0)), 0.0
    while t < n:
        a, b = t / n, min(t + on, n) / n
        d.line([x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b], fill="black", width=1)
        t += on + off


def next_rating(amps: float) -> int:
    return next((r for r in MCCB_RATINGS if r >= amps), MCCB_RATINGS[-1])


# --- IEC 60617 symbols: primitives in each symbol's own units, top terminal at the middle of the top edge. The scale is
# the library's own (px per unit at 1x, from its registry), so relative sizes stay as the library draws them.
def _library(name: str) -> tuple:
    root = ET.parse(os.path.join(ASSETS, name + ".svg")).getroot()
    prims = []
    for el in root:
        tag, a = el.tag.split("}")[-1], el.attrib
        num = lambda *keys: [float(a[k]) for k in keys]
        if tag == "line":
            x1, y1, x2, y2 = num("x1", "y1", "x2", "y2"); prims.append(("line", [(x1, y1), (x2, y2)]))
        elif tag == "ellipse":
            prims.append(("ellipse", num("cx", "cy", "rx", "ry")))
        elif tag == "rect":
            prims.append(("rect", num("x", "y", "width", "height")))
        elif tag == "polygon":
            prims.append(("polygon", [tuple(float(v) for v in p.split(",")) for p in a["points"].split()]))
        # text elements are left to the sheet, so the text lands in the ground truth
    return tuple(float(v) for v in root.get("viewBox").split()), prims


def _own() -> dict:
    """shapes the library lacks; switches use its switch frame, with the blade pivoting at the bottom terminal"""
    frame, blade = (3, 27, 16, 23), ("line", [(11, 50), (3, 27)])
    n = math.hypot(8, 23); ax, ay = -8 / n, -23 / n  # unit vector up the blade
    fuse = [(7 + i * 6.5 * ax - j * 2.8 * ay, 38.5 + i * 6.5 * ay + j * 2.8 * ax) for i, j in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return {"fuse-switch": (frame, [blade, ("line", [(8, 30), (14, 30)]), ("polygon", fuse)]),   # fuse-switch-disconnector
            "contactor": (frame, [blade, ("arc", (11, 30), 3, 180, 360)]),                         # contactor main contact
            "motor": ((0, 0, 48, 48), [("ellipse", (24, 24, 24, 24))]),                            # the sheet writes the M
            "meter": ((0, 0, 36, 36), [("ellipse", (18, 18, 18, 18))]),                            # indicating instrument
            "energy-meter": ((0, 0, 40, 30), [("rect", (0, 0, 40, 30)), ("line", [(0, 9), (40, 9)])])}  # integrating, kWh


SCALE = {"breaker": 2.0, "disconnector": 2.0, "fuse-switch": 2.0, "contactor": 2.0, "ct": 1.78, "ground": 1.6, "protection-relay": 1.0,
         "shunt-capacitor": 1.78, "surge-arrester": 2.4, "motor": 1.0, "meter": 1.0, "energy-meter": 1.0}
SYMBOLS = {**{n: _library(n) for n in ("breaker", "disconnector", "ct", "ground", "protection-relay", "shunt-capacitor", "surge-arrester")}, **_own()}


def draw_symbol(d, name: str, x: float, y: float, u: float, lw: int, rot: bool = False) -> tuple[list, tuple]:
    """draws `name` with its top terminal at (x, y), or its left terminal when rot (a quarter turn, for a bus-section),
    at u px per library unit; returns the symbol's tight box and its far terminal"""
    (vx, vy, vw, vh), prims = SYMBOLS[name]
    k, tx = SCALE[name] * u, vx + vw / 2

    def P(px, py):
        dx, dy = (px - tx) * k, (py - vy) * k
        return (x + dy, y - dx) if rot else (x + dx, y + dy)

    ext = []  # points bounding the ink, with the half line width that sticks out past them
    for kind, *g in prims:
        if kind in ("line", "polygon", "arc"):
            if kind == "arc":
                (cx, cy), r, a0, a1 = g
                p = [P(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in range(a0, a1 + 1, 15)]
            else:
                p = [P(*q) for q in g[0]]
            if kind == "polygon":
                d.polygon(p, outline="black", width=lw)
            else:
                d.line(p, fill="black", width=lw)
            ext += [(px, py, lw / 2) for px, py in p]
        else:  # ellipse or rect, outlined inside its bounds
            if kind == "ellipse":
                cx, cy, rx, ry = g[0]; c0, c1 = P(cx - rx, cy - ry), P(cx + rx, cy + ry)
            else:
                rx0, ry0, rw, rh = g[0]; c0, c1 = P(rx0, ry0), P(rx0 + rw, ry0 + rh)
            bb = [min(c0[0], c1[0]), min(c0[1], c1[1]), max(c0[0], c1[0]), max(c0[1], c1[1])]
            (d.ellipse if kind == "ellipse" else d.rectangle)(bb, outline="black", width=lw)
            ext += [(bb[0], bb[1], 0), (bb[2], bb[3], 0)]
    x0, y0 = math.floor(min(e[0] - e[2] for e in ext)), math.floor(min(e[1] - e[2] for e in ext))
    x1, y1 = math.ceil(max(e[0] + e[2] for e in ext)), math.ceil(max(e[1] + e[2] for e in ext))
    return [x0, y0, x1 - x0, y1 - y0], P(tx, vy + vh)


def text(d, sh: Sheet, xy, s, f, anchor="la"):
    d.text(xy, s, fill="black", font=f, anchor=anchor)
    bb = d.textbbox(xy, s, font=f, anchor=anchor)
    sh.texts.append({"text": s, "box": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]})


def image_box(box: list, size: list, rotate_deg: float = 0.0) -> list:
    """a ground-truth box (drawing frame) in the saved image's frame; scan copies are rotated about the page centre"""
    if not rotate_deg:
        return list(box)
    x, y, w, h = box; cx, cy = size[0] / 2, size[1] / 2
    a = math.radians(rotate_deg); c, s = math.cos(a), math.sin(a)
    pts = [((px - cx) * c + (py - cy) * s + cx, (py - cy) * c - (px - cx) * s + cy) for px in (x, x + w) for py in (y, y + h)]
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    return [round(x0), round(y0), round(max(p[0] for p in pts) - x0), round(max(p[1] for p in pts) - y0)]


def make_sheet(seed: int, scan: bool = False, level: str | None = None, fonts: str = "classic") -> tuple[Image.Image, Sheet]:
    rnd = random.Random(seed)
    level = level or ("medium" if random.Random(f"level-{seed}").random() < 0.5 else "simple")
    medium = level == "medium"
    style = {"font": rnd.choice(FONT_SETS[fonts]), "size": rnd.choice([17, 18, 20, 22]), "lw": rnd.choice([2, 2, 3]),
             "sym": rnd.choice([26, 30, 34]), "bus_lw": rnd.choice([5, 6, 8]), "label_pos": rnd.choice(["right", "right", "below"]),
             "scan": scan, "level": level, "breaker": rnd.choice(["iec", "box"])}
    wide = fonts != "classic"  # the wide set's own drawing rules (see the module notes); classic sheets stay as first drawn
    if wide:  # a classic sheet leaves the key out, so it stays byte-identical with those drawn before it existed
        style["font_set"] = fonts
    sh = Sheet(id=f"sheet-{seed:04d}", style=style)
    f = font(style["font"], style["size"]); fs = font(style["font"], max(13, style["size"] - 5)); fb = font("Arial Bold.ttf", 20)
    gname, gi = glyph_face(style["font"]) if wide else ("Arial Bold.ttf", 0)
    gf = lambda size: font(gname, size, gi)  # the letters inside symbols
    lw, s = style["lw"], style["sym"]
    ss = max(20, s - 6)  # outgoing breakers are drawn a little smaller than incomers
    u_in, u_out = s / 36, ss / 36  # px per symbol-library unit for incomers and outgoing ways
    right = style["label_pos"] == "right"
    style["boards"] = right and rnd.random() < 0.5  # some feeders end at a drawn downstream board (labels-right sheets)

    fault = rnd.choice(KA_OPTIONS); form = rnd.choice(FORMS); ip = rnd.choice(IPS); poles_default = rnd.choice([3, 4, 4])
    acb = rnd.choice(ACB_RATINGS)
    n_ways = rnd.randint(10, 14) if medium else rnd.randint(4, 9)
    has_bs = rnd.random() < 0.55 or medium
    has_gen = has_bs and rnd.random() < 0.7
    n_motor = rnd.choice([0, 1, 2, 2] if medium else [0, 0, 1, 2]); n_pfc = int(rnd.random() < 0.35); n_spd = int(rnd.random() < 0.5)
    fsw_share = rnd.choice([0.0, 0.0, 0.3, 0.8])  # share of ways on fuse switches rather than MCCBs
    # the page widens until the widest way label clears the next way's symbols and, for labels on the right, the page edge
    col = max((f if right else fs).getlength("MCCB 800 A 4P 80 kA"), fs.getlength("4C 2x240 mm² XLPE/SWA") if medium else 0)
    half = max(ss // 2 + 1, round(24 * u_out) + 2, 34 if style["boards"] else 0)  # widest half-symbol hanging from a way
    pitch = col + (2 * half + 20 if right else 20)
    edge = int(max(0, pitch - 200)) if right else 0
    w = max(W, int(400 + edge + (n_ways - 1) * pitch))
    im = Image.new("L", (w, H), 255)
    d = ImageDraw.Draw(im)
    bus_y = 400
    bus_x1, bus_x2 = 140, w - 140
    mid = (bus_x1 + bus_x2) // 2
    tx_y = 90

    def breaker(x: float, y: float, size: int, u: float, rot: bool = False):
        """the sheet's breaker style with its top terminal (left when rot) at (x, y); returns box and far terminal"""
        if style["breaker"] == "iec":
            return draw_symbol(d, "breaker", x, y, u, lw, rot)
        bx, by = (x, y - size // 2) if rot else (x - size // 2, y)
        breaker_symbol(d, bx, by, size, lw)
        return [bx, by, size, size], ((bx + size, y) if rot else (x, by + size))

    def metering(x: int, yb: float, acb_id: str, rating: int, relay: bool):
        """a current transformer on the incomer below its breaker, a meter beside it and sometimes a protection relay"""
        yc, um = round((yb + bus_y - 40) / 2), u_in * 0.85
        ctb, _ = draw_symbol(d, "ct", x, yc - 9 * SCALE["ct"] * u_in * 0.75, u_in * 0.75, lw)
        sh.devices.append(Device(f"{acb_id}-CT", "CT", ctb, rating=rating, label=f"CT {rating}/5A"))
        kind, unit, mw, mh = ("energy-meter", "kWh", 40, 30) if rnd.random() < 0.4 else ("meter", rnd.choice(["A", "V", "kW"]), 36, 36)
        mb, _ = draw_symbol(d, kind, ctb[0] + ctb[2] + 26 + mw * um / 2, yc - mh * um / 2, um, lw)
        dashed(d, (ctb[0] + ctb[2] + 2, yc), (mb[0] - 2, yc))
        fm = gf(max(10, round(mh * um * (0.34 if kind == "energy-meter" else 0.42))))
        text(d, sh, (mb[0] + mb[2] / 2, mb[1] + mb[3] * (0.66 if kind == "energy-meter" else 0.5)), unit, fm, anchor="mm")
        sh.devices.append(Device(f"{acb_id}-PM", "METER", mb, label=unit))
        sh.connections.append([f"{acb_id}-CT", f"{acb_id}-PM"])
        if relay:
            rb, _ = draw_symbol(d, "protection-relay", mb[0] + mb[2] + 22 + 16 * um, yc - 16 * um, um, lw)
            dashed(d, (mb[0] + mb[2] + 2, yc), (rb[0] - 2, yc))
            text(d, sh, (rb[0] + rb[2] / 2, rb[1] + rb[3] / 2), "I>", fm, anchor="mm")
            sh.devices.append(Device(f"{acb_id}-R", "RELAY", rb, label="I>"))
            sh.connections.append([f"{acb_id}-CT", f"{acb_id}-R"])
        text(d, sh, (mb[0], mb[1] + mb[3] + 6), f"CT {rating}/5A", fs)

    def tx_incomer(n: int, x: int, acb_id: str, bus_id: str):
        """transformer -> ACB (with metering) -> busbar"""
        circle(d, x, tx_y, 22, lw); circle(d, x, tx_y + 30, 22, lw)
        kva = rnd.choice([1000, 1250, 1600, 2000, 2500])
        text(d, sh, (x + 36, tx_y - 8), f"TX-{n} {kva} kVA", f); text(d, sh, (x + 36, tx_y + 16), "11 kV / 415 V", fs)
        sh.devices.append(Device(f"TX-{n}", "TRANSFORMER", [x - 22, tx_y - 22, 44, 74], rating=kva, label=f"TX-{n} {kva} kVA", on_bus=bus_id))
        d.line([x, tx_y + 52, x, 200], fill="black", width=lw)
        box, (_, yb) = breaker(x, 200, s, u_in)
        acb_label = f"{acb_id} {acb} A {poles_default}P"
        lx, ly = box[0] + box[2] + 12, box[1] + box[3] // 2
        text(d, sh, (lx, ly - 10), acb_label, f)
        text(d, sh, (lx, ly + 12), f"Icu {fault} kA, withdrawable", fs)
        sh.devices.append(Device(acb_id, "ACB", box, rating=acb, poles=poles_default, kA=fault, label=acb_label, on_bus=bus_id))
        d.line([x, yb, x, bus_y], fill="black", width=lw)
        sh.connections += [[f"TX-{n}", acb_id], [acb_id, bus_id]]
        if rnd.random() < 0.8:
            metering(x, yb, acb_id, acb, relay=medium and rnd.random() < 0.4)

    # --- supply: transformer -> ACB -> busbar A, and for medium boards a second one on busbar B
    bs_len = s if style["breaker"] == "box" else round(23 * SCALE["breaker"] * u_in)  # bus-section length along the bus
    tx_incomer(1, 220, "ACB-1", "BUS-A")
    if medium:
        tx_incomer(2, mid + bs_len // 2 + 80, "ACB-2", "BUS-B")

    # --- busbar(s)
    if has_bs:
        bx0 = mid - bs_len // 2
        d.line([bus_x1, bus_y, bx0, bus_y], fill="black", width=style["bus_lw"])
        bbox, (bx1, _) = breaker(bx0, bus_y, s, u_in, rot=True)
        bx1 = round(bx1)
        d.line([bx1, bus_y, bus_x2, bus_y], fill="black", width=style["bus_lw"])
        bs_label = f"BS-1 {acb} A"
        text(d, sh, (mid, bbox[1] + bbox[3] + 8), bs_label, fs, anchor="ma")
        sh.devices.append(Device("BS-1", "BUS_SECTION", bbox, rating=acb, poles=poles_default, kA=fault, label=bs_label))
        sh.busbars.append({"id": "BUS-A", "rating": acb, "x1": bus_x1, "x2": bx0, "y": bus_y, "thickness": style["bus_lw"]})
        sh.busbars.append({"id": "BUS-B", "rating": acb, "x1": bx1, "x2": bus_x2, "y": bus_y, "thickness": style["bus_lw"]})
        sh.connections += [["BUS-A", "BS-1"], ["BS-1", "BUS-B"]]
        text(d, sh, (bx1 + 10, bus_y - style["bus_lw"] - 24), f"BUSBAR B  {acb} A Cu  {fault} kA/1s", fb)
    else:
        d.line([bus_x1, bus_y, bus_x2, bus_y], fill="black", width=style["bus_lw"])
        sh.busbars.append({"id": "BUS-A", "rating": acb, "x1": bus_x1, "x2": bus_x2, "y": bus_y, "thickness": style["bus_lw"]})
    text(d, sh, (bus_x1 + 10, bus_y - style["bus_lw"] - 24), f"BUSBAR A  {acb} A Cu  {fault} kA/1s", fb)

    # --- generator incomer on busbar B
    if has_gen:
        g_cx = bus_x2 - 80; circle(d, g_cx, 90, 24, lw)
        text(d, sh, (g_cx, 90), "G", gf(20), anchor="mm")
        gk = rnd.choice([500, 630, 800, 1000])
        text(d, sh, (g_cx - 36, 60), f"GEN-1 {gk} kVA", f, anchor="ra"); text(d, sh, (g_cx - 36, 86), "standby", fs, anchor="ra")
        sh.devices.append(Device("GEN-1", "GENERATOR", [g_cx - 24, 66, 48, 48], rating=gk, label=f"GEN-1 {gk} kVA", on_bus="BUS-B"))
        d.line([g_cx, 114, g_cx, 200], fill="black", width=lw)
        box, (_, yb) = breaker(g_cx, 200, s, u_in)
        acb2 = rnd.choice([r for r in ACB_RATINGS if r <= acb])
        gid = "ACB-3" if medium else "ACB-2"
        l2 = f"{gid} {acb2} A {poles_default}P ATS"
        text(d, sh, (box[0] - 12, box[1] + box[3] // 2 - 10), l2, f, anchor="ra")
        sh.devices.append(Device(gid, "ACB", box, rating=acb2, poles=poles_default, kA=fault, label=l2, on_bus="BUS-B"))
        d.line([g_cx, yb, g_cx, bus_y], fill="black", width=lw)
        sh.connections += [["GEN-1", gid], [gid, "BUS-B"]]
        if rnd.random() < 0.5:
            metering(g_cx, yb, gid, acb2, relay=False)

    # --- outgoing ways: an MCCB, fuse switch or isolator, and below some of them a motor starter, power-factor
    # correction or surge protection
    usable_x1, usable_x2 = bus_x1 + 60, bus_x2 - 60 - edge
    xs = [int(usable_x1 + i * (usable_x2 - usable_x1) / (n_ways - 1)) for i in range(n_ways)]
    if has_bs:
        xs = [x for x in xs if abs(x - mid) > max(s, bs_len) + 40]  # keep clear of the bus-section symbol
    chains = [None] * len(xs)
    for j, i in enumerate(rnd.sample(range(len(xs)), min(len(xs), n_motor + n_pfc + n_spd))):
        chains[i] = "motor" if j < n_motor else "pfc" if j < n_motor + n_pfc else "spd"
    for i, x in enumerate(xs):
        chain, did = chains[i], f"OG-{i + 1}"
        bus_id = "BUS-A" if (not has_bs or x < mid) else "BUS-B"
        kind = "FUSE_SWITCH" if rnd.random() < fsw_share else "ISOLATOR" if chain is None and rnd.random() < 0.1 else "MCCB"
        poles = poles_default if rnd.random() < 0.8 else (3 if poles_default == 4 else 4)
        ka = fault if kind != "MCCB" or rnd.random() < 0.85 else rnd.choice(KA_OPTIONS)
        if chain == "motor":
            kw = rnd.choice(MOTOR_KW); r, poles = next_rating(4 * kw), 3
            extra, dest = f"{'DOL' if kw <= 37 else 'S/D'} {kw} kW", rnd.choice(MOTOR_LOADS).format(n=i + 1)
            parts = [("contactor", "CONTACTOR", "K", kw, extra), ("motor", "MOTOR", "M", kw, "M")]
        elif chain == "pfc":
            kvar = rnd.choice(PFC_KVAR); r, poles = next_rating(2 * kvar), 3
            extra, dest = f"PFC {kvar} kVAr", ""
            parts = [("shunt-capacitor", "CAPACITOR", "C", kvar, extra), ("ground", "EARTH", "E", None, "")]
        elif chain == "spd":
            r, extra, dest = 100, "SPD Type 1+2", ""
            parts = [("surge-arrester", "SPD", "F", None, extra), ("ground", "EARTH", "E", None, "")]
        else:
            r, extra, dest, parts = rnd.choice(MCCB_RATINGS), "", rnd.choice(DESTS).format(n=i + 1), []
        sym_y = bus_y + (100 if chain else rnd.choice([120, 150, 180]))
        d.line([x, bus_y, x, sym_y], fill="black", width=lw)
        if kind == "MCCB":
            box, (_, yb) = breaker(x, sym_y, ss, u_out)
        else:
            box, (_, yb) = draw_symbol(d, "fuse-switch" if kind == "FUSE_SWITCH" else "disconnector", x, sym_y, u_out, lw)
        lab = f"{WAY_WORD[kind]} {r} A {poles}P" + (f" {ka} kA" if ka != fault else "")
        sh.devices.append(Device(did, kind, box, rating=r, poles=poles, kA=ka if kind == "MCCB" else None, label=lab, dest=dest, on_bus=bus_id))
        sh.connections.append([bus_id, did])
        bottom, prev = yb, did
        for k, (name, typ, suffix, rating, lbl) in enumerate(parts):  # the chain hangs below the protective device
            gap = 16 if k == 0 else 12
            d.line([x, bottom, x, bottom + gap], fill="black", width=lw)
            pb, (_, bottom) = draw_symbol(d, name, x, bottom + gap, u_out, lw)
            if typ == "MOTOR":
                text(d, sh, (pb[0] + pb[2] / 2, pb[1] + pb[3] / 2), "M", gf(max(10, round(pb[3] * 0.42))), anchor="mm")
            sh.devices.append(Device(f"{did}-{suffix}", typ, pb, rating=rating, label=lbl))
            sh.connections.append([prev, f"{did}-{suffix}"]); prev = f"{did}-{suffix}"
        if not parts:  # a plain feeder: the outgoing cable, sometimes ending at a drawn downstream board
            board = style["boards"] and dest[:3] in ("DB-", "MCC", "sub") and rnd.random() < 0.6
            stub = 60 if board else 40
            d.line([x, yb, x, yb + stub], fill="black", width=lw)
            if wide:  # labels below start under the stub, so the cable never runs through them (boards come with labels right)
                bottom = yb + stub
            if board:
                name = f"SB-{i + 1}" if dest.startswith("sub") else dest.split(" ")[0]
                bw, bh = max(round(76 * u_out), round(fs.getlength(name)) + 14), max(round(34 * u_out), fs.size + 10)
                d.rectangle([x - bw // 2, yb + stub, x - bw // 2 + bw, yb + stub + bh], outline="black", width=lw)
                text(d, sh, (x, yb + stub + bh / 2), name, fs, anchor="mm")
                sh.devices.append(Device(f"{did}-DB", "BOARD", [x - bw // 2, round(yb + stub), bw, bh], label=name))
                sh.connections.append([did, f"{did}-DB"])
        lines = [lab] + [t for t in (extra, dest) if t]
        if medium and chain != "spd":
            lines.append(f"{'4C' if poles == 4 else '3C'} {CABLES[r]} mm² XLPE/SWA")
        for k, t in enumerate(lines):
            if right:
                text(d, sh, (x + half + 10, box[1] + box[3] // 2 - 10 + 22 * k), t, f if k == 0 else fs)
            else:
                text(d, sh, (x, bottom + 10 + 20 * k), t, fs, anchor="ma")

    # --- notes block (the rules), title block
    nx, ny = 60, 760
    std = rnd.choice(["BS EN 61439-2", "BS EN IEC 61439-2:2021"])
    note_lines = [f"1. Switchboard to {std}, {form}, {ip}.",
                  f"2. Prospective fault level {fault} kA for 1 s at the busbars.",
                  f"3. All MCCBs {poles_default}-pole unless stated.",
                  "4. Busbars hard-drawn copper, ratings as shown."]
    nw = 24 + math.ceil(max(fs.getlength(t) for t in note_lines)) if wide else 560  # 12 px either side of the longest note
    d.rectangle([nx, ny, nx + nw, ny + 150], outline="black", width=lw)
    text(d, sh, (nx + 12, ny + 10), "NOTES", fb)
    for k, ln in enumerate(note_lines):
        text(d, sh, (nx + 12, ny + 40 + k * 26), ln, fs)
    sh.notes = {"form": form, "ip": ip, "fault_kA": fault, "poles_default": poles_default, "standard": std}
    tx = w - 560
    d.rectangle([tx, 800, w - 60, 940], outline="black", width=lw)
    text(d, sh, (tx + 12, 810), f"PROJECT: {rnd.choice(['St Mary’s Hospital', 'Northgate Data Hall', 'Westside Retail Park', 'Quayside Mill', 'Harbour Pumping Station'])}", fs)
    text(d, sh, (tx + 12, 840), f"DRAWING: E-{rnd.randint(100, 999)}-SLD  REV {rnd.choice('ABC')}", fs)
    text(d, sh, (tx + 12, 870), "MAIN LV SWITCHBOARD, SINGLE LINE DIAGRAM", fs)
    text(d, sh, (tx + 12, 900), f"SHEET {sh.id}  (synthetic, for sld2quote)", fs)

    # --- specification (text) with possible conflicts
    spec_fault = fault; spec_form = form; spec_poles = poles_default
    if rnd.random() < 0.35:
        spec_fault = rnd.choice([k for k in KA_OPTIONS if k != fault]); sh.conflicts.append({"field": "fault_kA", "drawing": fault, "spec": spec_fault})
    if rnd.random() < 0.2:
        spec_form = rnd.choice([fm for fm in FORMS if fm != form]); sh.conflicts.append({"field": "form", "drawing": form, "spec": spec_form})
    clauses = [
        ("7.1", f"The switchboard shall be designed, manufactured and verified in accordance with {std}."),
        ("7.2", f"Internal separation shall be {spec_form} as defined in the UK National Annex."),
        ("7.3", f"The assembly shall withstand a prospective short-circuit current of {spec_fault} kA for 1 second at the main busbars."),
        ("7.4", f"Outgoing moulded-case circuit breakers shall be {spec_poles}-pole with adjustable thermal-magnetic or electronic trip units."),
        ("7.5", f"Degree of protection shall be not less than {ip} with doors closed."),
        ("7.6", "Main busbars shall be hard-drawn high-conductivity copper, tin-plated at joints."),
        ("7.7", "The contractor shall provide type-test or design-verification certificates prior to manufacture."),
        ("7.8", rnd.choice(["Witness testing at the manufacturer's works shall be allowed for.", "Two years' spares for all trip units shall be included.", "Enclosure finish RAL 7035, front access only."])),
    ]
    sh.spec = {"clauses": [{"id": c, "text": t} for c, t in clauses],
               "requirements": {"standard": std, "form": spec_form, "fault_kA": spec_fault, "poles": spec_poles, "ip": ip,
                                "busbar_material": "copper", "certificates": True, "extra": clauses[-1][1]}}

    # --- degrade to a scan if asked
    if scan:
        style["rotate_deg"] = round(rnd.uniform(-1.2, 1.2), 3)
        im = im.rotate(style["rotate_deg"], resample=Image.BICUBIC, fillcolor=255)
        im = im.filter(ImageFilter.GaussianBlur(rnd.uniform(0.4, 0.9)))
        px = im.load()
        for _ in range(int(w * H * 0.0015)):
            x, y = rnd.randrange(w), rnd.randrange(H); px[x, y] = rnd.choice([0, 40, 255])
        im = Image.eval(im, lambda v: int(60 + v * 0.72))
    return im, sh


def write_sheet(out_dir: str, seed: int, scan: bool = False, level: str | None = None, fonts: str = "classic") -> str:
    im, sh = make_sheet(seed, scan, level, fonts)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, sh.id + ("-scan" if scan else ""))
    im.save(stem + ".png")
    gt = asdict(sh); gt["devices"] = [asdict(x) if not isinstance(x, dict) else x for x in sh.devices]
    gt["image"] = os.path.basename(stem + ".png"); gt["size"] = list(im.size)
    with open(stem + ".json", "w") as fh:
        json.dump(gt, fh, indent=1)
    with open(stem + "-spec.txt", "w") as fh:
        fh.write(f"SPECIFICATION EXTRACT, SECTION 7: LV SWITCHBOARD ({sh.id})\n\n")
        for c in sh.spec["clauses"]:
            fh.write(f"{c['id']} {c['text']}\n")
    return stem


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="generate synthetic SLD sheets with ground truth")
    ap.add_argument("--out", default="data"); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scan", action="store_true", help="also write a scan-like degraded copy of each sheet")
    ap.add_argument("--level", choices=["simple", "medium", "mix"], default="mix", help="drawing complexity; mix picks one per seed")
    ap.add_argument("--fonts", choices=list(FONT_SETS), default="classic", help="the font set a sheet's font is drawn from")
    a = ap.parse_args()
    level = None if a.level == "mix" else a.level
    for i in range(a.n):
        stem = write_sheet(a.out, a.seed + i, level=level, fonts=a.fonts)
        if a.scan:
            write_sheet(a.out, a.seed + i, scan=True, level=level, fonts=a.fonts)
        print("wrote", stem)
