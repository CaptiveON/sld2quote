"""Synthetic single-line diagram generator with ground truth.

Draws a low-voltage switchboard SLD the way consultants draw them (one line per circuit, symbol plus rating text),
with style variation and an optional scan-like degradation, and writes a JSON ground truth beside each sheet:
devices with pixel boxes, busbars, connections, the notes block (form, IP, fault level) and a short specification
with clauses that may deliberately conflict with the drawing.

Nothing here is an R&B drawing. Ratings, names and prices are plausible, not real.
"""
from __future__ import annotations
import json, math, os, random
from dataclasses import dataclass, field, asdict
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_DIR = "/System/Library/Fonts/Supplemental"
FONTS = ["Arial.ttf", "Arial Narrow.ttf", "Verdana.ttf", "Courier New.ttf", "Trebuchet MS.ttf"]
W, H = 1600, 1000

ACB_RATINGS = [1600, 2000, 2500, 3200, 4000]
MCCB_RATINGS = [100, 160, 250, 400, 630, 800]
KA_OPTIONS = [36, 50, 65, 80]
FORMS = ["Form 3b", "Form 4 Type 2", "Form 4 Type 4", "Form 4 Type 6", "Form 4 Type 7"]
IPS = ["IP31", "IP41", "IP42", "IP54"]
DESTS = ["DB-{n} wards", "DB-{n} theatres", "MCC-{n} chillers", "MCC-{n} AHU", "lifts", "kitchen", "pumps", "lighting",
         "data hall", "UPS input", "sub-board {n}", "spare"]


def font(name: str, size: int):
    p = os.path.join(FONT_DIR, name)
    return ImageFont.truetype(p if os.path.exists(p) else os.path.join(FONT_DIR, "Arial.ttf"), size)


@dataclass
class Device:
    id: str
    type: str                      # TRANSFORMER | GENERATOR | ACB | MCCB | BUS_SECTION
    box: list                      # [x, y, w, h] of the symbol
    rating: int | None = None      # amps for breakers, kVA for sources
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


def text(d, sh: Sheet, xy, s, f, anchor="la"):
    d.text(xy, s, fill="black", font=f, anchor=anchor)
    bb = d.textbbox(xy, s, font=f, anchor=anchor)
    sh.texts.append({"text": s, "box": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]})


def make_sheet(seed: int, scan: bool = False) -> tuple[Image.Image, Sheet]:
    rnd = random.Random(seed)
    style = {"font": rnd.choice(FONTS), "size": rnd.choice([17, 18, 20, 22]), "lw": rnd.choice([2, 2, 3]),
             "sym": rnd.choice([26, 30, 34]), "bus_lw": rnd.choice([5, 6, 8]), "label_pos": rnd.choice(["right", "right", "below"]),
             "scan": scan}
    sh = Sheet(id=f"sheet-{seed:04d}", style=style)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    f = font(style["font"], style["size"]); fs = font(style["font"], max(13, style["size"] - 5)); fb = font("Arial Bold.ttf", 20)
    lw, s = style["lw"], style["sym"]

    fault = rnd.choice(KA_OPTIONS); form = rnd.choice(FORMS); ip = rnd.choice(IPS); poles_default = rnd.choice([3, 4, 4])
    acb = rnd.choice(ACB_RATINGS)
    n_ways = rnd.randint(4, 9)
    has_bs = rnd.random() < 0.55
    has_gen = has_bs and rnd.random() < 0.7
    bus_y = 400
    bus_x1, bus_x2 = 140, 1460
    mid = (bus_x1 + bus_x2) // 2

    # --- supply: transformer -> ACB -> busbar A
    tx_cx, tx_y = 220, 90
    circle(d, tx_cx, tx_y, 22, lw); circle(d, tx_cx, tx_y + 30, 22, lw)
    kva = rnd.choice([1000, 1250, 1600, 2000, 2500])
    text(d, sh, (tx_cx + 36, tx_y - 8), f"TX-1 {kva} kVA", f); text(d, sh, (tx_cx + 36, tx_y + 16), "11 kV / 415 V", fs)
    sh.devices.append(Device("TX-1", "TRANSFORMER", [tx_cx - 22, tx_y - 22, 44, 74], rating=kva, label=f"TX-1 {kva} kVA", on_bus="BUS-A"))
    d.line([tx_cx, tx_y + 52, tx_cx, 200], fill="black", width=lw)
    breaker_symbol(d, tx_cx - s // 2, 200, s, lw)
    acb_label = f"ACB-1 {acb} A {poles_default}P"
    text(d, sh, (tx_cx + s // 2 + 12, 200 + s // 2 - 10), acb_label, f)
    text(d, sh, (tx_cx + s // 2 + 12, 200 + s // 2 + 12), f"Icu {fault} kA, withdrawable", fs)
    sh.devices.append(Device("ACB-1", "ACB", [tx_cx - s // 2, 200, s, s], rating=acb, poles=poles_default, kA=fault, label=acb_label, on_bus="BUS-A"))
    d.line([tx_cx, 200 + s, tx_cx, bus_y], fill="black", width=lw)
    sh.connections += [["TX-1", "ACB-1"], ["ACB-1", "BUS-A"]]

    # --- busbar(s)
    if has_bs:
        d.line([bus_x1, bus_y, mid - s // 2, bus_y], fill="black", width=style["bus_lw"])
        d.line([mid + s // 2, bus_y, bus_x2, bus_y], fill="black", width=style["bus_lw"])
        breaker_symbol(d, mid - s // 2, bus_y - s // 2, s, lw)
        bs_label = f"BS-1 {acb} A"
        text(d, sh, (mid, bus_y + s // 2 + 8), bs_label, fs, anchor="ma")
        sh.devices.append(Device("BS-1", "BUS_SECTION", [mid - s // 2, bus_y - s // 2, s, s], rating=acb, poles=poles_default, kA=fault, label=bs_label))
        sh.busbars.append({"id": "BUS-A", "rating": acb, "x1": bus_x1, "x2": mid - s // 2, "y": bus_y, "thickness": style["bus_lw"]})
        sh.busbars.append({"id": "BUS-B", "rating": acb, "x1": mid + s // 2, "x2": bus_x2, "y": bus_y, "thickness": style["bus_lw"]})
        sh.connections += [["BUS-A", "BS-1"], ["BS-1", "BUS-B"]]
    else:
        d.line([bus_x1, bus_y, bus_x2, bus_y], fill="black", width=style["bus_lw"])
        sh.busbars.append({"id": "BUS-A", "rating": acb, "x1": bus_x1, "x2": bus_x2, "y": bus_y, "thickness": style["bus_lw"]})
    text(d, sh, (bus_x1 + 10, bus_y - style["bus_lw"] - 24), f"BUSBAR A  {acb} A Cu  {fault} kA/1s", fb)

    # --- generator incomer on busbar B
    if has_gen:
        g_cx = 1380; circle(d, g_cx, 90, 24, lw)
        text(d, sh, (g_cx, 90), "G", fb, anchor="mm")
        gk = rnd.choice([500, 630, 800, 1000])
        text(d, sh, (g_cx - 36, 60), f"GEN-1 {gk} kVA", f, anchor="ra"); text(d, sh, (g_cx - 36, 86), "standby", fs, anchor="ra")
        sh.devices.append(Device("GEN-1", "GENERATOR", [g_cx - 24, 66, 48, 48], rating=gk, label=f"GEN-1 {gk} kVA", on_bus="BUS-B"))
        d.line([g_cx, 114, g_cx, 200], fill="black", width=lw)
        breaker_symbol(d, g_cx - s // 2, 200, s, lw)
        acb2 = rnd.choice([r for r in ACB_RATINGS if r <= acb])
        l2 = f"ACB-2 {acb2} A {poles_default}P ATS"
        text(d, sh, (g_cx - s // 2 - 12, 200 + s // 2 - 10), l2, f, anchor="ra")
        sh.devices.append(Device("ACB-2", "ACB", [g_cx - s // 2, 200, s, s], rating=acb2, poles=poles_default, kA=fault, label=l2, on_bus="BUS-B"))
        d.line([g_cx, 200 + s, g_cx, bus_y], fill="black", width=lw)
        sh.connections += [["GEN-1", "ACB-2"], ["ACB-2", "BUS-B"]]

    # --- outgoing ways
    usable_x1, usable_x2 = bus_x1 + 60, bus_x2 - 60
    xs = [int(usable_x1 + i * (usable_x2 - usable_x1) / (n_ways - 1)) for i in range(n_ways)]
    if has_bs:
        xs = [x for x in xs if abs(x - mid) > s + 40]  # keep clear of the bus-section symbol
    for i, x in enumerate(xs):
        r = rnd.choice(MCCB_RATINGS)
        poles = poles_default if rnd.random() < 0.8 else (3 if poles_default == 4 else 4)
        ka = fault if rnd.random() < 0.85 else rnd.choice(KA_OPTIONS)
        dest = rnd.choice(DESTS).format(n=i + 1)
        bus_id = "BUS-A" if (not has_bs or x < mid) else "BUS-B"
        sym_y = bus_y + rnd.choice([120, 150, 180])
        d.line([x, bus_y, x, sym_y], fill="black", width=lw)
        ss = max(20, s - 6)
        breaker_symbol(d, x - ss // 2, sym_y, ss, lw)
        lab = f"MCCB {r} A {poles}P" + (f" {ka} kA" if ka != fault else "")
        if style["label_pos"] == "right":
            text(d, sh, (x + ss // 2 + 10, sym_y + ss // 2 - 10), lab, f)
            text(d, sh, (x + ss // 2 + 10, sym_y + ss // 2 + 12), dest, fs)
        else:
            text(d, sh, (x, sym_y + ss + 10), lab, fs, anchor="ma")
            text(d, sh, (x, sym_y + ss + 30), dest, fs, anchor="ma")
        d.line([x, sym_y + ss, x, sym_y + ss + 40], fill="black", width=lw)
        did = f"OG-{i + 1}"
        sh.devices.append(Device(did, "MCCB", [x - ss // 2, sym_y, ss, ss], rating=r, poles=poles, kA=ka, label=lab, dest=dest, on_bus=bus_id))
        sh.connections.append([bus_id, did])

    # --- notes block (the rules), title block
    nx, ny = 60, 760
    d.rectangle([nx, ny, nx + 560, ny + 150], outline="black", width=lw)
    text(d, sh, (nx + 12, ny + 10), "NOTES", fb)
    std = rnd.choice(["BS EN 61439-2", "BS EN IEC 61439-2:2021"])
    note_lines = [f"1. Switchboard to {std}, {form}, {ip}.",
                  f"2. Prospective fault level {fault} kA for 1 s at the busbars.",
                  f"3. All MCCBs {poles_default}-pole unless stated.",
                  "4. Busbars hard-drawn copper, ratings as shown."]
    for k, ln in enumerate(note_lines):
        text(d, sh, (nx + 12, ny + 40 + k * 26), ln, fs)
    sh.notes = {"form": form, "ip": ip, "fault_kA": fault, "poles_default": poles_default, "standard": std}
    tx = 1040
    d.rectangle([tx, 800, 1540, 940], outline="black", width=lw)
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
        im = im.rotate(rnd.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor=255)
        im = im.filter(ImageFilter.GaussianBlur(rnd.uniform(0.4, 0.9)))
        px = im.load()
        for _ in range(int(W * H * 0.0015)):
            x, y = rnd.randrange(W), rnd.randrange(H); px[x, y] = rnd.choice([0, 40, 255])
        im = Image.eval(im, lambda v: int(60 + v * 0.72))
    return im, sh


def write_sheet(out_dir: str, seed: int, scan: bool = False) -> str:
    im, sh = make_sheet(seed, scan)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, sh.id + ("-scan" if scan else ""))
    im.save(stem + ".png")
    gt = asdict(sh); gt["devices"] = [asdict(x) if not isinstance(x, dict) else x for x in sh.devices]
    gt["image"] = os.path.basename(stem + ".png"); gt["size"] = [W, H]
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
    a = ap.parse_args()
    for i in range(a.n):
        stem = write_sheet(a.out, a.seed + i)
        if a.scan:
            write_sheet(a.out, a.seed + i, scan=True)
        print("wrote", stem)
