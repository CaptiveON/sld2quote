"""COCO training set for the symbol detector, cut from generated sheets.

Every sheet image, clean and scan copy, is cut into square tiles (640 px with a quarter overlap, by tiles.py, which the trained
reader's inference shares). A symbol's box goes into each tile that holds at least 60% of it, clipped to the tile; tiles with no
box are kept at --empty-keep, so the notes block, title block and busbar labels are also seen as background. Scan boxes go
through generate.image_box() so that they match the rotated image; clean boxes are used as drawn.

Splits are by whole sheet, in ascending seed order. A sheet drawn in the held-out font or label layout joins the unseen pool, and
test_unseen takes the first --unseen of those and val_unseen, when --val-unseen asks for it, the next ones, so that a model can
be selected on an unseen style without touching test_unseen; the others fill train, val and test in turn. train, val, test_unseen
and val_unseen hold both copies of their sheets; test_clean and test_scan are the two copies of the same test sheets, so those columns compare like for
like. Sheets beyond the requested counts are left out and counted in the manifest.

Writes, under --out: <split>/images/<sheet-id>[-scan]_<x>_<y>.png (grey, mode L) and <split>/annotations.json in standard COCO
(file_name is relative to images/; image and annotation ids count from 1, since pycocotools scores a match to annotation id 0 as
a miss; category ids are 0-based in classes.CLASSES order). Extra keys: each image carries sheet, copy, tile_x, tile_y, level,
breaker, font and label_pos, each annotation device_id, device_type and visible_frac. manifest.json records the parameters, the
counts and sheet ids per split and the sheets left unused; contact_sheet.png shows sampled tiles with their boxes, for checking
the labels by eye. Synthetic drawings only.

Run: python3 -m sld2quote.export_coco --pool data/pool --out data/coco
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, random, shutil, time
from collections import Counter
from datetime import datetime
from PIL import Image, ImageDraw
from . import __version__
from .classes import CLASSES, CLASS_ID, DEVICE_TO_CLASS
from .generate import FONTS, font, image_box
from .tiles import clip_box, cut_tile, tile_grid

SPLITS = ["train", "val", "test_clean", "test_scan", "test_unseen"]
VAL_UNSEEN = "val_unseen"  # written only when --val-unseen asks for sheets, so an export without it is as before
MIN_VISIBLE = 0.6  # share of a symbol's box that must lie inside a tile for the tile to carry its label
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (0, 128, 128), (240, 50, 230),
           (128, 128, 0), (170, 110, 40), (0, 0, 128), (128, 0, 0), (200, 160, 0), (0, 170, 220), (110, 110, 110)]  # by class


def read_pool(pool: str) -> list[dict]:
    """one small record per sheet, in ascending seed order, with the ground-truth path of each copy; the rest of each JSON is
    dropped at once, so the pool is never held in memory"""
    sheets = {}
    for path in sorted(glob.glob(os.path.join(pool, "sheet-*.json"))):
        with open(path) as fh:
            gt = json.load(fh)
        st = gt["style"]
        s = sheets.setdefault(gt["id"], {"id": gt["id"], "seed": int(gt["id"].split("-")[1]), "font": st["font"], "label_pos": st["label_pos"]})
        s["scan" if st["scan"] else "clean"] = path
    if not sheets:
        raise SystemExit(f"no sheet-*.json in {pool}")
    lone = [k for k, s in sheets.items() if "clean" not in s or "scan" not in s]
    if lone:
        raise SystemExit(f"{len(lone)} sheets lack a clean or a scan copy, {lone[0]} first; generate the pool with --scan")
    return sorted(sheets.values(), key=lambda s: s["seed"])


def assign(sheets: list, held_font: str, held_pos: str, n_train: int, n_val: int, n_test: int, n_unseen: int,
           n_val_unseen: int = 0) -> tuple[dict, dict]:
    """the splits as (sheet, copy) pairs, filled in ascending seed order, and the count of sheets left over; a pool too small
    for the requested sizes stops the export before anything is written, rather than train or score on short splits"""
    unseen_style = lambda s: s["font"] == held_font or s["label_pos"] == held_pos
    held, rest = [s for s in sheets if unseen_style(s)], [s for s in sheets if not unseen_style(s)]
    a, b, c = n_train, n_train + n_val, n_train + n_val + n_test
    u = n_unseen + n_val_unseen
    if len(rest) < c or len(held) < u:
        raise SystemExit(f"the pool has {len(rest)} regular and {len(held)} held-out-style sheets; {c} and {u} were asked for")
    both = lambda group: [(s, copy) for s in group for copy in ("clean", "scan")]
    splits = {"train": both(rest[:a]), "val": both(rest[a:b]), "test_clean": [(s, "clean") for s in rest[b:c]],
              "test_scan": [(s, "scan") for s in rest[b:c]], "test_unseen": both(held[:n_unseen])}
    if n_val_unseen:
        splits[VAL_UNSEEN] = both(held[n_unseen:u])
    return splits, {"held_out_style": len(held[u:]), "regular": len(rest[c:])}


def keep_empty(name: str, seed: int, frac: float) -> bool:
    """whether a tile with no box is kept: a stable hash of its file name and the seed (Python's hash() changes from one
    process to the next), so a rerun keeps the same tiles"""
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big") < frac * 2 ** 64


def export_split(name: str, items: list, out: str, tile: int, overlap: float, empty_keep: float, seed: int) -> dict:
    """cuts one split's sheet images into tiles, one image open at a time, writes them with the split's COCO file and returns
    the split's manifest entry"""
    img_dir = os.path.join(out, name, "images")
    os.makedirs(img_dir)
    images, anns, empty = [], [], Counter()
    for sheet, copy in items:
        with open(sheet[copy]) as fh:
            gt = json.load(fh)
        st = gt["style"]
        boxes = [(d, d["box"] if copy == "clean" else image_box(d["box"], gt["size"], st["rotate_deg"])) for d in gt["devices"]]
        with Image.open(os.path.join(os.path.dirname(sheet[copy]), gt["image"])) as f:
            page = f.convert("L")
        if list(page.size) != gt["size"]:
            raise SystemExit(f"{gt['image']} is {page.size} px but its ground truth says {gt['size']}")
        stem = os.path.splitext(gt["image"])[0]
        for x, y in tile_grid(*page.size, tile, overlap):
            fname = f"{stem}_{x}_{y}.png"
            hits = [(d, c) for d, b in boxes if (c := clip_box(b, x, y, tile, MIN_VISIBLE))]
            if not hits:
                kept = keep_empty(fname, seed, empty_keep)
                empty["kept" if kept else "dropped"] += 1
                if not kept:
                    continue
            cut_tile(page, x, y, tile).save(os.path.join(img_dir, fname))
            images.append({"id": len(images) + 1, "file_name": fname, "width": tile, "height": tile, "sheet": gt["id"], "copy": copy,
                           "tile_x": x, "tile_y": y, "level": st["level"], "breaker": st["breaker"], "font": st["font"], "label_pos": st["label_pos"]})
            for d, (bb, frac) in hits:
                anns.append({"id": len(anns) + 1, "image_id": len(images), "category_id": CLASS_ID[DEVICE_TO_CLASS[d["type"]]], "bbox": bb,
                             "area": bb[2] * bb[3], "iscrowd": 0, "device_id": d["id"], "device_type": d["type"], "visible_frac": round(frac, 4)})
    coco = {"info": {"description": f"sld2quote symbol tiles, {name} split, cut from synthetic drawings", "version": __version__},
            "licenses": [], "images": images, "annotations": anns, "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
    with open(os.path.join(out, name, "annotations.json"), "w") as fh:
        json.dump(coco, fh)
    per_class, ids = Counter(a["category_id"] for a in anns), list(dict.fromkeys(s["id"] for s, _ in items))
    return {"sheets": len(ids), "images": len(images), "boxes": len(anns), "boxes_per_class": {n: per_class[i] for i, n in enumerate(CLASSES)},
            "empty_tiles_kept": empty["kept"], "empty_tiles_dropped": empty["dropped"], "sheet_ids": ids}


def _meets(a: list, b: list) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _draw_labels(d: ImageDraw.ImageDraw, boxes: list, f, lim: tuple):
    """each box in its class colour, its name on a filled tag above, below, left or right of it, whichever first covers no
    other box or tag; a box cut by the tile edge also shows the share of it inside the tile"""
    taken = [[x0 - 1, y0 - 1, x1 + 1, y1 + 1] for (x0, y0, x1, y1), _ in boxes]
    for r, (_, a) in zip(taken, boxes):
        d.rectangle(r, outline=PALETTE[a["category_id"]], width=2)
    for (x0, y0, x1, y1), a in boxes:
        c = PALETTE[a["category_id"]]
        txt = CLASSES[a["category_id"]] + ("" if a["visible_frac"] == 1 else f" {a['visible_frac']:.0%}")
        tw, th = d.textlength(txt, font=f) + 6, 16
        spots = [[x0 - 1, y0 - 2 - th], [x0 - 1, y1 + 2], [x0 - 2 - tw, y0 - 1], [x1 + 2, y0 - 1]]
        fits = [[px, py, px + tw, py + th] for px, py in spots if lim[0] <= px and lim[1] <= py and px + tw <= lim[2] and py + th <= lim[3]]
        free = [r for r in fits if not any(_meets(r, q) for q in taken)]
        r = (free or fits or [[x0 - 1, y0 - 1, x0 - 1 + tw, y0 - 1 + th]])[0]
        taken.append(r)
        d.rectangle(r, fill=c)
        dark = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 150
        d.text((r[0] + 3, r[1] + th / 2), txt, fill="white" if dark else "black", font=f, anchor="lm")


def contact_sheet(out: str, seed: int, held: str, n_train: int = 20, n_unseen: int = 4, cols: int = 4, cell: int = 588) -> str:
    """sampled train and test_unseen tiles drawn with their boxes and class names, read back from the written files so the
    picture checks what the trainer will load"""
    rnd, picks, n = random.Random(seed), [], {}
    for split, k in (("train", n_train), ("test_unseen", n_unseen)):
        with open(os.path.join(out, split, "annotations.json")) as fh:
            coco = json.load(fh)
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        sample = rnd.sample(coco["images"], min(k, len(coco["images"])))
        picks += [(split, im, anns.get(im["id"], [])) for im in sample]; n[split] = len(sample)
    title = (f"sld2quote detector tiles, cut from synthetic drawings: {n['train']} random train tiles, then {n['test_unseen']} "
             f"test_unseen tiles ({held}); seed {seed}")
    g, head, cap = 16, 84, 26
    sheet = Image.new("RGB", (cols * cell + (cols + 1) * g, head + -(-len(picks) // cols) * (cap + cell + g)), "white")
    d = ImageDraw.Draw(sheet)
    f_cap, f_tag = font("Arial.ttf", 17), font("Arial Bold.ttf", 13)
    d.text((g, g), title, fill="black", font=font("Arial Bold.ttf", 22))
    x = g
    for i, name in enumerate(CLASSES):  # the legend, one colour per class
        d.rectangle([x, 56, x + 16, 72], fill=PALETTE[i])
        d.text((x + 22, 64), name, fill="black", font=f_cap, anchor="lm")
        x += 40 + d.textlength(name, font=f_cap)
    for k, (split, im, anns) in enumerate(picks):
        cx, cy = g + k % cols * (cell + g), head + k // cols * (cap + cell + g) + cap
        caption = f"{split} | {im['sheet']} | {im['copy']} | tile {im['tile_x']}, {im['tile_y']} | {len(anns)} boxes"
        d.text((cx, cy - 6), caption, fill="black", font=f_cap, anchor="ls")
        with Image.open(os.path.join(out, split, "images", im["file_name"])) as t:
            sheet.paste(t.convert("RGB").resize((cell, cell), Image.LANCZOS), (cx, cy))
        d.rectangle([cx - 1, cy - 1, cx + cell, cy + cell], outline=(170, 170, 170))
        boxes = []
        for a in anns:
            bx, by, bw, bh = (v * cell / im["width"] for v in a["bbox"])
            boxes.append(([cx + bx, cy + by, cx + bx + bw, cy + by + bh], a))
        _draw_labels(d, boxes, f_tag, (cx, cy, cx + cell, cy + cell))
    path = os.path.join(out, "contact_sheet.png")
    sheet.save(path)
    return path


def export(pool: str, out: str, tile: int = 640, overlap: float = 0.25, held_out_font: str = "Courier New.ttf",
           held_out_label_pos: str = "below", train: int = 400, val: int = 40, test: int = 30, unseen: int = 20,
           empty_keep: float = 0.15, seed: int = 0, val_unseen: int = 0) -> dict:
    """the whole export (splits, tiles, COCO files, contact sheet, manifest); returns the manifest"""
    t0 = time.time()
    sheets = read_pool(pool)
    splits, unused = assign(sheets, held_out_font, held_out_label_pos, train, val, test, unseen, val_unseen)
    for name in SPLITS + [VAL_UNSEEN]:
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)  # a rerun never mixes in tiles from an older export
    stats = {name: export_split(name, splits[name], out, tile, overlap, empty_keep, seed) for name in splits}
    contact_sheet(out, seed, f"held-out font {held_out_font}, labels {held_out_label_pos}")
    seeds = [s["seed"] for s in sheets]
    m = {"note": "tiles cut from synthetic drawings (sld2quote/generate.py); no real data", "classes": CLASSES, "tile": tile,
         "overlap": overlap, "min_visible": MIN_VISIBLE, "held_out": {"font": held_out_font, "label_pos": held_out_label_pos},
         "empty_keep": empty_keep, "seed": seed, "requested": {"train": train, "val": val, "test": test, "unseen": unseen} | ({VAL_UNSEEN: val_unseen} if val_unseen else {}),
         "pool": {"path": pool, "sheets": len(sheets), "seeds": [min(seeds), max(seeds)]}, "unused_sheets": unused, "splits": stats,
         "created": datetime.now().astimezone().isoformat(timespec="seconds"), "export_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    return m


def summary(m: dict) -> str:
    """the one-screen report printed after an export"""
    sp, p, u = m["splits"], m["pool"], m["unused_sheets"]
    want, names = {"train": "train", "val": "val", "test_clean": "test", "test_scan": "test", "test_unseen": "unseen", VAL_UNSEEN: VAL_UNSEEN}, list(sp)
    row = lambda label, vals: f"{label:<20}" + "".join(f"{v:>12}" for v in vals)
    L = [f"sld2quote COCO export of synthetic drawings, {m['export_seconds']} s: {p['sheets']} sheets in {p['path']} (seeds {p['seeds'][0]} to {p['seeds'][1]})",
         f"tile {m['tile']} px, overlap {m['overlap']}; held out: font {m['held_out']['font']}, labels {m['held_out']['label_pos']}; "
         f"unused: {u['held_out_style']} held-out-style and {u['regular']} regular sheets", "",
         row("", names), row("sheets requested", [m["requested"][want[s]] for s in names])]
    L += [row(k, [sp[s][k] for s in names]) for k in ("sheets", "images", "boxes", "empty_tiles_kept", "empty_tiles_dropped")]
    L += ["", "boxes per class"] + [row(c, [sp[s]["boxes_per_class"][c] for s in names]) for c in CLASSES]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="cut generated sheets into a COCO tile set for the symbol detector")
    ap.add_argument("--pool", default="data/pool"); ap.add_argument("--out", default="data/coco")
    ap.add_argument("--tile", type=int, default=640); ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--held-out-font", default="Courier New.ttf", choices=FONTS)
    ap.add_argument("--held-out-label-pos", default="below", choices=["right", "below"])
    ap.add_argument("--train", type=int, default=400); ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--test", type=int, default=30); ap.add_argument("--unseen", type=int, default=20)
    ap.add_argument("--val-unseen", type=int, default=0, help="held-out-style sheets after test_unseen's, for selecting a model on an unseen style")
    ap.add_argument("--empty-keep", type=float, default=0.15, help="share of tiles with no symbol that are kept")
    ap.add_argument("--seed", type=int, default=0, help="picks the empty tiles kept and the contact-sheet tiles")
    a = ap.parse_args(argv)
    if a.tile < 1 or not 0 <= a.overlap < 1:
        ap.error("--tile must be positive and --overlap at least 0 and below 1")
    if not 0 <= a.empty_keep <= 1 or min(a.train, a.val, a.test, a.unseen, a.val_unseen) < 0:
        ap.error("--empty-keep must lie between 0 and 1, and no split size may be negative")
    print(summary(export(**vars(a))))
    print(f"wrote {a.out}: <split>/images, <split>/annotations.json, manifest.json, contact_sheet.png")


if __name__ == "__main__":
    main()
