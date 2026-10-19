"""Trained reader and the shared reading pipeline: merging detections across overlapping tiles, dropping boxes cut by an
inner tile edge and boxes holding a better box, mapping tile boxes back to the page, detector classes to device types with the three breaker roles, deskew levelling a scan, the classical reader
unchanged by the refactor (against an extraction of a clean sheet saved before it), the RESULTS.md section writer, the
evaluation sets, the threshold chosen on whole validation drawings, and one end-to-end run of the trained reader on a generated sheet (the one slow test, about 15 s; skipped without trained weights;
deselect with -k "not end_to_end")."""
import hashlib, json, math, os, sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from sld2quote import reader, reader_v2 as rv2, select_threshold as sel
from sld2quote.classes import CLASSES, CLASS_ID
from sld2quote.evaluate import iou, make_sets, write_section
from sld2quote.generate import make_sheet, write_sheet
from sld2quote.tiles import tile_grid

WEIGHTS = os.path.join(ROOT, "weights", "rtdetr_r18_full")


def tile(*dets):
    """one tile's detections as detect() gives them, from (score, class name, x0, y0, x1, y1) in tile pixels"""
    return {"scores": torch.tensor([d[0] for d in dets]), "labels": torch.tensor([CLASS_ID[d[1]] for d in dets]),
            "boxes": torch.tensor([d[2:] for d in dets], dtype=torch.float32).reshape(-1, 4)}


def test_a_symbol_in_the_overlap_of_two_tiles_becomes_one_box_with_the_higher_score():
    # a breaker at page x 500 to 540 lies in the tile at x 0 and in the one at x 480, seen a pixel apart by each
    r = rv2.merge([(0, 0), (480, 0)], [tile((0.6, "breaker", 500, 100, 540, 140)), tile((0.9, "breaker", 21, 101, 61, 141))], (1600, 1000))
    assert r["scores"].tolist() == [pytest.approx(0.9)] and r["boxes"].tolist() == [[501, 101, 541, 141]]


def test_two_symbols_that_overlap_less_than_half_both_survive():
    # page boxes 100..140 and 125..165 by 100..140 overlap at IoU 600 / 2600 = 0.23, here each from its own tile
    r = rv2.merge([(0, 0), (100, 0)], [tile((0.9, "contactor", 100, 100, 140, 140)), tile((0.5, "motor", 25, 100, 65, 140))], (1600, 1000))
    assert sorted(r["scores"].tolist()) == pytest.approx([0.5, 0.9]) and sorted(r["labels"].tolist()) == sorted([CLASS_ID["contactor"], CLASS_ID["motor"]])


def test_a_box_at_a_tile_edge_maps_back_to_page_pixels_in_reading_order():
    x0, y0 = tile_grid(3138, 1000)[-1]  # the last tile, flush with the page's bottom-right corner
    assert (x0, y0) == (3138 - 640, 1000 - 640)
    # the earth touches that tile's right and bottom borders, which are the page edge, so it is not a cut symbol
    r = rv2.merge([(0, 0), (x0, y0)], [tile((0.7, "meter", 300.2, 50.1, 330.0, 80.4)), tile((0.8, "earth", 610.2, 600.3, 638.1, 639.2))], (3138, 1000))
    assert rv2.to_symbols(r) == [{"id": "S1", "kind": "meter", "box": [300, 50, 30, 30], "score": 0.7, "shape_score": 0.7},
                                 {"id": "S2", "kind": "earth", "box": [x0 + 610, y0 + 600, 28, 39], "score": 0.8, "shape_score": 0.8}]


def test_a_box_cut_by_an_inner_tile_edge_is_dropped_and_one_at_the_page_edge_is_kept():
    size = (1600, 1000)
    assert tile_grid(*size) == [(x, y) for y in (0, 360) for x in (0, 480, 960)]
    origins = [(480, 0), (960, 0), (0, 360)]
    res = [tile((0.9, "breaker", 0, 100, 30, 140),      # on the tile's left border, inner: a sliver of a symbol whole in the tile at x 0
                (0.8, "breaker", 610, 300, 640, 340),   # on its right border, inner as well
                (0.7, "breaker", 300, 300, 330, 340)),  # clear of every border
           tile((0.9, "motor", 610, 100, 640, 140)),    # on the right border of the last column, which is the page edge
           tile((0.9, "earth", 300, 0, 330, 25),        # on the top border of the bottom row, inner
                (0.8, "earth", 300, 610, 330, 640))]    # on its bottom border, the page edge
    assert sorted(rv2.merge(origins, res, size)["boxes"].tolist()) == [[300, 970, 330, 1000], [780, 300, 810, 340], [1570, 100, 1600, 140]]
    assert len(rv2.merge(origins, res, size, edge_px=None)["scores"]) == 6  # the filter off: all six stay


def test_a_box_holding_a_better_box_is_dropped_and_partial_overlaps_are_kept():
    # a contactor above a motor on one feeder, a weaker tall box spanning both (IoU 0.38 and 0.39 with them, so NMS keeps it),
    # and a meter and a relay that share a quarter of their area
    dets = tile((0.9, "contactor", 100, 100, 130, 140), (0.8, "motor", 98, 152, 132, 190), (0.2, "contactor", 99, 98, 133, 192),
                (0.7, "meter", 300, 100, 340, 140), (0.6, "relay", 330, 100, 370, 140))
    assert sorted(rv2.merge([(0, 0)], [dets], (1600, 1000))["scores"].tolist()) == pytest.approx([0.6, 0.7, 0.8, 0.9])
    assert len(rv2.merge([(0, 0)], [dets], (1600, 1000), contain=None)["scores"]) == 5  # the filter off: the spanning box stays


def text(s, x, y, w=120):
    return {"text": s, "conf": 1.0, "box": [x, y, w, 20], "src": "page"}


def test_detector_classes_become_device_types_and_a_breaker_takes_its_role(tmp_path, monkeypatch):
    monkeypatch.setattr(reader, "_ocr_image", lambda img, scale, tag: [])  # no OCR: the texts given are the only ones
    page = tmp_path / "layout.png"; im = Image.new("L", (1600, 1000), 255)
    ImageDraw.Draw(im).line([140, 400, 1460, 400], fill=0, width=6); im.save(page)  # one busbar at y = 400
    box = lambda kind, x, y: {"kind": kind, "box": [x, y, 30, 30], "shape_score": 0.9}
    others = [c for c in CLASSES if c != "breaker"]
    syms = [box("breaker", 200, 200), box("breaker", 785, 385), box("breaker", 300, 520), box("breaker", 600, 520)] + \
           [box(c, 40 + 120 * k, 760) for k, c in enumerate(others)]
    for i, s in enumerate(syms):
        s["id"] = f"S{i + 1}"
    texts = [text("MCCB 250 A 3P", 340, 525), text("ACB-2 1600 A 4P", 640, 525),  # right of the two breakers below the busbar
             text("FSW 400 A 4P", 80, 765, 70), text("PFC 200 kVAr", 40 + 120 * others.index("capacitor") + 40, 765, 70)]
    steps = []
    ex = reader.read_sheet(str(page), texts, finder=lambda page_img, ink, t: syms, progress=lambda step, stats: steps.append((step, stats)))
    assert [s for s, _ in steps] == ["text", "symbols", "labels"] and steps[0][1] == {"texts": len(texts)} and steps[1][1] == {"symbols": len(syms)}
    assert steps[2][1]["devices"] == len(syms) and 0 < steps[2][1]["linked"] <= len(syms)
    dev = [(d["type"], d.get("rating"), d.get("poles")) for d in ex["devices"]]
    assert dev[:4] == [("ACB", None, None),          # above the busbar: an incomer
                       ("BUS_SECTION", None, None),  # in the busbar
                       ("MCCB", 250, 3),             # below it: an outgoing way
                       ("ACB", 1600, 4)]             # below it, but its label says ACB
    assert {c: t for c, (t, _, _) in zip(others, dev[4:])} == {
        "fuse_switch": "FUSE_SWITCH", "isolator": "ISOLATOR", "contactor": "CONTACTOR", "transformer": "TRANSFORMER", "generator": "GENERATOR",
        "motor": "MOTOR", "board": "BOARD", "ct": "CT", "meter": "METER", "relay": "RELAY", "spd": "SPD", "capacitor": "CAPACITOR", "earth": "EARTH"}
    assert dev[4] == ("FUSE_SWITCH", 400, 4) and dev[4 + others.index("capacitor")] == ("CAPACITOR", 200, None)  # amps and kVAr are read


def bus_tilt(ink, x0: int, x1: int, y: int = 400, half: int = 80, win: int = 40) -> float:
    """the busbar's tilt in degrees (positive when its right end is lower) from two windows on it: in each, the centre of the
    rows that are nearly all ink, which only the busbar is (feeder lines and text cross a window but do not fill it)"""
    def row(x):
        return y - half + np.nonzero(ink[y - half: y + half, x: x + win].sum(axis=1) >= 0.8 * win)[0].mean()
    return math.degrees(math.atan2(row(x1) - row(x0), x1 - x0))


def test_deskew_levels_a_scan_instead_of_doubling_its_tilt(tmp_path):
    im, sh = make_sheet(1, scan=True); im.save(tmp_path / "scan.png"); W = im.width  # its busbar runs at y 400 from x 140 to W - 140
    raw = reader.binarise(reader.load_gray(str(tmp_path / "scan.png")))
    assert abs(bus_tilt(raw, 180, W - 220) + sh.style["rotate_deg"]) < 0.1  # the measure sees the 0.752 degrees the scan was turned by
    g, b, ang = reader.prepare(str(tmp_path / "scan.png"))
    assert abs(bus_tilt(b, 180, W - 220)) < 0.15, (sh.style["rotate_deg"], ang)  # 1.47 while prepare turned the page the wrong way


# the classical reader on make_sheet(1, level="medium") (box breakers, every breaker role), saved before the refactor:
# its devices, and the sha256 of the whole extraction as compact JSON in key order, without the keys added since. Label
# texts updated on 19 Oct for the way-based link (intended): each way takes its whole label block and each circle the
# letter inside it, every one the text the answer key drew for that device; types, boxes and ratings are as before
BEFORE_DEVICES = [
    ("ACB", [207, 200, 28, 28], 3200, ["T3", "T4"]), ("ACB", [1437, 200, 28, 28], 3200, ["T10", "T11"]),
    ("GENERATOR", [255, 281, 23, 24], None, ["T5"]), ("GENERATOR", [300, 283, 20, 20], None, ["T6"]),
    ("BUS_SECTION", [1344, 387, 28, 28], 3200, ["T12"]), ("MCCB", [768, 520, 22, 22], 630, ["T26", "T27", "T28"]),
    ("MCCB", [1154, 550, 22, 22], 800, ["T32", "T33", "T34"]), ("MCCB", [1733, 550, 22, 22], 800, ["T38", "T39", "T40"]),
    ("MCCB", [2119, 550, 22, 22], 630, ["T44", "T45", "T46"]), ("MCCB", [382, 580, 22, 22], 100, ["T20", "T21", "T22"]),
    ("MCCB", [575, 580, 22, 22], 250, ["T23", "T24", "T25"]), ("MCCB", [961, 580, 22, 22], 630, ["T29", "T30", "T31"]),
    ("MCCB", [1540, 580, 22, 22], 400, ["T35", "T36", "T37"]), ("MCCB", [2312, 580, 22, 22], 250, ["T47", "T48", "T49"])]
BEFORE_SHA256 = "1358f4f0699d19b5be9a5cc8c9aa91b01a8de73dd438b7652464ee610db98809"


def test_the_classical_reader_reads_a_generated_sheet_exactly_as_before_the_refactor(tmp_path, monkeypatch):
    """page texts from the answer key and no window OCR, so the run is exact and needs no OCR engine"""
    monkeypatch.setattr(reader, "_ocr_image", lambda img, scale, tag: [])
    im, sh = make_sheet(1, level="medium"); im.save(tmp_path / "sheet-medium-1.png")
    ex = reader.read_sheet(str(tmp_path / "sheet-medium-1.png"), [{"text": t["text"], "conf": 1.0, "box": list(t["box"]), "src": "page"} for t in sh.texts])
    assert [(d["type"], d["box"], d.get("rating"), d["label_texts"]) for d in ex["devices"]] == BEFORE_DEVICES
    old = {k: v for k, v in ex.items() if k not in ("reader", "seconds", "crops", "ocr_engine")}  # crops, ocr_engine: added 19 Oct, settings not readings
    assert hashlib.sha256(json.dumps(old, separators=(",", ":")).encode()).hexdigest() == BEFORE_SHA256
    assert ex["reader"] == "classical" and set(ex["seconds"]) == {"total", "ocr", "detector"}


def test_a_rerun_replaces_only_its_own_section_of_results_md(tmp_path):
    path = str(tmp_path / "RESULTS.md")
    write_section(path, "## data/a, classical reader", ["first", "", "### Per type", "x"])
    write_section(path, "## data/b, trained reader", ["second"])
    write_section(path, "## data/a, classical reader", ["rerun"])
    text = open(path).read()
    assert text.startswith("# sld2quote evaluation") and "first" not in text and "### Per type" not in text
    assert text.index("## data/a, classical reader\n\nrerun") < text.index("## data/b, trained reader\n\nsecond")


def test_the_drawing_sweep_keeps_a_weak_ghost_out_and_counts_it_on_blank_paper():
    # one tile: a breaker, a motor and a capacitor where the answer key has them, and a weak ghost capacitor on blank paper
    dets = tile((0.9, "breaker", 100, 100, 130, 130), (0.3, "motor", 300, 100, 340, 140), (0.5, "capacitor", 500, 100, 530, 140),
                (0.08, "capacitor", 100, 400, 200, 500))
    gts = [("breaker", [100, 100, 30, 30]), ("motor", [300, 100, 40, 40]), ("capacitor", [500, 100, 30, 40])]
    ink = np.zeros((1000, 1600), bool)
    for _, (x, y, w, h) in gts:
        ink[y: y + h, x: x + w] = True
    table = sel.sweep([([(0, 0)], [dets], (1600, 1000), gts, ink)], [0.05, 0.1, 0.2, 0.35])
    # mean class F1: the ghost halves the capacitor's precision at 0.05; at 0.35 the motor is lost
    assert [table[t]["mean_f1"] for t in (0.05, 0.1, 0.2, 0.35)] == [pytest.approx(v, abs=1e-4) for v in (8 / 9, 1, 1, 2 / 3)]
    assert [table[t]["blank_fp"] for t in (0.05, 0.1)] == [1, 0]
    assert sel.choose(table) == 0.2  # 0.1 and 0.2 tie: the middle of the tie, as the trainer chooses per tile
    assert json.loads(json.dumps(table))  # plain numbers only, so the sweep can be written beside the weights


def test_the_drawing_threshold_goes_beside_the_per_tile_one_and_the_reader_prefers_it(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"score_threshold": 0.1, "nms_iou": 0.5}))
    sel.write(str(tmp_path), 0.2, {0.2: {"mean_f1": 1.0, "blank_fp": 0, "classes": {}}}, {"sets": ["data/eval/val"]})
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert (cfg["score_threshold"], cfg["score_threshold_drawing"]) == (0.1, 0.2) and json.load(open(tmp_path / "threshold_drawing.json"))["chosen"] == 0.2
    assert rv2.threshold(SimpleNamespace(**cfg)) == (0.2, "drawing") and rv2.threshold(SimpleNamespace(**cfg), 0.3) == (0.3, "argument")
    assert rv2.threshold(SimpleNamespace(score_threshold=0.1)) == (0.1, "tile")
    with pytest.raises(SystemExit):  # never on a test set
        sel.main(["--weights", str(tmp_path), "--sets", "data/eval/test"])


def test_the_score_calibration_rises_with_the_score_and_every_bin_holds_enough_boxes():
    # pooled boxes and right boxes at each sweep threshold: the bins [0.10, 0.11) and [0.11, 0.12) hold 50 and 100 boxes, so
    # they merge; [0.13, 0.14) is right 100 times in 100 but the top bin, from 0.14 up, only 190 in 200, so the two pool
    sweep = {t: {"classes": {"breaker": {"pred": n, "tp": r}}} for t, n, r in ((0.10, 600, 500), (0.11, 550, 480), (0.12, 450, 400), (0.13, 300, 290), (0.14, 200, 190))}
    bins = sel.calibration(sweep, 0.10)
    assert bins == [[0.10, 150, 100], [0.12, 150, 110], [0.13, 300, 290]]
    assert all(n >= sel.MIN_BOXES for _, n, _ in bins) and all(a[2] / a[1] <= b[2] / b[1] for a, b in zip(bins, bins[1:]))
    assert sel.calibration(sweep, 0.12)[0][0] == 0.12  # bins start at the operating threshold
    table = [[e, round(r / n, 4)] for e, n, r in bins]
    assert [rv2.calibrated(x, table) for x in (0.05, 0.10, 0.1199, 0.12, 0.13, 0.95)] == [0.6667, 0.6667, 0.6667, 0.7333, 0.9667, 0.9667]
    r = rv2.merge([(0, 0)], [tile((0.125, "motor", 100, 100, 140, 140))], (1600, 1000))
    assert [(y["score"], y["shape_score"]) for y in rv2.to_symbols(r, table)] == [(0.12, 0.73)] and rv2.to_symbols(r)[0]["shape_score"] == 0.12
    # the device keeps the raw score and makes its conf from the calibrated one as before: the label's lower conf, else half
    sym = {"id": "S1", "kind": "motor", "box": [100, 100, 40, 40], "score": 0.12, "shape_score": 0.98}
    assert (lambda d: (d["score"], d["conf"]))(reader.make_device(sym, [], [], [], 400)) == (0.12, 0.49)
    assert reader.make_device(sym, [text("M", 150, 110, 20) | {"id": "T1", "conf": 0.9}], [], [], 400)["conf"] == 0.9
    assert "score" not in reader.make_device({k: v for k, v in sym.items() if k != "score"}, [], [], [], 400)  # the classical reader's symbols


def test_make_sets_takes_each_split_from_the_export_s_own_pool_into_its_own_folder(tmp_path):
    pool, coco = tmp_path / "pool_fonts", tmp_path / "coco_fonts"; pool.mkdir(); coco.mkdir()
    ids = {"val": ["sheet-20001"], "val_unseen": ["sheet-20002"], "test_clean": ["sheet-20003"], "test_scan": ["sheet-20003"], "test_unseen": ["sheet-20004"], "train": ["sheet-20005"]}
    names = lambda i: [f"{i}{c}{e}" for c in ("", "-scan") for e in (".png", ".json", "-spec.txt")]
    for i in {i for v in ids.values() for i in v}:
        for f in names(i):
            (pool / f).write_text("x")
    (coco / "manifest.json").write_text(json.dumps({"pool": {"path": str(pool)}, "splits": {k: {"sheet_ids": v} for k, v in ids.items()}}))
    assert make_sets(str(coco)) == {str(tmp_path / "eval_fonts" / n): 1 for n in ("test", "unseen", "val", "val_unseen")}  # data/coco_fonts: data/eval_fonts
    assert sorted(os.listdir(tmp_path / "eval_fonts" / "val_unseen")) == sorted(names("sheet-20002"))
    assert os.path.samefile(tmp_path / "eval_fonts" / "val" / "sheet-20001.png", pool / "sheet-20001.png")  # hard links, not copies


@pytest.mark.skipif(not os.path.isfile(os.path.join(WEIGHTS, "config.json")), reason=f"no trained detector at {WEIGHTS} (weights are not committed; "
                    "train them with sld2quote.train_detector)")
def test_end_to_end_the_trained_reader_finds_the_transformer_and_the_incomer_breaker(tmp_path):
    """detector on cpu, the real OCR engine, a sheet drawn here that the detector never saw"""
    stem = write_sheet(str(tmp_path), 7, level="simple"); gt = json.load(open(stem + ".json"))
    ex = rv2.read_sheet(stem + ".png", weights=WEIGHTS, device="cpu")
    assert {"image", "size", "deskew_deg", "texts", "busbars", "symbols", "devices", "notes"} <= set(ex) and ex["reader"] == "trained"
    assert all(s["kind"] in CLASSES and 0 <= s["shape_score"] <= 1 and len(s["box"]) == 4 for s in ex["symbols"]) and ex["symbols"]
    assert all({"id", "symbol", "type", "box", "rating", "label", "label_texts", "conf", "on_bus", "connected"} <= set(d) for d in ex["devices"])
    assert set(ex["seconds"]) == {"total", "ocr", "detector"} and ex["detector"]["score_threshold"] > 0 and ex["detector"]["score_threshold_source"] in ("drawing", "tile")
    assert (ex["detector"]["edge_px"], ex["detector"]["contain"], ex["detector"]["merge_iou"]) == (rv2.EDGE_PX, rv2.CONTAIN, rv2.MERGE_IOU)
    for want in ("TX-1", "ACB-1"):
        g = next(d for d in gt["devices"] if d["id"] == want)
        assert any(d["type"] == g["type"] and iou(d["box"], g["box"]) >= 0.3 for d in ex["devices"]), (want, ex["devices"])
