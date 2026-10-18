"""COCO export checks: tiles cover every pixel, every label survives tiling where the ground truth puts it, the held-out
styles stay out of training, val_unseen takes the next held-out sheets without moving any other split, and the manifest matches
what is on disk."""
import json, os, shutil, sys
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sld2quote.classes import CLASSES, CLASS_ID, DEVICE_TO_CLASS
from sld2quote.export_coco import SPLITS, main
from sld2quote.generate import image_box, write_sheet
from sld2quote.tiles import tile_grid, tile_starts

# In ascending order, seeds 2 and 3 draw labels below and 11 and 12 draw Courier New with labels right; the other nine are
# regular. With these split sizes a sheet that slipped past either hold-out would land in train or val, and one sheet of
# each kind is left over, so the unused counts are exercised too.
SEEDS = [1, 2, 3, 4, 6, 7, 10, 11, 12, 14, 15, 16, 17]
SIZES = {"train": 5, "val": 1, "test": 2, "unseen": 3}
KINDS = {"train": ("clean", "scan"), "val": ("clean", "scan"), "test_clean": ("clean",), "test_scan": ("scan",), "test_unseen": ("clean", "scan"),
         "val_unseen": ("clean", "scan")}
held = lambda st: st["font"] == "Courier New.ttf" or st["label_pos"] == "below"  # the export's default hold-out


def _gt(pool, sheet, copy):
    with open(os.path.join(pool, sheet + ("-scan" if copy == "scan" else "") + ".json")) as fh:
        return json.load(fh)


def _export(pool, out, *extra):
    main(["--pool", pool, "--out", out] + [x for k, v in SIZES.items() for x in (f"--{k}", str(v))] + list(extra))
    with open(os.path.join(out, "manifest.json")) as fh:
        m = json.load(fh)
    coco = {}
    for s in m["splits"]:
        with open(os.path.join(out, s, "annotations.json")) as fh:
            coco[s] = json.load(fh)
    return pool, out, m, coco


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    pool = str(root / "pool")
    for s in SEEDS:
        write_sheet(pool, s); write_sheet(pool, s, scan=True)
    return _export(pool, str(root / "coco"))


def _counts_match_disk(pool, out, m, coco, tile):
    for s in m["splits"]:
        e, c = m["splits"][s], coco[s]
        files = sorted(os.listdir(os.path.join(out, s, "images")))
        assert e["images"] == len(c["images"]) == len(files) and files == sorted(im["file_name"] for im in c["images"]), s
        assert all(im["file_name"] == f"{im['sheet']}{'-scan' if im['copy'] == 'scan' else ''}_{im['tile_x']}_{im['tile_y']}.png" for im in c["images"]), s
        assert e["boxes"] == len(c["annotations"]) == sum(e["boxes_per_class"].values()), s
        assert e["boxes_per_class"] == {n: sum(a["category_id"] == i for a in c["annotations"]) for i, n in enumerate(CLASSES)}, s
        assert e["sheets"] == len(e["sheet_ids"]) == len({im["sheet"] for im in c["images"]}), s
        assert [im["id"] for im in c["images"]] == list(range(1, len(c["images"]) + 1)), s
        assert [a["id"] for a in c["annotations"]] == list(range(1, len(c["annotations"]) + 1)), s
        assert {a["image_id"] for a in c["annotations"]} <= {im["id"] for im in c["images"]}, s
        assert len(c["images"]) - len({a["image_id"] for a in c["annotations"]}) == e["empty_tiles_kept"], s
        pages = [(i, k) for i in e["sheet_ids"] for k in KINDS[s]]  # every tile of every page is either written or dropped
        assert sum(len(tile_grid(*_gt(pool, i, k)["size"], tile)) for i, k in pages) == e["images"] + e["empty_tiles_dropped"], s
        for f in files:
            with Image.open(os.path.join(out, s, "images", f)) as t:
                assert t.mode == "L" and t.size == (tile, tile), (s, f)
    assert m["pool"]["sheets"] == len(SEEDS) and m["pool"]["seeds"] == [min(SEEDS), max(SEEDS)]
    assert os.path.getsize(os.path.join(out, "contact_sheet.png")) > 0


def test_tiles_cover_every_pixel_and_neighbours_overlap_by_a_quarter():
    for length in (300, 640, 641, 1000, 1600, 2524, 4848):
        starts = tile_starts(length)
        covered = {p for s in starts for p in range(s, s + 640)}
        assert starts == sorted(set(starts)) and starts[0] == 0 and set(range(length)) <= covered, (length, starts)
        assert max(starts) + 640 == max(length, 640), (length, starts)
        assert all(a + 640 - b >= 0.25 * 640 for a, b in zip(starts, starts[1:])), (length, starts)


def test_every_train_box_is_whole_in_some_tile_where_the_ground_truth_puts_it(ds):
    pool, _, m, coco = ds
    tiles, seen = {im["id"]: im for im in coco["train"]["images"]}, {}
    for a in coco["train"]["annotations"]:
        im = tiles[a["image_id"]]; x, y, w, h = a["bbox"]
        seen.setdefault((im["sheet"], im["copy"], a["device_id"]), []).append((a["visible_frac"], [x + im["tile_x"], y + im["tile_y"], w, h]))
    for sheet in m["splits"]["train"]["sheet_ids"]:
        for copy in ("clean", "scan"):
            gt = _gt(pool, sheet, copy)
            for d in gt["devices"]:
                box = d["box"] if copy == "clean" else image_box(d["box"], gt["size"], gt["style"]["rotate_deg"])
                got = seen.get((sheet, copy, d["id"]), [])
                assert any(v >= 0.99 for v, _ in got), (sheet, copy, d["id"], got)
                assert all(b == box for v, b in got if v == 1), (sheet, copy, d["id"], box, got)  # scan boxes are mapped


def test_every_clipped_box_lies_inside_its_tile(ds):
    for split, c in ds[3].items():
        size = {im["id"]: (im["width"], im["height"]) for im in c["images"]}
        for a in c["annotations"]:
            (x, y, w, h), (tw, th) = a["bbox"], size[a["image_id"]]
            assert 0 <= x and 0 <= y and 0 < w and 0 < h and x + w <= tw and y + h <= th, (split, a)
            assert a["area"] == w * h and 0.6 <= a["visible_frac"] <= 1 and a["iscrowd"] == 0, (split, a)


def test_splits_are_whole_sheets_in_seed_order_and_keep_the_held_out_styles_out_of_training(ds):
    pool, _, m, coco = ds
    style = {f"sheet-{s:04d}": _gt(pool, f"sheet-{s:04d}", "clean")["style"] for s in SEEDS}
    regular, unseen = [i for i in style if not held(style[i])], [i for i in style if held(style[i])]
    a, b, c = SIZES["train"], SIZES["train"] + SIZES["val"], SIZES["train"] + SIZES["val"] + SIZES["test"]
    want = {"train": regular[:a], "val": regular[a:b], "test_clean": regular[b:c], "test_scan": regular[b:c], "test_unseen": unseen[:SIZES["unseen"]]}
    assert {s: m["splits"][s]["sheet_ids"] for s in SPLITS} == want
    assert m["splits"]["test_clean"]["sheet_ids"] == m["splits"]["test_scan"]["sheet_ids"]  # the two columns compare like for like
    assert m["unused_sheets"] == {"held_out_style": len(unseen) - SIZES["unseen"], "regular": len(regular) - c} == {"held_out_style": 1, "regular": 1}
    # a sheet of each held-out kind sits below the last train or val seed, so a leak through either would show here
    last = int(want["val"][-1].split("-")[1])
    assert any(st["font"] == "Courier New.ttf" and st["label_pos"] == "right" and int(i.split("-")[1]) < last for i, st in style.items())
    assert any(st["font"] != "Courier New.ttf" and st["label_pos"] == "below" and int(i.split("-")[1]) < last for i, st in style.items())
    assert not any(held(im) for s in ("train", "val") for im in coco[s]["images"]) and all(held(im) for im in coco["test_unseen"]["images"])
    groups = [set(want[s]) for s in ("train", "val", "test_clean", "test_unseen")]
    assert all(not (g & h) for i, g in enumerate(groups) for h in groups[i + 1:])
    for s in SPLITS:
        assert {(im["sheet"], im["copy"]) for im in coco[s]["images"]} == {(i, k) for i in want[s] for k in KINDS[s]}, s


def test_val_unseen_takes_the_next_held_out_sheets_and_leaves_every_other_split_as_it_was(ds, tmp_path):
    pool, _, m0, coco0 = ds
    _, out, m, coco = _export(pool, str(tmp_path / "coco"), "--val-unseen", "1")
    _counts_match_disk(pool, out, m, coco, 640)
    style = {f"sheet-{s:04d}": _gt(pool, f"sheet-{s:04d}", "clean")["style"] for s in SEEDS}
    unseen = [i for i in style if held(style[i])]
    assert m["splits"]["val_unseen"]["sheet_ids"] == unseen[SIZES["unseen"]:SIZES["unseen"] + 1]  # the held-out sheet after test_unseen's
    assert {s: m["splits"][s]["sheet_ids"] for s in SPLITS} == {s: m0["splits"][s]["sheet_ids"] for s in SPLITS}
    assert all(coco[s] == coco0[s] for s in SPLITS)  # every other split exactly as before
    assert not set(m["splits"]["val_unseen"]["sheet_ids"]) & {i for s in SPLITS for i in m["splits"][s]["sheet_ids"]}
    assert all(held(im) for im in coco["val_unseen"]["images"]) and m["unused_sheets"]["held_out_style"] == 0
    assert m["requested"]["val_unseen"] == 1 and "val_unseen" not in m0["requested"] and "val_unseen" not in m0["splits"]  # off by default
    _, _, m2, _ = _export(pool, out)  # a rerun without it removes the old val_unseen
    assert not os.path.exists(os.path.join(out, "val_unseen")) and "val_unseen" not in m2["splits"]


def test_manifest_counts_match_the_files_and_annotations_on_disk(ds):
    _counts_match_disk(*ds, 640)


def test_tiles_without_a_symbol_are_kept_at_the_requested_share_and_a_rerun_leaves_no_old_tile(ds, tmp_path):
    # 640 px tiles on these sheets always hold a symbol; at 320 px the space above the busbar and the notes block do not.
    # The export goes over a copy of the 640 px one, so a tile left from it would fail the disk checks.
    shutil.copytree(ds[1], tmp_path / "coco")
    pool, out, m, coco = _export(ds[0], str(tmp_path / "coco"), "--tile", "320", "--empty-keep", "0.5")
    _counts_match_disk(pool, out, m, coco, 320)
    kept, dropped = (sum(m["splits"][s][k] for s in SPLITS) for k in ("empty_tiles_kept", "empty_tiles_dropped"))
    assert kept > 20 and dropped > 20 and 0.35 < kept / (kept + dropped) < 0.65, (kept, dropped)


def test_a_pool_too_small_for_the_splits_or_a_negative_size_stops_before_the_old_export_is_touched(ds, tmp_path):
    out = tmp_path / "coco"
    shutil.copytree(ds[1], out)
    before = {s: sorted(os.listdir(out / s / "images")) for s in SPLITS}, (out / "manifest.json").read_bytes()
    with pytest.raises(SystemExit, match="9 regular and 4 held-out-style sheets; 10 and 3 were asked for"):
        main(["--pool", ds[0], "--out", str(out), "--train", "7", "--val", "1", "--test", "2", "--unseen", "3"])
    with pytest.raises(SystemExit) as stop:  # these sizes fit the pool, so only the argument check can stop it
        main(["--pool", ds[0], "--out", str(out), "--train", "-1", "--val", "1", "--test", "2", "--unseen", "3"])
    assert stop.value.code == 2
    assert ({s: sorted(os.listdir(out / s / "images")) for s in SPLITS}, (out / "manifest.json").read_bytes()) == before


def test_category_ids_and_names_follow_classes_py(ds):
    assert ds[2]["classes"] == CLASSES
    for s, c in ds[3].items():
        assert c["categories"] == [{"id": i, "name": n} for i, n in enumerate(CLASSES)], s
        assert all(a["category_id"] == CLASS_ID[DEVICE_TO_CLASS[a["device_type"]]] for a in c["annotations"]), s
