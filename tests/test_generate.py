"""Generator checks that protect the training labels: drawings stay legible in every font set, boxes match the drawn symbols,
and the default font set still draws the sheets the committed numbers rest on, byte for byte."""
import hashlib, json, os, sys
import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sld2quote"))
from generate import FONT_SETS, font, font_path, glyph_face, make_sheet, image_box, write_sheet

# seed 1 as written by the generator before the font sets existed (18 Oct); a Pillow or zlib upgrade can change the PNG bytes
# while the pixels stay the same, so on a failure compare the pixels before deciding the drawing changed
SEED_1 = {"sheet-0001.png": "02afc7e50652ebeb2334b322c126d6051b4766301129fabbd7cc9d09b58adfde",
          "sheet-0001.json": "76242578de594de4cbe8da972827cab16be7783f0c5b3b77abec73c0d250354c",
          "sheet-0001-scan.png": "38a01ce98a2201ed7025d5b76796e9a256379dc73f6cb1b47f7e52b4a291f7f2",
          "sheet-0001-scan.json": "d2957d5f57f1408414bb6508212a03f20562b9212c1b3e6cadcfde82d2d491cf"}


def _overlap(a, b):
    return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))


def _inside(t, b):
    return t[0] >= b[0] and t[1] >= b[1] and t[0] + t[2] <= b[0] + b[2] and t[1] + t[3] <= b[1] + b[3]


def test_the_classic_font_set_reproduces_a_known_sheet_byte_for_byte(tmp_path):
    write_sheet(str(tmp_path), 1); write_sheet(str(tmp_path), 1, scan=True)
    got = {}
    for n in SEED_1:
        b = (tmp_path / n).read_bytes()
        if n.endswith(".json"):  # the answer key as written then, once each text's "device" (added 19 Oct) is set aside
            gt = json.loads(b); [t.pop("device") for t in gt["texts"]]; b = json.dumps(gt, indent=1).encode()
        got[n] = hashlib.sha256(b).hexdigest()
    assert got == SEED_1


@pytest.mark.parametrize("fonts", list(FONT_SETS))
def test_no_label_collides_with_a_symbol_another_label_or_the_page_edge(fonts):
    assert all(os.path.basename(font_path(n)) == n for n in FONT_SETS[fonts])  # no font of the set falls back to Arial
    for level in ("simple", "medium"):
        for seed in range(1, 41):
            im, sh = make_sheet(seed, level=level, fonts=fonts)
            W, H = im.size
            assert sh.style.get("font_set") == (None if fonts == "classic" else fonts) and sh.style["font"] in FONT_SETS[fonts]
            texts, boxes = [t["box"] for t in sh.texts], [d.box for d in sh.devices]
            assert all(b[0] >= 0 and b[0] + b[2] <= W and b[1] + b[3] <= H for b in texts), (level, seed)
            # text may sit inside a symbol (the G, M, meter units, I>, board names), never across one
            assert not any(_overlap(t, b) and not _inside(t, b) for t in texts for b in boxes), (level, seed)
            assert not any(_overlap(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]), (level, seed)
            assert not any(_overlap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]), (level, seed)
            if fonts == "classic":  # classic sheets keep two drafting faults, so they reproduce: both are fixed in the wide set
                continue
            ink = np.array(im) < 128
            right = 60 + int(np.argmin(ink[761, 60:]))  # the notes box's right edge as drawn
            assert all(t[0] + t[2] < right for t in texts if t[1] > 760 and t[0] < right), (level, seed)  # no note runs past it
            for d in sh.devices:  # with labels below, no cable runs into a way's first label line
                if sh.style["label_pos"] == "below" and d.type in ("MCCB", "FUSE_SWITCH", "ISOLATOR"):
                    cx = d.box[0] + d.box[2] / 2
                    x, y, w, h = map(round, min((t["box"] for t in sh.texts if t["text"] == d.label), key=lambda b: abs(b[0] + b[2] / 2 - cx)))
                    assert not ink[y - 3: y, x + 1: x + w - 1].any(), (level, seed, d.id)


def test_a_wide_sheet_writes_inside_its_symbols_in_its_fonts_bold_face():
    faces = {n: font(glyph_face(n)[0], 20, glyph_face(n)[1]).getname() for n in FONT_SETS["wide"]}
    assert all(f[0] == font(n, 20).getname()[0] for n, f in faces.items())  # the sheet font's own family
    assert {n for n, f in faces.items() if f[1] != "Bold"} == {"Geneva.ttf", "Monaco.ttf"}  # no bold face on this Mac


@pytest.mark.parametrize("fonts", list(FONT_SETS))
def test_every_box_hugs_the_symbol_it_labels(fonts):
    for level in ("simple", "medium"):
        for seed in range(1, 9):
            im, sh = make_sheet(seed, level=level, fonts=fonts)
            ink = np.array(im) < 128
            for d in sh.devices:
                x, y, w, h = d.box  # connecting lines may run up to the window's edge, hence the 3 px slack
                ys, xs = np.nonzero(ink[y - 2: y + h + 3, x - 2: x + w + 3])
                assert len(xs), (level, seed, d.id)
                got = [x - 2 + xs.min(), y - 2 + ys.min(), x - 2 + xs.max(), y - 2 + ys.max()]
                assert all(abs(a - b) <= 3 for a, b in zip(got, [x, y, x + w, y + h])), (level, seed, d.id, d.type, got, d.box)


def test_image_box_follows_the_scan_rotation():
    size, box, deg = (3000, 1000), [2700, 80, 30, 30], 1.1  # far from the centre, where a wrong mapping shows most
    im = Image.new("L", size, 255)
    ImageDraw.Draw(im).rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], fill=0)
    ys, xs = np.nonzero(np.array(im.rotate(deg, resample=Image.BICUBIC, fillcolor=255)) < 128)
    seen = [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]
    assert all(abs(a - b) <= 2 for a, b in zip(seen, image_box(box, size, deg))), (seen, image_box(box, size, deg))
