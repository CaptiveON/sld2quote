"""Generator checks that protect the training labels: drawings stay legible and boxes match the drawn symbols."""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sld2quote"))
from generate import make_sheet, image_box


def _overlap(a, b):
    return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))


def _inside(t, b):
    return t[0] >= b[0] and t[1] >= b[1] and t[0] + t[2] <= b[0] + b[2] and t[1] + t[3] <= b[1] + b[3]


def test_no_label_collides_with_a_symbol_another_label_or_the_page_edge():
    for level in ("simple", "medium"):
        for seed in range(1, 41):
            im, sh = make_sheet(seed, level=level)
            W, H = im.size
            texts, boxes = [t["box"] for t in sh.texts], [d.box for d in sh.devices]
            assert all(b[0] >= 0 and b[0] + b[2] <= W and b[1] + b[3] <= H for b in texts), (level, seed)
            # text may sit inside a symbol (the G, M, meter units, I>, board names), never across one
            assert not any(_overlap(t, b) and not _inside(t, b) for t in texts for b in boxes), (level, seed)
            assert not any(_overlap(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]), (level, seed)
            assert not any(_overlap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]), (level, seed)


def test_every_box_hugs_the_symbol_it_labels():
    for level in ("simple", "medium"):
        for seed in range(1, 9):
            im, sh = make_sheet(seed, level=level)
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
