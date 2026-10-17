"""Trainer checks: the precision and recall matcher, the threshold chooser, the split report, NMS, the size rule, early stopping, the dataset,
collate and augmentation on a two-tile COCO fixture, and one optimisation step plus the shared inference path of rtdetr_r18vd on
cpu, and reloading trained weights for more passes (skipped unless the weights are already in the Hugging Face cache)."""
import json, math, os, sys, time
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from torchvision import ops
from transformers import RTDetrImageProcessor, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sld2quote import train_detector as td
from sld2quote.classes import CLASSES, CLASS_ID

B, T, M = CLASS_ID["breaker"], CLASS_ID["transformer"], CLASS_ID["motor"]


def gt(img, cls, box):
    return {"image_id": img, "category_id": cls, "bbox": box}


def det(img, cls, box, score):
    return {"image_id": img, "category_id": cls, "bbox": box, "score": score}


def coco_fixture(root):
    """two blank 640 px tiles with two boxes each, laid out as the export writes a split"""
    d = root / "train"; (d / "images").mkdir(parents=True); images, anns = [], []
    for i in (1, 2):
        Image.new("L", (640, 640), 255).save(d / "images" / f"tile-{i}.png")
        images.append({"id": i, "file_name": f"tile-{i}.png", "width": 640, "height": 640, "copy": "clean", "breaker": "iec"})
        for k, (cls, box) in enumerate([(B, [100, 200, 40, 20]), (T, [400, 300, 60, 80])]):
            anns.append({"id": 10 * i + k, "image_id": i, "category_id": cls, "bbox": box, "area": box[2] * box[3], "iscrowd": 0, "visible_frac": 1.0})
    json.dump({"images": images, "annotations": anns, "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}, open(d / "annotations.json", "w"))
    return str(root)


def test_matcher_counts_hits_duplicates_near_misses_wrong_classes_and_misses():
    anns = [gt(1, B, [0, 0, 10, 10]), gt(1, B, [50, 0, 10, 10]), gt(1, T, [100, 0, 20, 20]), gt(2, B, [0, 0, 10, 10]),
            gt(3, B, [0, 0, 10, 10]), gt(3, B, [6, 0, 10, 10])]
    dets = [det(1, B, [0, 0, 10, 10], 0.9),     # finds the first breaker
            det(1, B, [1, 0, 10, 10], 0.8),     # a duplicate of it (IoU 0.82 with a box already taken): false positive
            det(1, B, [55, 0, 10, 10], 0.7),    # IoU 0.33 with the second breaker, below 0.5: false positive, and that breaker is missed
            det(1, M, [100, 0, 20, 20], 0.95),  # right place, wrong class: a motor false positive, and the transformer is missed
            det(2, B, [0, 0, 10, 10], 0.3),     # finds tile 2's breaker, but only at thresholds up to 0.3
            det(3, B, [4, 0, 10, 10], 0.6),     # overlaps both tile-3 breakers (IoU 0.43 and 0.67) and takes the one it overlaps most
            det(3, B, [0, 0, 10, 10], 0.5)]     # so the other one is still free for this detection
    got, hit = td.match(anns, dets)
    assert got.tolist() == [0.9, -1, -1, 0.3, 0.5, 0.6] and hit.tolist() == [True, False, False, False, True, True, True]
    assert td.match(anns, dets, iou_thr=0.3)[0].tolist()[1] == 0.7  # at IoU 0.3 the near miss counts
    p, r, f1, tp, npred, ngt = td.pr(td.scored(anns, dets), [0.5, 0.2])
    assert (tp[B].tolist(), npred[B].tolist(), int(ngt[B, 0])) == ([3, 4], [5, 6], 5)
    assert np.allclose(p[B], [3 / 5, 4 / 6]) and np.allclose(r[B], [3 / 5, 4 / 5]) and np.allclose(f1[B], [3 / 5, 8 / 11])
    assert (tp[T].tolist(), npred[T].tolist(), r[T].tolist()) == ([0, 0], [0, 0], [0.0, 0.0])  # the transformer is never found
    assert (npred[M].tolist(), p[M].tolist(), int(ngt[M, 0])) == ([1, 1], [0.0, 0.0], 0)        # the motor is a pure false positive


def test_threshold_chooser_maximises_mean_f1_and_takes_the_middle_of_a_tie():
    anns = [gt(1, B, [0, 0, 10, 10]), gt(2, B, [0, 0, 10, 10])]
    dets = [det(1, B, [0, 0, 10, 10], 0.9), det(2, B, [0, 0, 10, 10], 0.8),  # both breakers found
            det(3, B, [0, 0, 10, 10], 0.5), det(4, B, [0, 0, 10, 10], 0.3)]  # and two false positives on tiles without breakers
    # breaker F1 is 1 for thresholds from 0.51 to 0.80 and lower everywhere else: the middle of that tie is 0.66
    assert td.choose_threshold(td.scored(anns, dets)) == 0.66
    # a transformer found only at 0.4 pulls the best mean F1 down to 0.31..0.40 (breaker 0.8, transformer 1): the middle is 0.36
    anns.append(gt(5, T, [0, 0, 20, 20])); dets.append(det(5, T, [0, 0, 20, 20], 0.4))
    assert td.choose_threshold(td.scored(anns, dets)) == 0.36


def test_split_report_counts_false_positives_of_absent_classes_and_leaves_empty_ratios_undefined(tmp_path):
    ds = td.Tiles(coco_fixture(tmp_path), "train", RTDetrImageProcessor(do_resize=False))
    dets = [det(1, B, [100, 200, 40, 20], 0.9), det(1, T, [400, 300, 60, 80], 0.8), det(2, B, [100, 200, 40, 20], 0.7),  # tile 2's transformer is missed
            det(2, M, [0, 50, 30, 30], 0.6)]  # a motor at the tile's left edge, where none is drawn
    m = td.split_metrics(ds, dets, 1.0, td.scored(ds.annotations(), dets), 0.5); pc = m["per_class"]
    assert [pc[c][k] for c in ("breaker", "transformer") for k in ("precision", "recall")] == [1.0, 1.0, 1.0, 0.5]
    assert [pc[c][k] for c in ("motor", "earth") for k in ("precision", "recall")] == [0.0, None, None, None]
    assert (m["min_precision"], m["mean_precision"], m["min_recall"], m["mean_recall"]) == (0.0, 0.6667, 0.5, 0.75)
    assert m["false_positives"] == {"count": 1, "at_tile_edge": 1}
    assert m["recall_by"]["copy"] == {"clean": 0.75} and m["breaker_recall_by_style"] == {"iec": 1.0}
    assert m["map50"] == pytest.approx((1 + 51 / 101) / 2, abs=1e-4)  # pycocotools: both breakers found, half the transformers


def test_nms_removes_a_duplicate_and_keeps_two_symbols_that_overlap_less_than_half():
    I = CLASS_ID["isolator"]
    r = {"scores": torch.tensor([0.9, 0.8, 0.7, 0.6]), "labels": torch.tensor([B, B, I, T]),
         "boxes": torch.tensor([[0, 0, 10, 10], [1, 0, 11, 10], [8, 0, 18, 10], [0, 0, 10, 10.0]])}
    k = td.nms(r, 0.5)
    # the second breaker box duplicates the first (IoU 0.82) and the transformer label sits on the same symbol (IoU 1), so both
    # go whatever their class; the isolator overlaps the breaker with IoU 0.11, two symbols drawn close together, so it stays
    assert k["labels"].tolist() == [B, I] and k["scores"].tolist() == pytest.approx([0.9, 0.7]) and k["boxes"].tolist()[1] == [8, 0, 18, 10]
    assert td.nms({key: v[:0] for key, v in r.items()}, 0.5)["scores"].numel() == 0  # a tile with nothing detected


def test_size_rule_passes_a_class_only_at_target_on_both_val_and_test_unseen():
    def split(**per):
        pc = {c: {"gt": 10, "precision": 1.0, "recall": 1.0} for c in CLASSES}
        for c, (gt, p, r) in per.items():
            pc[c] = {"gt": gt, "precision": p, "recall": r}
        return {"per_class": pc}
    val = split(earth=(10, 0.99, 0.96), relay=(0, None, None), spd=(10, None, 0.0))
    unseen = split(earth=(10, 0.97, 0.94), meter=(10, 0.95, 0.95))
    rule = td.size_rule(val, unseen, 0.95)
    assert (rule["breaker"], rule["meter"], rule["earth"], rule["relay"], rule["spd"]) == ("pass", "pass", "fail", "n/a", "fail")


def run_curve(scores, patience, cap=None):
    """the trainer's epoch loop on a hand-made val curve: passes run, the stop reason and the best epoch"""
    a, hist = SimpleNamespace(patience=patience, min_delta=0.002, epochs=cap or len(scores), nms_iou=0.5), {"epochs": []}
    for epoch in range(a.epochs):
        if stop := td.patience_stop(a, hist, epoch):
            break
        hist["epochs"].append({"epoch": epoch + 1, "val": {"nms": {"map50": scores[epoch]}}})
    return len(hist["epochs"]), stop, td.early_stop([e["val"]["nms"]["map50"] for e in hist["epochs"]], a.min_delta)[0] + 1


def test_early_stopping_keeps_the_best_pass_and_stops_after_patience_passes_without_a_real_gain():
    # it improves to 0.80 at epoch 3, then 0.801 (a rise below min_delta), 0.79 and 0.80: three passes without a real gain, so it
    # stops after epoch 6 with epoch 3 as the best pass, although 0.801 is higher by a hair
    n, stop, best = run_curve([0.5, 0.7, 0.8, 0.801, 0.79, 0.80, 0.95, 0.96], 3)
    assert (n, best) == (6, 3) and stop.startswith("early stop after epoch 6 of 8: 3 val passes") and "best pass was epoch 3 at 0.8" in stop
    # a late improvement resets the count: two flat passes, a gain at epoch 6, two more flat passes, and the cap of 8 ends it
    assert run_curve([0.5, 0.7, 0.8, 0.8, 0.8, 0.85, 0.85, 0.849], 3) == (8, None, 6)
    # on a curve that keeps improving patience never triggers and the cap ends it; patience 0 never stops, even on a falling curve
    assert run_curve([0.1, 0.2, 0.3, 0.4], 3) == (4, None, 4) and run_curve([0.5, 0.4, 0.3, 0.2, 0.1], 0) == (5, None, 1)


def test_dataset_and_collate_give_the_model_its_inputs(tmp_path):
    proc = RTDetrImageProcessor(do_resize=False)  # the hub processor's settings, without a download
    ds = td.Tiles(coco_fixture(tmp_path), "train", proc)
    b = td.collate([ds[0], ds[1]])
    assert len(ds) == 2 and b["pixel_values"].shape == (2, 3, 640, 640) and float(b["pixel_values"].min()) == 1.0  # white, rescaled to 0..1
    lab = b["labels"][0]
    assert lab["class_labels"].tolist() == [B, T] and int(lab["image_id"]) == 1
    centres = [[120 / 640, 210 / 640], [430 / 640, 340 / 640]]
    assert np.allclose(lab["boxes"].numpy(), [c + s for c, s in zip(centres, [[40 / 640, 20 / 640], [60 / 640, 80 / 640]])])  # cx, cy, w, h
    aug = td.Tiles(str(tmp_path), "train", proc, aug=(0.2, 0.2, 0.1, 0.6))
    assert torch.equal(aug[0][0], aug[0][0])  # seeded by tile and epoch
    for epoch in range(6):
        aug.epoch = epoch; boxes = aug[0][1]["boxes"].numpy()
        # a zoom of 10% about the centre moves a centre by 32 px at most; a flip would send the breaker to the other side
        assert boxes.shape == (2, 4) and np.all(np.abs(boxes[:, :2] - centres) <= (32 + 2) / 640)


def test_jitter_moves_each_box_with_its_symbol_and_drops_what_the_zoom_cuts_off():
    img = Image.new("RGB", (640, 640), "white")
    ImageDraw.Draw(img).rectangle([100, 200, 139, 219], fill="black")  # a whole symbol at [100, 200, 40, 20], in the left half
    anns = [{"bbox": [100, 200, 40, 20], "area": 800, "category_id": B},
            {"bbox": [610, 300, 30, 40], "area": 1200, "category_id": T, "visible_frac": 0.75}]  # one the tile edge already cuts
    for end, kept_classes in ((max, [B]), (min, [B, T])):  # the strongest zoom in, then out, with brightness and contrast at that end
        out, kept = td.jitter(img, anns, SimpleNamespace(uniform=lambda lo, hi: end(lo, hi)), 0.2, 0.2, 0.1, 0.6)
        ys, xs = np.nonzero(np.asarray(out.convert("L"))[:, :320] < 128)  # the symbol's ink, still in the left half: no flip
        x, y, w, h = kept[0]["bbox"]
        assert np.allclose([x, y, x + w, y + h], [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], atol=1.5)
        assert [k["category_id"] for k in kept] == kept_classes  # zooming in leaves 1 px of the cut symbol, below 0.6 of it


def test_trained_weights_load_for_more_passes_without_a_new_head(tmp_path):
    try:
        model = td.load_model("PekingU/rtdetr_r18vd", local_files_only=True)
    except OSError as e:
        pytest.skip(f"PekingU/rtdetr_r18vd is not in the Hugging Face cache, and this test does not download it ({type(e).__name__})")
    model.save_pretrained(tmp_path); RTDetrImageProcessor(do_resize=False).save_pretrained(tmp_path)  # what a run writes
    again, info = td.load_model(str(tmp_path), output_loading_info=True)
    assert not (info["missing_keys"] or info["mismatched_keys"] or info["unexpected_keys"])  # nothing newly initialised
    assert torch.equal(again.class_embed[-1].weight, model.class_embed[-1].weight) and torch.equal(again.model.enc_score_head.weight, model.model.enc_score_head.weight)
    assert td.load_processor(str(tmp_path)).do_resize is False


def test_one_optimisation_step_on_cpu(tmp_path):
    t0 = time.time()
    try:
        model = td.load_model("PekingU/rtdetr_r18vd", local_files_only=True)
    except OSError as e:
        pytest.skip(f"PekingU/rtdetr_r18vd is not in the Hugging Face cache, and this test does not download it ({type(e).__name__}); `bench` does")
    ds = td.Tiles(coco_fixture(tmp_path), "train", RTDetrImageProcessor(do_resize=False))
    opt = td.optimiser(model, lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4)
    assert [g["lr"] for g in opt.param_groups] == [1e-5, 1e-4]
    assert len(opt.param_groups[0]["params"]) == sum(1 for n, p in model.named_parameters() if n.startswith("model.backbone.") and p.requires_grad) > 0
    head = model.class_embed[-1].weight.detach().clone(); b = td.collate([ds[0], ds[1]])
    model.train(); loss = td.step(model, opt, get_cosine_schedule_with_warmup(opt, 0, 10), b, "cpu", 0.1)
    assert math.isfinite(loss) and loss > 0 and not torch.equal(head, model.class_embed[-1].weight)
    model.eval()
    with torch.no_grad():  # the shared inference path: raw, then with NMS, which never leaves two boxes overlapping by more than 0.5
        raw, kept = (td.detect(model, ds.proc, b["pixel_values"], [(640, 640)] * 2, 0.01, n) for n in (0, 0.5))
    assert all(len(k["scores"]) <= len(r["scores"]) for r, k in zip(raw, kept))
    assert all(float(ops.box_iou(k["boxes"], k["boxes"]).fill_diagonal_(0).max()) <= 0.5 for k in kept if len(k["boxes"]) > 1)
    assert time.time() - t0 < 60
