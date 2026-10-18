"""OCR checks: every installed engine keeps the reader's contract on a generated sheet, the engine choice follows the
platform, the matcher and the character accuracy score hand-made cases as they should, and reader._ocr_image still hands
the reader the same shape, with look-alike letters mapped back and boxes in the frame of the image it was given."""
import importlib.util, os, shutil, sys
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sld2quote import ocr, reader
from sld2quote.evaluate_ocr import char_acc, match, score
from sld2quote.generate import make_sheet

MODULE = {"rapidocr": "rapidocr_onnxruntime", "rapidocr-ch": "rapidocr_onnxruntime", "tesseract": "pytesseract", "easyocr": "easyocr", "doctr": "doctr"}


def _missing(engine):
    if engine == "vision":
        return None if ocr.vision_tool() else "needs macOS and the ocr/ocr tool"
    if importlib.util.find_spec(MODULE[engine]) is None:
        return MODULE[engine] + " is not installed"
    return None if engine != "tesseract" or shutil.which("tesseract") else "the tesseract binary is not installed"


@pytest.fixture(scope="module")
def notes():
    """the notes block of a generated sheet: small, and its lines are known"""
    im, sh = make_sheet(7)
    return im.crop((40, 740, 700, 930))


@pytest.mark.parametrize("engine", list(ocr.ENGINES))
def test_every_installed_engine_keeps_the_contract(engine, notes):
    if _missing(engine):
        pytest.skip(_missing(engine))
    res = ocr.ocr_image(notes, 2.0, "page", engine)
    assert res and all(set(b) == {"text", "conf", "box", "src"} and b["src"] == "page" and isinstance(b["text"], str) for b in res)
    assert all(isinstance(b["conf"], float) and 0 <= b["conf"] <= 1 for b in res), [b["conf"] for b in res]
    for b in res:  # in the pixels of the image passed in, not of the upscaled copy the engine read
        x, y, w, h = b["box"]
        assert x >= -2 and y >= -2 and w > 0 and h > 0 and x + w <= notes.width + 2 and y + h <= notes.height + 2, (engine, b)
    assert any("NOTES" in b["text"].upper() for b in res), (engine, [b["text"] for b in res])


def test_vision_is_the_default_only_where_its_tool_can_run(monkeypatch):
    try:
        monkeypatch.setattr(ocr.sys, "platform", "linux")
        ocr.vision_tool.cache_clear(); ocr.default_engine.cache_clear()
        assert ocr.vision_tool() is None and ocr.default_engine() == "rapidocr"
    finally:
        monkeypatch.undo(); ocr.vision_tool.cache_clear(); ocr.default_engine.cache_clear()


def test_reader_ocr_keeps_its_shape_maps_look_alikes_and_page_offsets(monkeypatch):
    monkeypatch.setitem(ocr.ENGINES, "fake", lambda img: [("СТ 100/5А", 0.9, 30, 15, 60, 12)])  # Cyrillic С, Т and А
    monkeypatch.setenv("SLD2QUOTE_OCR", "fake")
    page = Image.new("L", (600, 300), 255)
    assert reader._ocr_image(page, 3.0, "crop") == [{"text": "CT 100/5A", "conf": 0.9, "box": [10.0, 5.0, 20.0, 4.0], "src": "crop"}]
    # the crop pass adds its window's offset (pad left 60, top 40): page coordinates
    assert reader.ocr_region(page, [100, 100, 20, 20])[0]["box"] == [50.0, 65.0, 20.0, 4.0]


def test_reader_ocr_with_the_default_engine_returns_the_same_shape(notes):
    if _missing(ocr.default_engine()):
        pytest.skip(_missing(ocr.default_engine()))
    res = reader._ocr_image(notes, 2.0, "page")
    assert res and all(list(b) == ["text", "conf", "box", "src"] and len(b["box"]) == 4 for b in res)


def test_match_pairs_each_box_once_by_overlap_or_either_centre():
    gt = [[0, 0, 100, 20], [0, 40, 100, 20], [300, 0, 10, 10]]
    pred = [[2, 41, 96, 18],    # IoU 0.86 with the second line
            [0, 0, 40, 20],     # IoU 0.40 with the first line
            [10, 2, 30, 16],    # centre inside the first line too, but that line is taken by the better overlap
            [292, -6, 40, 20],  # IoU 0.13, yet the small box's centre lies inside it
            [500, 500, 10, 10]]
    assert sorted(match(gt, pred)) == [(0, 1), (1, 0), (2, 3)]


def test_character_accuracy_on_hand_made_strings():
    assert char_acc("MCCB 250 A 4P", "MCCB 250 A 4P") == 1.0
    assert char_acc(" MCCB  250 A\t4P ", "MCCB 250 A 4P") == 1.0          # whitespace collapsed
    assert char_acc("St Mary's", "St Mary’s") == 1.0                 # typographic apostrophe
    assert char_acc("MCCB 250A 4P", "MCCB 250 A 4P") == pytest.approx(1 - 1 / 13)
    assert char_acc("kitten", "sitting") == pytest.approx(1 - 3 / 7)
    assert char_acc("", "IP54") == 0.0 and char_acc("IP54", "") == 0.0 and char_acc("", "") == 1.0


def test_score_counts_finding_and_reading_and_a_missed_key_string_scores_zero():
    gt = [{"text": "MCCB 250 A 4P", "box": [0, 0, 120, 14]}, {"text": "2. Prospective fault level 65 kA for 1 s at the busbars.", "box": [0, 100, 400, 14]},
          {"text": "kitchen", "box": [0, 30, 50, 12]}]
    pred = [{"text": "MCCB 250 A 4F", "box": [1, 0, 118, 15]}, {"text": "kitchen", "box": [0, 31, 50, 11]}, {"text": "|", "box": [300, 300, 4, 20]}]
    c = score(gt, pred)
    assert (c["gt"], c["pred"], c["matched"], c["exact"]) == (3, 3, 2, 1)
    assert (c["label_n"], c["label_exact"], c["notes_n"], c["notes_exact"], c["notes_char"]) == (1, 0, 1, 0, 0.0)
    assert c["label_char"] == pytest.approx(1 - 1 / 13)
