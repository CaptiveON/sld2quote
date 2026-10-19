"""The shared reader's label link and role rules on small hand-made layouts (no image, no OCR, no detector): label lines
read through the text reader's quirks, each line routed to the device it names, label blocks placed on ways, the notes kept
out of them, the main bus chosen by length, the bus-section tolerance, the meter's unit as the rules price it and the link
score evaluate.py keeps; then the rules on the answer keys of 20 fresh drawings of each font set (about 4 s)."""
import json, os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sld2quote import reader, rules
from sld2quote.evaluate import link


def bus(*bands):
    """busbar bands (x1, x2, y, thickness) as find_busbars gives them"""
    return [{"id": f"BUS-{i + 1}", "x1": x1, "x2": x2, "y1": y - t // 2, "y2": y + t // 2, "y": y} for i, (x1, x2, y, t) in enumerate(bands)]


def sym(i, kind, x, y, w, h):
    return {"id": f"S{i}", "kind": kind, "box": [x, y, w, h], "shape_score": 0.9}


def txt(i, s, x, y, w, h=16):
    return {"id": f"T{i}", "text": s, "conf": 0.95, "box": [x, y, w, h], "src": "page"}


@pytest.mark.parametrize("line, parsed, goes_to", [
    ("MCCB 630 A 3P 80 kA", {"type_word": "MCCB", "rating": 630, "poles": 3, "kA": 80}, "SWITCH"),
    ("ACB-1 1600 A4P", {"type_word": "ACB", "rating": 1600, "poles": 4}, "SWITCH"),  # a dropped space
    ("BS-1.1600-A", {"type_word": "BS", "rating": 1600}, "SWITCH"),                  # specks between number and unit
    ("FSW 8Q0 A 4P", {"rating": 800, "poles": 4}, "SWITCH"),                         # Q read for 0
    ("MCCB 1OO A", {"type_word": "MCCB", "rating": 100}, "SWITCH"),                  # a run of O
    ("TX-1 16OO kVA", {"kva": 1600}, "SOURCE"),
    ("Icu 50 kA, withdrawable", {"kA": 50}, "SWITCH"),
    ("4C 2x185 mm2 XLPE/SWA", {}, "SWITCH"),                                         # the way's cable
    ("TX-1 1600 kVA", {"kva": 1600}, "SOURCE"),
    ("11kV415V", {}, "SOURCE"),
    ("DOL l5 kW", {"kw": 15}, "CONTACTOR"),                                          # l read for 1
    ("DOI15 kW", {"kw": 15}, "CONTACTOR"),                                           # but not an I ending a word
    ("S/D 45.kW.", {"kw": 45}, "CONTACTOR"),
    ("PFC 150 kVAr", {"kvar": 150}, "CAPACITOR"),                                    # kVAr is not kVA
    ("CT 2000/5A", {"ct": 2000}, "CT"),                                              # a ratio, not 5 A
    ("SPD Type 1+2", {}, "SPD"),
    ("MCC-10 AHU", {}, "DEST"),                                                      # 10 A only if no letter follows
    ("pump P-4", {}, "DEST"),
])
def test_a_label_line_is_read_through_the_readers_quirks_and_routed_by_what_it_says(line, parsed, goes_to):
    assert reader.parse_label(line) == parsed and reader.route(line) == goes_to


# one board: a transformer incomer with its ACB, CT, meter and relay; a generator labelled on its left; outgoing ways below
# the bus with labels on the right (a motor starter chain, and a plain way whose label reaches within 12 px of the next
# way's breaker), a PFC chain labelled below, a way whose breaker the detector missed, and the notes
BUS = bus((140, 800, 400, 6), (840, 1500, 400, 6), (60, 620, 760, 4), (60, 620, 910, 4), (1040, 1540, 800, 4))  # with a scan's box borders
SYMS = [sym(1, "transformer", 196, 80, 44, 74), sym(2, "breaker", 205, 200, 26, 26), sym(3, "ct", 210, 280, 16, 20),
        sym(4, "meter", 252, 275, 30, 30), sym(5, "relay", 304, 277, 26, 26), sym(6, "generator", 1380, 66, 48, 48),
        sym(7, "board", 1250, 63, 20, 14),                                         # a false box lying on the generator's label
        sym(8, "breaker", 800, 394, 40, 23),                                       # IEC bus-section: the bus a quarter down its box
        sym(9, "breaker", 300, 520, 24, 24), sym(10, "contactor", 304, 560, 16, 30), sym(11, "motor", 298, 602, 28, 28),
        sym(12, "breaker", 482, 520, 24, 24),
        sym(13, "breaker", 700, 520, 24, 24), sym(14, "capacitor", 702, 560, 20, 30), sym(15, "earth", 702, 600, 20, 20),
        sym(16, "breaker", 1080, 520, 24, 24)]
TEXTS = [txt(1, "TX-1 1600 kVA", 256, 84, 110, 17), txt(2, "11 kV / 415 V", 256, 108, 73),
         txt(3, "ACB-1 2000 A 4P", 245, 203, 130, 17), txt(4, "Icu 50 kA, withdrawable", 245, 225, 129),
         txt(5, "kWh", 258, 285, 18, 9), txt(6, "I>", 311, 284, 12, 10), txt(7, "CT 2000/5A", 252, 311, 80),
         txt(8, "BUSBAR A  2000 A Cu  50 kA/1s", 150, 372, 304),
         txt(9, "G", 1398, 82, 12, 14), txt(10, "GEN-1 800 kVA", 1248, 62, 120, 17), txt(11, "standby", 1322, 86, 46),
         txt(12, "BS-1 2000 A", 785, 425, 70),
         txt(13, "MCCB 100 A 3P", 340, 522, 130, 17), txt(14, "DOL 22 kW", 340, 544, 80), txt(15, "pump P-9", 340, 566, 70),
         txt(16, "M", 308, 610, 9, 10),
         txt(17, "MCCB 250 A 4P", 524, 522, 120, 17), txt(18, "kitchen", 524, 544, 60),
         txt(19, "MCCB 400 A 3P", 652, 630, 120), txt(20, "PFC 200 kVAr", 667, 650, 90), txt(21, "3C 240 mm2 XLPE/SWA", 632, 670, 160),
         txt(22, "MCCB 630 A 4P", 928, 522, 130, 17),                               # the missed breaker's label, 22 px left of S16
         txt(23, "MCCB 160 A 3P", 1112, 522, 120, 17),
         txt(24, "NOTES", 72, 770, 60), txt(25, "1. Switchboard to BS EN 61439-2, Form 4 Type 6, IP54.", 72, 800, 300)]


def test_labels_go_by_ways_and_each_line_to_the_device_it_names():
    devs = {d["symbol"]: d for d in reader.make_devices(SYMS, {}, TEXTS, BUS, [], 1000)}
    got = {s: (d["type"], d["rating"], d["label_texts"]) for s, d in devs.items()}
    assert got == {
        "S1": ("TRANSFORMER", 1600, ["T1", "T2"]), "S2": ("ACB", 2000, ["T3", "T4"]),
        "S3": ("CT", 2000, ["T7"]),                     # the ratio drawn under the meter is the CT's
        "S4": ("METER", None, ["T5"]), "S5": ("RELAY", None, ["T6"]),
        "S6": ("GENERATOR", 800, ["T10", "T9", "T11"]),  # labelled on its left, its G inside
        "S7": ("BOARD", None, []),                      # a box on the text is not beside it
        "S8": ("BUS_SECTION", 2000, ["T12"]),           # 6 px off the line, still in the bus
        "S9": ("MCCB", 100, ["T13", "T15"]), "S10": ("CONTACTOR", 22, ["T14"]), "S11": ("MOTOR", 22, ["T16"]),  # kW from the starter
        "S12": ("MCCB", 250, ["T17", "T18"]),           # not T13, though it ends 12 px from this breaker
        "S13": ("MCCB", 400, ["T19", "T21"]), "S14": ("CAPACITOR", 200, ["T20"]), "S15": ("EARTH", None, []),
        "S16": ("MCCB", 160, ["T23"])}                  # T22 is contested: its own breaker was missed, so it stays unlinked
    assert devs["S9"]["dest"] == "pump P-9" and devs["S9"]["poles"] == 3 and devs["S2"]["kA"] == 50


def test_the_notes_never_join_a_ways_label_block():
    # a way just above the notes box, as on seed 10394, where NOTES and note 1 became the way's labels
    texts = [txt(1, "MCCB 250 A 3P", 340, 702, 130, 17), txt(2, "NOTES", 300, 760, 60),
             txt(3, "1. Switchboard to BS EN 61439-2, Form 4 Type 6, IP54.", 300, 790, 300)]
    assert reader.make_devices([sym(1, "breaker", 300, 700, 24, 24)], {}, texts, BUS, [], 1000)[0]["label_texts"] == ["T1"]
    assert reader.read_notes(texts)["form"] == "Form 4 Type 6"  # still the notes' own

def test_the_main_bus_is_the_longest_level_not_the_median_of_every_band():
    # the notes and title borders pass the busbar finder on a scan; their median level (780) would make S9 an incomer
    assert reader.main_bus(BUS, 1000) == 400 and reader.main_bus([], 1000) == 500
    assert [reader.role(s, {}, BUS, 400) for s in (SYMS[1], SYMS[7], SYMS[8])] == ["ACB", "BUS_SECTION", "MCCB"]
    assert not reader.touches_bus(sym(0, "breaker", 300, 404, 24, 24), BUS)  # hanging just below the bus is not in it


def test_the_meter_is_priced_by_its_unit_whatever_marks_the_reader_adds():
    ex = {"notes": {"fault_kA": 50, "form": "Form 4 Type 6", "ip": "IP54", "poles_default": 4, "evidence": []}, "busbars": [],
          "devices": [{"id": f"D{i}", "symbol": f"S{i}", "type": "METER", "label": lab, "label_texts": [], "conf": 0.9}
                      for i, lab in enumerate(["kWh]", "V.", "kW", "", "A"])]}
    assert [l["part"] for l in rules.build_bom(ex, None, [])["lines"] if l["part"].startswith("METER")] == \
        ["METER-kWh", "METER-V", "METER-kW", "METER-MF", "METER-A"]


def test_the_link_score_counts_right_wrong_and_missed_links_apart_from_finding_and_reading():
    gt = {"devices": [{"id": "OG-1"}, {"id": "OG-2"}, {"id": "OG-3"}],
          "texts": [{"text": "a", "box": [0, 0, 50, 10], "device": "OG-1"}, {"text": "b", "box": [0, 20, 50, 10], "device": "OG-2"},
                    {"text": "c", "box": [0, 40, 50, 10], "device": "OG-3"}, {"text": "n", "box": [0, 60, 50, 10], "device": None},
                    {"text": "d", "box": [0, 80, 50, 10], "device": "OG-2"}]}
    ex = {"texts": [{"id": "T1", "box": [1, 1, 48, 9]}, {"id": "T2", "box": [1, 21, 48, 9]}, {"id": "T3", "box": [1, 41, 48, 9]},
                    {"id": "T4", "box": [1, 61, 48, 9]}, {"id": "T5", "box": [300, 300, 20, 9]}],
          "devices": [{"id": "D1", "label_texts": ["T1", "T4", "T5"]}, {"id": "D2", "label_texts": ["T3"]}, {"id": "D9", "label_texts": ["T2"]}]}
    # D1 is OG-1 and D2 is OG-2; D9 matched nothing, T5 reads nothing drawn, "d" was not read and OG-3 was not found:
    # made T1 (right), T4 and T3 (wrong); due OG-1's "a" (found) and OG-2's "b" (missed)
    assert link(ex, gt, {"D1": 0, "D2": 1}) == {"link_made": 3, "link_right": 1, "link_due": 2, "link_found": 1}
    assert link(ex, {"devices": gt["devices"], "texts": [{"text": "a", "box": [0, 0, 50, 10]}]}, {"D1": 0}) == {}  # an answer key from before 19 Oct


def test_with_the_crop_pass_on_its_closer_reads_replace_the_page_reads_they_cover():
    page = [txt(1, "MCCB 25O A", 100, 100, 120), txt(2, "pumps", 100, 122, 50)]
    crop = lambda i, s, x, w: {"id": i, "text": s, "conf": 0.9, "box": [x, 99, w, 18], "src": "crop"}
    local = {"S1": [crop("S1t1", "MCCB 250 A 4P", 98, 150), crop("S1t2", "M", 60, 9)],
             "S2": [crop("S2t1", "MCCB 250 A", 98, 110)]}  # this window cut the line
    assert [t["id"] for t in reader.with_crops(page, local)] == ["S1t1", "T2", "S1t2"]  # the M: a place the page pass missed


@pytest.mark.parametrize("fonts", ["classic", "wide"])
def test_on_fresh_drawings_the_answer_keys_own_symbols_and_texts_are_linked_and_rated_without_error(tmp_path, fonts):
    """the link and role rules alone, on 20 drawings of seeds no evaluation set uses: the answer key's symbols, texts and
    busbars in, every device typed and rated as drawn and every link right out. On the rule as first reviewed (a penalty
    for a block left of a symbol instead of a ranking, five text heights below) seed 60038 failed in both sets and 60041
    in the wide one; over seeds 60000 to 60499 the rules now make no error in either set"""
    from sld2quote.classes import DEVICE_TO_CLASS
    from sld2quote.generate import write_sheet
    for seed in range(60030, 60050):
        gt = json.load(open(write_sheet(str(tmp_path), seed, fonts=fonts) + ".json"))
        syms = [sym(i, DEVICE_TO_CLASS[d["type"]], *d["box"]) for i, d in enumerate(gt["devices"])]
        texts = [txt(i, t["text"], *t["box"]) for i, t in enumerate(gt["texts"])]
        devs = reader.make_devices(syms, {}, texts, bus(*[(b["x1"], b["x2"], b["y"], b["thickness"]) for b in gt["busbars"]]), [], gt["size"][1])
        assert [(d["type"], d["rating"]) for d in devs] == [(g["type"], g["rating"]) for g in gt["devices"]], (fonts, seed)
        lk = link({"texts": texts, "devices": devs}, gt, {d["id"]: i for i, d in enumerate(devs)})
        assert lk["link_right"] == lk["link_made"] and lk["link_found"] == lk["link_due"], (fonts, seed, lk)
