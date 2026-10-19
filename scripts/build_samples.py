"""The site's precomputed samples: site/samples/<id>/ holds result.json and the files it references, written by
sld2quote.result.build_result exactly as a live run writes them, plus thumb.jpg (320 px wide JPEG of the drawing) and, for a
synthetic sheet, its answer key as answer.json and specification as spec.txt (POST /api/samples/<id>/run scores a live rerun with them);
site/samples/index.json lists the samples in display order. One command rebuilds the set from the data directories:

  python3 scripts/build_samples.py --weights weights/rtdetr_r18_full --device cpu

A sample whose result.json exists is skipped unless --force. Paths are relative to the repository root. The text engine
is RapidOCR and torch runs 4 threads unless SLD2QUOTE_OCR or OMP_NUM_THREADS say otherwise, as on the Linux site.

The set, by rule, so a rebuild picks the same sheets:
- test: the first 6 clean sheets of data/eval/test in seed order and the scan copies of the first 3 of them (both groups
  cover both levels and both breaker styles; asserted), scored against their answer keys;
- unseen: the first 4 clean sheets of data/eval/unseen in seed order (styles held out of training; same assertion);
- preview: data/preview sheets 1007 and 1009;
- real: the 19 drawings of data/real/sheets, in manifest order, unscored (no answer key yet), titled by their document and
  figure caption from data/real/manifest.json (written with SOURCES.md's tables by scripts/real_sheets/build.py).
"""
import argparse, json, os, re, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
os.environ.setdefault("SLD2QUOTE_OCR", "rapidocr")
os.environ.setdefault("OMP_NUM_THREADS", "4")  # before torch is imported
from PIL import Image

OUT, THUMB_W, MAX_MB = "site/samples", 320, 40
LEVELS, BREAKERS = {"simple", "medium"}, {"iec", "box"}
BREAKER = {"iec": "IEC breakers", "box": "box-style breakers"}
REAL_NOTE = ("Real public-domain drawing (US federal document), American symbol style the model never trained on; "
             "not scored, because it has no answer key yet")


def synthetic(folder: str, stem: str, title: str, copy: str, note: str) -> dict:
    """a generated sheet with its answer key and the clean copy's specification, as evaluate.py pairs them"""
    seed = stem.removesuffix("-scan").split("-")[1]
    return {"id": stem, "image": f"{folder}/{stem}.png", "gt": f"{folder}/{stem}.json", "spec": f"{folder}/sheet-{seed}-spec.txt",
            "source": {"title": f"{title} (sheet {seed})", "kind": "sample", "copy": copy, "has_answer_key": True, "note": note}}


def style(folder: str, stem: str) -> dict:
    return json.load(open(f"{folder}/{stem}.json"))["style"]


def first(folder: str, n: int) -> list[str]:
    """the first n clean sheets of an evaluation set in seed order, checked to cover both levels and both breaker styles"""
    stems = sorted((f[:-5] for f in os.listdir(folder) if re.fullmatch(r"sheet-\d+\.json", f)), key=lambda s: int(s.split("-")[1]))[:n]
    st = [style(folder, s) for s in stems]
    assert {s["level"] for s in st} == LEVELS and {s["breaker"] for s in st} == BREAKERS, f"{folder}: first {n} sheets miss a level or breaker style"
    return stems


def board(st: dict) -> str:
    return f"{st['level']} board, {BREAKER[st['breaker']]}"


def samples() -> list[dict]:
    test, unseen, preview = "data/eval/test", "data/eval/unseen", "data/preview"
    held = json.load(open("data/coco/manifest.json"))["held_out"]  # {"font": "Courier New.ttf", "label_pos": "below"}
    out, clean = [], first(test, 6)
    first(test, 3)  # the scan copies' sheets cover both levels and styles too
    out += [synthetic(test, s, f"Clean drawing, {board(style(test, s))}", "clean",
                      "Synthetic test drawing, never seen in training; scored against its answer key") for s in clean]
    out += [synthetic(test, s + "-scan", f"Scan-like drawing, {board(style(test, s))}", "scan",
                      "Scan-like copy of a synthetic test drawing (tilt, blur, noise); scored against its answer key") for s in clean[:3]]
    for s in first(unseen, 4):
        st = style(unseen, s)
        traits = [t for t, on in (("Courier New text", st["font"] == held["font"]), ("labels below", st["label_pos"] == held["label_pos"])) if on]
        out.append(synthetic(unseen, s, f"Unseen-style drawing ({' and '.join(traits)}), {board(st)}", "clean",
                             "Synthetic drawing in a style held out of training (Courier New text or labels below); scored against its answer key"))
    out += [synthetic(preview, s, f"Preview drawing, {board(style(preview, s))}", "clean",
                      "Synthetic drawing from the generator's preview set, not in the training pool; scored against its answer key")
            for s in ("sheet-1007", "sheet-1009")]
    for r in json.load(open("data/real/manifest.json"))["sheets"]:
        doc = " ".join(r["document_title"].split()[:2])  # "UFC 3-540-01", "TM 5-693", "NAVFAC MO-201", ...
        out.append({"id": r["sheet_id"], "image": f"data/real/{r['image']}", "gt": None, "spec": None,
                    "source": {"title": f"{doc}: {r['caption'].removeprefix('Detail title: ')}", "kind": "sample",
                               "copy": "scan" if r["kind"].startswith("scanned") else "clean", "has_answer_key": False, "note": REAL_NOTE}})
    assert len({s["id"] for s in out}) == len(out) == 34
    return out


def thumb(image: str, path: str):
    im = Image.open(image)
    im.resize((THUMB_W, round(im.height * THUMB_W / im.width)), Image.LANCZOS).convert("L").save(path, quality=85)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="weights/rtdetr_r18_full"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true", help="rebuild samples already built")
    a = ap.parse_args(argv)
    S = samples()
    for s in S:  # cheap, so always rewritten; the index goes first so the site can be built against it
        os.makedirs(f"{OUT}/{s['id']}", exist_ok=True); thumb(s["image"], f"{OUT}/{s['id']}/thumb.jpg")
        for src, name in ((s["spec"], "spec.txt"), (s["gt"], "answer.json")):
            if src:
                shutil.copyfile(src, f"{OUT}/{s['id']}/{name}")
    json.dump([{"id": s["id"], **s["source"], "thumb": "thumb.jpg"} for s in S], open(f"{OUT}/index.json", "w"), indent=1)  # bare names, as in result.json
    from sld2quote.result import build_result
    for s in S:
        d = f"{OUT}/{s['id']}"
        if a.force or not os.path.exists(f"{d}/result.json"):
            print(f"building {s['id']}", flush=True)
            build_result(image_path=s["image"], spec_path=s["spec"], out_dir=d, gt_path=s["gt"], reader="trained", weights=a.weights, device=a.device, source=s["source"])
    table(S)
    mb = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(OUT) for f in fs) / 1e6
    print(f"{OUT}: {len(S)} samples, {mb:.1f} MB")
    if mb > MAX_MB:
        raise SystemExit(f"{OUT} is {mb:.1f} MB, over the {MAX_MB} MB budget")


def table(S: list):
    """one line per sample: what the reader found and, with an answer key, evaluate.py's symbol precision and recall and price error"""
    print(f"{'sample':<21} {'copy':<5} {'devices':>7} {'total £':>9} {'sym P':>6} {'sym R':>6} {'price err':>9}  title")
    for s in S:
        r = json.load(open(f"{OUT}/{s['id']}/result.json")); sc = r["score"]
        p, rc, pe = (f"{sc['precision']:.3f}", f"{sc['recall']:.3f}", f"{sc['price_error']:.3f}") if sc else ("-", "-", "-")
        print(f"{s['id']:<21} {s['source']['copy']:<5} {len(r['devices']):>7} {r['bom']['total_gbp']:>9,.0f} {p:>6} {rc:>6} {pe:>9}  {s['source']['title'][:48]}")


if __name__ == "__main__":
    main()
