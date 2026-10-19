# sld2quote

**v0.3.0, 19 October 2026.** A miniature of the system the University of Salford and R&B Switchgear Knowledge Transfer
Partnership (KTP) sets out to build: it reads a low-voltage switchboard single-line diagram (SLD) and its specification,
builds a bill of materials with the evidence behind every line, prices it, writes a draft quotation for an estimator to
check, and measures every step against an answer key. It was built as evidence for a job application and is deliberately
small: every drawing it was trained and scored on is synthetic, drawn by its own generator; the catalogue, part
numbers and prices are invented; nothing in it comes from R&B.

Live demo: https://sld2quote-500318073364.europe-west1.run.app (one page: try it on precomputed samples, a freshly generated drawing or your own upload; how it
works, with the scorecards; the road map of the real project). It runs on Google Cloud Run and sleeps when nobody visits, so the first
visit takes about 20 seconds to wake it; a run then takes 16 to 50 seconds depending on the drawing.

![The review page for a synthetic drawing the detector never saw](docs/review-sheet-1007.png)
*The review page (`review.html`) for a synthetic drawing the detector never saw (`sheet-1007`, the site's reader): each
device box with its calibrated confidence (how often a box with that score was right on the validation drawings), the
rating text used (blue) and the notes used (copper). The device table continues below it, then the bill of materials with
the evidence for each line, and the issues raised for the estimator.*

## What it does

1. **Draws** a synthetic switchboard SLD with IEC 60617 symbols for the devices an estimator prices (breakers in two drawn
   styles, fuse switches, isolators, motor starters, power-factor correction, surge protection, metering, downstream
   boards), in varied fonts, sizes and layouts, with a scan-like copy, its answer key, and a short specification that
   sometimes conflicts with the drawing on purpose (`sld2quote/generate.py`).
2. **Finds the symbols** with an RT-DETR detector fine-tuned on synthetic drawings, reading the page in overlapping tiles
   and merging them (`reader_v2.py`); the classical shape tests it replaced are kept as the baseline (`reader.py`).
3. **Reads the text** with RapidOCR, or Apple Vision on a Mac (`ocr.py`), finds the busbars and lines, and links the
   labels: symbols on one vertical line form a way, the lines stacked beside or below it form its label block, and each
   line goes to the device it names (a rating to the breaker, kW to the starter, kVAr to the capacitor, a CT ratio to the
   CT); then reads the notes block: form, IP rating, fault level, poles (`reader.py`).
4. **Reads the specification** clause by clause and lists where it conflicts with the drawing, citing the clause
   (`spec.py`).
5. **Prices** with deterministic rules over the invented catalogue: every line names the symbol, text or clause it rests
   on, and anything doubtful is raised for the estimator, never silently fixed (`rules.py`, `catalogue.json`); then writes
   `quotation.md`, `bom.csv` and `review.html` (`quote.py`).
6. **Measures** symbols, label links, ratings, notes, conflicts, bill-of-materials lines and price against the answer key
   (`evaluate.py`), and the detector and the text readers on their own (`train_detector.py eval`, `evaluate_ocr.py`).

## Results

Every result figure in this README is copied by a script from the files the evaluations wrote, not typed by hand:
[`eval/RESULTS.md`](eval/RESULTS.md) (every set, reader, device type and sheet), [`eval/DETECTOR.md`](eval/DETECTOR.md)
(every training run, per class) and [`eval/OCR.md`](eval/OCR.md) (every engine). Synthetic drawings only.

### Whole drawings: classical against trained reader, and the label link before and after step 6

| Measure | Reader | Test, clean (30) | Test, scan-like (30) | Unseen style, clean (60) | Unseen style, scan-like (60) | Baseline, clean (30) | Baseline, scan-like (30) |
|---|---|---|---|---|---|---|---|
| Symbol precision | classical, Vision | 0.238 | 0.240 | 0.428 | 0.251 | 0.350 | 0.213 |
|  | trained, old link | 0.982 | 0.938 | 0.983 | 0.861 | 0.981 | 0.848 |
|  | trained, step 6 | 0.990 | 0.995 | 0.989 | 1.000 | 0.992 | 0.997 |
| Symbol recall | classical, Vision | 0.059 | 0.079 | 0.133 | 0.100 | 0.090 | 0.073 |
|  | trained, old link | 0.973 | 0.924 | 0.979 | 0.839 | 0.972 | 0.825 |
|  | trained, step 6 | 0.981 | 0.981 | 0.985 | 0.974 | 0.983 | 0.970 |
| Link precision / recall | trained, old link | 0.738 / 0.612 | 0.741 / 0.637 | 0.743 / 0.653 | 0.719 / 0.664 | 0.750 / 0.625 | 0.720 / 0.646 |
|  | trained, step 6 | 1.000 / 1.000 | 0.999 / 0.999 | 0.998 / 0.999 | 0.997 / 0.999 | 0.999 / 1.000 | 0.997 / 0.999 |
| Ratings read right | trained, old link | 0.753 | 0.608 | 0.556 | 0.531 | 0.672 | 0.640 |
|  | trained, step 6 | 1.000 | 0.995 | 0.813 | 0.840 | 0.897 | 0.911 |
| Price error, mean | classical, Vision | 0.833 | 0.751 | 0.727 | 0.624 | 0.772 | 0.746 |
|  | trained, old link | 0.125 | 0.333 | 0.249 | 0.316 | 0.189 | 0.368 |
|  | trained, step 6 | 0.014 | 0.014 | 0.125 | 0.098 | 0.054 | 0.051 |

Measured by `python3 -m sld2quote.evaluate` on a Mac. The trained rows are the site's reader, measured on
19 October 2026: the detector `weights/rtdetr_r18_plain_ema_fix_pass4` (the RESULTS sections name its run,
`rtdetr_r18_plain_ema_fix`) on 4 CPU threads at score threshold 0.11, chosen on whole validation drawings, with RapidOCR
reading the text and the close-up pass around each symbol off; *old link* is the same reader before step 6, which
took the nearest text to the right or below a symbol. The classical rows are the shape-test reader as measured on
18 October 2026, with Apple Vision and the close-up pass. *Test* is 30 drawings in the styles the detector trained on,
none of which it saw in training; *unseen style* is 60 drawings in styles held out of training (the font
Courier New, or labels below the symbols); *baseline* is the 30 drawings of the first measurement, in every style;
each comes as a clean and a scan-like copy. A symbol counts as found when its device type is right and its box overlaps
the answer key's by IoU 0.3 or more. Link precision and recall score the symbol-to-label link alone, counting only links
between a device the answer key matched and a text the reader located, so a wrong link is told apart from a missed symbol
or a misread text. Price error is the gap between the quoted total and the total the same rules give on the answer key,
as a share of that total, so it measures reading, not pricing. One caution: the step 6 link and role rules were designed
by reading the error lists of these same sets, so on them the rules are scored in sample ("unseen" is held out from the
detector, not from the rule author); the out-of-sample check feeds the answer key's own symbols, texts and busbars to
the rules on 500 fresh seeds of each font set (60000 to 60499, used for no choice), where the reviewed rules
make no link or rating error.

### The detector on its own, per tile

| Set | Tiles | mAP@0.5 | Mean precision | Mean recall |
|---|---|---|---|---|
| Test, clean | 348 | 0.912 | 0.962 | 0.888 |
| Test, scan-like | 348 | 0.907 | 0.962 | 0.843 |
| Unseen style | 1,140 | 0.937 | 0.955 | 0.904 |

From the `rtdetr_r18_plain_ema_fix` section of `eval/DETECTOR.md` (committed 19 October 2026): RT-DETR r18vd with averaged
weights and the gradient fix, on 640 px tiles with duplicate suppression, at the threshold chosen on validation tiles
(0.06), mAP@0.5 from pycocotools. These rows are the run's final weights (pass 10, chosen per tile); the
site uses its pass 4, chosen on whole validation drawings, which scored mAP@0.5 0.933 on validation tiles and
0.933 on unseen-style validation tiles, against 0.925 and 0.930 for pass 10 (the run's training
curve). A symbol cut by a tile edge counts as missed in that tile, although the reader can find it whole in a
neighbouring one, so these are a floor. Against the project's bar of 0.95 precision and recall for every class on both
validation and unseen-style tiles, 3 of 14 classes pass (fuse switch, transformer, current transformer); furthest off on
unseen-style tiles: relay, earth, generator, motor.

![Training tiles with their answer-key boxes](docs/contact-sheet-tiles.png)
*Training tiles cut from synthetic drawings, with their answer-key boxes (the top of `data/coco/contact_sheet.png`, which
the export writes for checking by eye).*

### Text readers

| Engine | Sheets | Labels and notes, page pass | Labels, close-up pass | Seconds per standard sheet, page pass |
|---|---|---|---|---|
| RapidOCR, English recogniser (the site's engine) | 60 | 0.960 | 0.914 | 2.51 |
| Apple Vision (Mac default) | 60 | 0.868 | 0.953 | 0.46 |
| Tesseract | 60 | 0.868 | 0.820 | 0.59 |
| docTR | 60 | 0.727 | 0.907 | 4.23 |
| EasyOCR | 10 | 0.732 | 0.876 | 32.03 |

From `eval/OCR.md`, measured on 18 October 2026 by `python3 -m sld2quote.evaluate_ocr` on the baseline sheets (clean and
scan-like) against the generator's own text: character accuracy on the rating labels and the notes lines, a string the
engine misses scoring 0, and wall time on 2 CPU threads (Vision's threads are set by macOS). RapidOCR is the site's
engine and the default wherever the Vision tool cannot run, with the close-up pass off; Apple Vision is still the Mac
default, with the close-up pass on, where it reads labels better than RapidOCR (fourth column).

## Where it fails, and what remains

From the per-type tables in `eval/RESULTS.md` and the causes recorded in [`docs/PROBLEMS.md`](docs/PROBLEMS.md):

- **A cable drawn through its own label.** On the classic drawings with labels below, a plain feeder's cable runs
  through the rating line, and the text reader returns readings such as "8b0 A": the link is right and the reading
  wrong. Most unseen-style drawings have labels below, so on their clean copies ratings read right are 0.813
  (MCCB 0.568, fuse switch 0.610) against 1.000 on the test set, whose labels sit to the right; that is
  most of the price error left on the unseen and baseline sets (PROBLEMS 27).
- **A missed contactor loses its motor's kW.** A motor's kW is read from its starter's line, so a contactor the detector
  misses leaves its motor unrated too (PROBLEMS 24).
- **Scans.** The text reader's slips inside numbers on scans ("16OO kVA", "A4P") are repaired where a number starts
  (PROBLEMS 26), but the notes block still reads worse (notes right 0.854 to 0.942 on scan-like copies against
  0.933 to 0.983 on clean ones) and small symbols are missed more often (earth recall 0.667 to 0.882 against
  0.889 to 1.000).
- **Rare and small symbols.** Relays have the fewest training examples (recall 0.273 to 0.882 over the six columns) and
  earths are small; per-type results also swing from one training pass to the next, and a wider font set left the
  relay unlearned (PROBLEMS 10 to 10c).
- **Ghost parts on blank paper** at low thresholds, held down by choosing the threshold on whole validation drawings,
  never per tile (PROBLEMS 10b).
- **Speed.** The site's reader takes 5.2 to 7.6 s a drawing on 4 CPU threads of this Mac: 2.9 to 4.3 s of text
  reading and 2.1 to 3.1 s of detection (PROBLEMS 19).
- **Real drawings** use American ANSI/IEEE 315 symbols, which the detector has never seen; they are not labelled or scored
  yet (see below).

Not done in 0.3.0: the scorecard of failures by cause; hand labels and scores for the real drawings; the wide font set
through the whole reader (its labels start under the cable, so they are not crossed, but that is not measured end to
end); and relays on the wide font set (PROBLEMS 10c). The plan and every result so far, dated, are in
[`HANDOFF.md`](HANDOFF.md) section 7.

## Run it

Python 3.12; `site/requirements.txt` pins the versions tested (on macOS; on Linux without a GPU, install torch from the
CPU index first, as its header says).

```sh
git clone https://github.com/CaptiveON/sld2quote && cd sld2quote
python3 -m pip install -r site/requirements.txt   # the pipeline, both readers, RapidOCR and the site
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('itsmoadnan/sld2quote-detector', local_dir='weights/rtdetr_r18_plain_ema_fix_pass4')"
./run_demo.sh
```

`run_demo.sh` builds the Apple Vision text tool on a Mac (`ocr/ocr.swift`, needs the Xcode command-line tools), draws
three synthetic sheets with their scan-like copies into `data/demo`, quotes one with the trained reader (or, with a note,
the classical one when the weights are missing) and prints where the outputs are.

One sheet, read as the site reads it:

```sh
SLD2QUOTE_OCR=rapidocr python3 -m sld2quote.cli run data/demo/sheet-0002-scan.png --reader trained \
  --weights weights/rtdetr_r18_plain_ema_fix_pass4 --device cpu
# writes out/sheet-0002-scan/: quotation.md, bom.csv, bom.json, review.html, extraction.json, spec_check.json
```

`--spec` names the specification (by default the `-spec.txt` beside a generated sheet; without one the cross-check is
skipped), `--out` the output folder, `--weights` the detector (without it the reader takes `weights/rtdetr_r18_full`, the
0.2.0 model), `--device` cpu or mps (default mps when available), `--crops 0|1` the close-up pass (default on with
Vision, off with RapidOCR). Any PNG can be read, but the detector knows only the synthetic IEC style.

**Text reader on Linux.** With no Vision tool the reader uses RapidOCR, installed above; its OpenCV needs
`sudo apt-get install libgl1 libglib2.0-0`, and its English recogniser (9 MB) is fetched once from the Hugging Face hub.
`SLD2QUOTE_OCR=rapidocr` picks it on a Mac too, `SLD2QUOTE_OCR=vision` the Mac engine.

**The site.** `uvicorn app:app --app-dir site --port 7860` serves it at http://localhost:7860, live runs included
(`site/app.py`, which loads `weights/rtdetr_r18_plain_ema_fix_pass4`); `site/Dockerfile` builds the server's image for
Cloud Run (or a Docker Space).

**Evaluation.** Each run writes its own section of `eval/RESULTS.md`, titled by its data, reader, weights and engine;
`SLD2QUOTE_OCR=rapidocr` and `OMP_NUM_THREADS=4` give the site's engine and the thread count used above.

```sh
python3 sld2quote/generate.py --out data/baseline --n 30 --seed 100 --scan                  # 30 drawings, clean and scan-like
python3 -m sld2quote.evaluate --data data/baseline --out eval                                # classical reader
SLD2QUOTE_OCR=rapidocr python3 -m sld2quote.evaluate --data data/baseline --out eval --reader trained \
  --weights weights/rtdetr_r18_plain_ema_fix_pass4 --device cpu                              # the site's reader

# the test and unseen-style sets come from the training pool and its export
python3 sld2quote/generate.py --out data/pool --n 900 --seed 10000 --scan
python3 -m sld2quote.export_coco --pool data/pool --out data/coco --unseen 60 --val-unseen 20
python3 -m sld2quote.evaluate --make-sets                     # data/eval/test, unseen, val and val_unseen
for s in test unseen; do
  SLD2QUOTE_OCR=rapidocr python3 -m sld2quote.evaluate --data data/eval/$s --out eval --reader trained \
    --weights weights/rtdetr_r18_plain_ema_fix_pass4 --device cpu
done
python3 -m pip install pycocotools==2.0.11                    # the detector per tile (eval/DETECTOR.md)
python3 -m sld2quote.train_detector eval --weights weights/rtdetr_r18_plain_ema_fix_pass4 --data data/coco --split test_unseen --nms-iou 0.5 --device cpu
python3 -m sld2quote.evaluate_ocr --data data/baseline --engines vision,rapidocr --crops    # eval/OCR.md

python3 -m pip install pytest && python3 -m pytest -q tests   # tests needing absent weights, hub models or OCR engines skip
```

## The trained model

The detector is RT-DETR with the r18vd backbone, fine-tuned from `PekingU/rtdetr_r18vd` on tiles cut from synthetic
drawings, with averaged weights (EMA) and the gradient fix (PROBLEMS 17a): pass 4 of the run `rtdetr_r18_plain_ema_fix`, chosen on whole
validation drawings, with its drawing threshold and the calibration of its confidence stored in its `config.json`. The
weights are not kept in git: they are in the Hugging Face model repository
[itsmoadnan/sld2quote-detector](https://huggingface.co/itsmoadnan/sld2quote-detector) (the download line above puts them
in `weights/rtdetr_r18_plain_ema_fix_pass4`, where the site looks) and attached to the GitHub release:
<release asset URL>.

To train it again on a Mac GPU, with the data built as under Evaluation (the first command is the one recorded for the
run, with a new output folder, and keeps every pass; the second chooses a pass's threshold and calibration on whole
validation drawings; the site's model is pass 4 of the recorded run):

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m sld2quote.train_detector train --data data/coco --out weights/rtdetr_r18_new \
  --model PekingU/rtdetr_r18vd --epochs 10 --batch 16 --lr 1e-4 --backbone-lr 1e-5 --weight-decay 1e-4 --warmup-steps 300 \
  --grad-clip 0.1 --patience 3 --min-delta 0.002 --select f1 --keep-passes --eval-every 1 --nms-iou 0.5 --device mps \
  --seed 0 --time-budget-min 420 --ema 0.999 --ema-warmup 200
python3 -m sld2quote.select_threshold --weights weights/rtdetr_r18_new/pass-4 --sets data/eval/val,data/eval/val_unseen --device cpu
```

Training on the Mac GPU is seeded but not bit-for-bit repeatable, so a rerun gives a close model, not the same one.

## How the KTP would replace each part

| Here | In the KTP |
|---|---|
| Synthetic generator and answer keys | R&B's drawings labelled in a review tool, and a generator seeded with their real board topologies |
| RT-DETR detector trained on synthetic tiles | The same detector retrained on labelled R&B drawings, behind the same interface |
| Apple Vision or RapidOCR | Whichever engine measures best on R&B's drawings, run inside the company's boundary |
| Label linking by rules | A learned link model trained on estimators' corrections, and joins to schedule tables |
| Specification reader by rules | An LLM extractor with a fixed schema that cites the clause for every field |
| Invented catalogue and rules | R&B's design rules and supplier catalogues |
| Static review page | The estimator's workbench, where every correction becomes training data |
| Evaluation on synthetic sets | A golden set of real drawings and a regression gate on every model change |

## Real test set

`data/real/` holds 19 genuine one-line diagrams from 8 US federal documents (Unified Facilities Criteria, Army
technical manuals, the VA electrical design manual and a NAVFAC manual), public domain as works of the US government
(17 U.S.C. 105). [`data/real/SOURCES.md`](data/real/SOURCES.md) records each sheet's document, URL, page and the rights statement found, what was
excluded and why, and the doubts; `manifest.json` holds the render settings and image hashes. They are for scoring only,
never for training. They are not labelled yet, so no score is reported; they use American ANSI/IEEE 315 symbols, which
the detector has never seen.

## Attribution

- Symbol artwork in `assets/symbols/` (breaker, disconnector, current transformer, earth, protection relay, capacitor,
  surge arrester): IEC 60617 symbols from the QElectroTech elements library (https://qelectrotech.org/, by the
  QElectroTech team and contributors), licensed CC BY 3.0, taken unchanged from
  [synergycodes/ng-diagram-single-line-diagram](https://github.com/synergycodes/ng-diagram-single-line-diagram); details
  in [`assets/symbols/ATTRIBUTION.md`](assets/symbols/ATTRIBUTION.md).
- RT-DETR (Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024;
  https://github.com/lyuwenyu/RT-DETR), Apache-2.0: the pretrained `PekingU/rtdetr_r18vd` weights, used through Hugging
  Face `transformers` (Apache-2.0). The trainer's weight averaging follows the authors' recipe.
- RapidOCR (https://github.com/RapidAI/RapidOCR), Apache-2.0, running PaddleOCR's PP-OCR models (Apache-2.0); the English
  recogniser comes from the `SWHL/RapidOCR` repository on the Hugging Face hub.
- The real drawings in `data/real/sheets/` come from US federal government documents and are in the public domain
  (17 U.S.C. 105); sources in `data/real/SOURCES.md`.

## Licence

MIT ([`LICENSE`](LICENSE)), copyright 2026 Muhammad Adnan. The symbol artwork in `assets/symbols/` keeps its CC BY 3.0 licence and
the drawings in `data/real/sheets/` are public domain; dependencies and downloaded models keep their own licences.

## AI assistance

Written with AI coding agents (Claude, by Anthropic) as implementation tools: they wrote the code, tests and documents
under the author's direction, and the author set the scope and the plan, took the decisions, set the acceptance tests and
reviewed the results. [CONFIRM wording with the author]

## More detail

- [`HANDOFF.md`](HANDOFF.md): the build record, with every decision, step and result, dated.
- [`docs/PROBLEMS.md`](docs/PROBLEMS.md): every real problem met, why it happened, what was done, and what it means for the KTP.
- [`eval/`](eval/): the full results (`RESULTS.md`, `DETECTOR.md`, `OCR.md`).
- [`CHANGELOG.md`](CHANGELOG.md) and [`CITATION.cff`](CITATION.cff).
