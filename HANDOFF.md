# sld2quote: build handoff (locked 14 October 2026, build not started)

Written for the next build session and for Muhammad Adnan. Read this file first; it is self-contained.
Context lives in the job-hunt workspace at
`~/Documents/Work/TechJob/Portfolio/LinkedIn/Profile/04-Applications/2026-10-13_Salford-RB-Switchgear_AI-ML-Engineer-KTP-Associate_1049/`
(research 01 to 08; `01-Research/04-technology-deep-dive-multimodal-document-ai-and-safe-ai.md` holds the reference architecture this repo
miniaturises; the readable brief is in the same folder as `01-Research/RB-Switchgear-KTP-Brief-v1.html`, second tab "The business").

## 1. What this is, in one paragraph

A miniature, not a prototype, of the single-line-diagram (SLD) to quotation system that the University of Salford / R&B Switchgear KTP
(Salford posting 1049, closes 1 Nov 2026, start 1 Dec 2026) will build. Every part of the real system is present at small scale and every
part is measured: synthetic drawings with ground truth, a reader (classical now, a trained detector next), a specification reader, a
deterministic rules layer, a bill of materials (BoM) with evidence on every line, a quotation, a review page, and an evaluation harness.
Synthetic data only; illustrative prices; nothing from R&B. The numbers it produces are the ones Adnan shows the panel, with the sentence
"this is what improves when I have your real drawings".

## 2. Decisions already taken (do not reopen)

| Decision | Choice | Why |
|---|---|---|
| Scope | Miniature with trained detector, IEC-style symbols, evaluation on clean, scan-like and unseen-style sheets, plus a small public-domain real test set | Agreed by Adnan on 14 Oct 2026 |
| Detector | RT-DETR via Hugging Face `transformers` (`RTDetrForObjectDetection`, pretrained `PekingU/rtdetr_r50vd`, Apache-2.0), fine-tuned on synthetic tiles; fallback torchvision `fasterrcnn_resnet50_fpn_v2` (BSD) | Commercial-friendly licences; both libraries already installed; Ultralytics YOLO is AGPL and not installed |
| Symbols | Draw IEC 60617-style symbols from an open symbol library (the `synergycodes/ng-diagram-single-line-diagram` SLD set, derived from QElectroTech elements, CC BY 3.0) alongside the UK "box with diagonal" breaker style; attribute in README | No public labelled SLD dataset exists; realism of the synthetic symbols is what makes the numbers credible |
| Data | No R&B data, ever, in this repo. Real sheets only in a small hand-labelled public-domain test set (US federal works), never for training | Honesty and copyright |
| OCR | Apple Vision via `ocr/ocr.swift` (on-device); keep the OCR call behind one function so a different engine can be swapped in | Already works; no cloud |
| Interfaces | The JSON shapes in section 5 are the contract; a module may be replaced only if it keeps them | So classical and trained readers are compared like for like |
| Numbers | Every figure in README comes from `python3 -m sld2quote.evaluate`, never typed by hand; failures listed by cause | Credibility with an academic panel |
| Prices | Invented catalogue (`sld2quote/catalogue.json`), labelled illustrative | No supplier figures available or wanted |
| Repository | Name `sld2quote`; private on GitHub now, public later (date is Adnan's call); Zenodo DOI after Phase 6; commits are authored by Adnan only, no co-author lines | |
| Deadline | Everything is finished before 22 Oct 2026: all 11 steps of the plan in section 7 and every decided feature (the live one-link site with its three parts, the Linux text reader, the scorecard, packaging, the real-drawing test set). Nothing is dropped or deferred. Adnan then adds sld2quote as a project to his application before submitting it (window 20 to 23 Oct). Missed targets are reported honestly, never hidden. The application files are never edited from this repo | Locked by Adnan on 15 Oct 2026; replaces the earlier rule that an unfinished repo would go to interview instead |
| One-link site | A live demo on a Hugging Face Space that also hosts the whole site, in three parts: "Try it" (make a new drawing from a seed, pick a held-out one, or upload a PNG; the server runs the pipeline and scores generated drawings against their answer keys live), "How it works" (implementation, scorecard, how to reproduce), "The real project" (how the KTP would build it, from the research brief) | Chosen by Adnan on 15 Oct 2026 over a replayed static site, and over GitHub Pages or a shared claude.ai page. Hugging Face account `itsmoadnan` is logged in on this Mac (checked 15 Oct) |
| Working agreement | Nothing new is built until Adnan says go; each step is proposed first and waits for his answer | Adnan, 15 Oct 2026 |
| Symbol-to-label link | Rules, no second model: symbols on one vertical line form a way (a chain, top to bottom); text lines stacked at regular spacing beside the chain's top device or below its bottom device form the way's label block; a block goes to the way with the shortest edge-to-edge gap between the symbol box and the text box (not centre distance), right or below as tie-break, each text used once, a contested block flagged low confidence; lines are routed to devices by content (device word with A to the switching device, kW to the contactor, kVAr to the capacitor, kVA to the source, CT ratio to the CT, mm² is the way's cable, text inside a symbol is its tag, the remainder is the destination); missing poles or kA come from the notes and are marked assumed. The generator records which device each text was drawn for, and `evaluate.py` gains link precision and recall so a wrong link is told apart from a missed symbol or a misread | Agreed by Adnan on 17 Oct 2026. To be refined with industrially practised methods (a learned link model trained on estimator corrections, schedule-table joins) once he is in the KTP |
| Build standard | Real engineering, as near as possible to the product built after the KTP, never prototype shortcuts: best practice, efficient code, configuration, logging, tests, resumable runs and recorded artefacts for anything that trains or serves. The finished product works end to end, shows value, and a non-technical user can test it on the site step by step, with a plain-language explanation of how each step relates to the estimator's problem | Locked by Adnan on 17 Oct 2026 |
| Orchestration | Claude (Fable 5.1) orchestrates as the senior staff ML, AI and computer-vision engineer: defines interfaces, briefs, reviews, integrates and keeps git. Sub-agents are Opus 5.5 at most, always at maximum reasoning effort, at most three at a time, briefed as staff engineers with explicit contracts and acceptance criteria. A Fable sub-agent needs Adnan's permission. Adnan is asked whenever a decision is his | Locked by Adnan on 17 Oct 2026 |

Still open (Adnan decides, see section 10): README AI-assistance wording; LICENSE text (MIT proposed); public date; whether to add the link to form answer Q6 / Link 4; whether to email the PETRONAS/Mahidol authors for their SLD data from a Salford address later.

## 3. Environment (verified 14 Oct 2026 on this Mac)

macOS 27.0.1, Apple M2 Max, 32 GB. Python 3.12.1 with numpy 2.2.6, Pillow 11.0.0, scipy 1.14.1, torch 2.5.1 (MPS available), torchvision 0.20.1,
transformers 4.46.1, datasets 3.2.0, accelerate 1.1.1, scikit-learn 1.5.2, pandas 3.0.3, matplotlib 3.9.3, networkx 3.4.2, PyYAML 6.0.2.
Not installed: cv2, scikit-image, pytesseract/tesseract, ultralytics, timm, onnxruntime, coremltools. swiftc 6.3.3 (Apple Swift). Fonts in
`/System/Library/Fonts/Supplemental` (Arial, Arial Narrow, Verdana, Courier New, Trebuchet MS are used). Network is available for `pip` and
model-weight downloads. The Apple Vision OCR tool is compiled at `ocr/ocr` (rebuild: `swiftc -O ocr/ocr.swift -o ocr/ocr -framework Vision -framework AppKit`).

## 4. Current state of the code (v0.1.0, 1,035 lines of Python)

| File | Does | Status |
|---|---|---|
| `sld2quote/generate.py` | Synthetic sheets (1600x1000, grey PNG) with style variation and `--scan` degradation; ground truth JSON with device boxes, busbars, connections, notes, every drawn text with its box, a specification with clauses and deliberate conflicts | Works. Symbols since 15 Oct follow IEC 60617 (see Phase 2 note in section 7): breakers as the standard switch with a cross or the box with a diagonal (`style.breaker`, one per sheet), fuse switches, isolators, motor starters (contactor and motor), PFC (capacitor and earth), SPD (arrester and earth), CT with meter and sometimes a relay on incomers, downstream board boxes; seven bodies from the open library in `assets/symbols/` (CC BY 3.0, `ATTRIBUTION.md`), the rest drawn in `generate.py`. Styles: 5 fonts, 4 sizes, 2 line widths, 3 symbol sizes, labels right or below. Since 15 Oct: `--level simple` (one transformer, 4 to 9 ways), `medium` (second transformer on busbar B, bus-section, 10 to 14 ways, cable sizes) or `mix` (default, per seed); the page widens so no label overlaps a symbol, another label or the edge (audited on 300 sheets, previously 16% of sheets had a label over a symbol); scan copies record `style.rotate_deg` and `image_box()` maps ground-truth boxes into the rotated image (unmapped boxes were up to 35 px off on wide scans) |
| `ocr/ocr.swift` | Vision OCR to JSON boxes (page pixels, origin top-left) | Works. Reads small text far better when the image is upscaled 2 to 3x, which the reader does |
| `sld2quote/reader.py` | `read_sheet(image_path)`: adaptive binarise, deskew by projection, busbars by horizontal runs, vertical lines, line removal, components, shape tests (diagonal, ring), page OCR at 2x plus a second OCR pass around each symbol at 3x with same-line merging, label association, notes by the NOTES anchor | Runs; three known faults (section 6) |
| `sld2quote/spec.py` | Clauses to requirements with clause ids as evidence; `cross_check(notes, spec)` lists drawing-vs-spec conflicts | Works on the generator's clauses |
| `sld2quote/rules.py` + `catalogue.json` | Deterministic part selection (kA bands, poles, nearest frame size), busbar copper by weight, enclosure sections by form and IP, labour, docs, delivery, margin; issues raised (below_fault, missing, assumed, conflict, low_confidence); evidence ids on every line | Works |
| `sld2quote/quote.py` | `quotation.md`, `bom.csv`, `bom.json`, `extraction.json`, `spec_check.json`, `review.html` (sheet with coloured evidence boxes, device table, BoM table, issues) | Works |
| `sld2quote/cli.py` | `python3 -m sld2quote.cli run data/sheet-0001.png` | Works, 1.2 s per sheet |
| `sld2quote/evaluate.py` | Per-sheet and aggregate metrics (section 8); reference BoM built by the same rules from ground truth | Run 17 Oct on `data/baseline` (seeds 100 to 129, 30 clean + 30 scan): see `eval/RESULTS.md` |
| `sld2quote/tiles.py` | Page tiling shared by the COCO export and the trained reader's inference: 640 px tiles, stride 480, last tile flush with the edge, boxes kept in a tile at 60% visible or more, white padding for short pages | Done 17 Oct; tests in `tests/test_export.py` |
| `sld2quote/export_coco.py` | `python3 -m sld2quote.export_coco --pool data/pool --out data/coco`: splits by whole sheet in seed order (train 400, val 40, test 30 as clean and scan copies of the same sheets, test_unseen 20 of the held-out styles: font Courier New, labels below), standard COCO with 0-based ids in `classes.CLASSES` order and extra keys for per-style scoring, manifest, contact sheet | Done 17 Oct: 11,244 tiles, 47,531 boxes, 100 s; audited against the pool; 14 deliberate faults each caught by a test |
| `sld2quote/train_detector.py` | `train` (RT-DETR fine-tune: one-tile-at-a-time dataset, zoom and brightness jitter without flips, two-group AdamW, warm-up and cosine, clipping, per-epoch checkpoint with optimiser state for `--resume`, time budget summed over resumes, val every epoch, final metrics.json and card.md, operating threshold saved in config.json), `eval` (any split), `bench` (seconds per sheet on cpu and mps). Metrics: pycocotools mAP, per-class precision and recall at IoU 0.5 at the threshold that maximises mean F1 on val, recall by copy and style, false positives at tile edges | Done 17 Oct; 6 tests in `tests/test_train.py`; needs pycocotools (installed 17 Oct, noted in README) |
| `README.md`, `run_demo.sh` | Status and run instructions | Written; README numbers must be replaced by evaluate output |

Run: `python3 sld2quote/generate.py --out data --n 10 --seed 1 --scan`; `python3 -m sld2quote.cli run data/sheet-0001.png`;
`python3 -m sld2quote.evaluate --data data --out eval`. Everything runs from the repo root; `data/`, `out/`, `eval/results.json` and `ocr/ocr` are build products (gitignored).

## 5. Interface contracts (keep these exactly; extend only by adding keys)

**Extraction** (`read_sheet` output, `out/<sheet>/extraction.json`):
```
{ "image": str, "size": [W, H], "deskew_deg": float,
  "texts":   [ {"id": "T1" | "S3t2", "text": str, "conf": 0..1, "box": [x, y, w, h], "src": "page" | "crop"} ],
  "busbars": [ {"id": "BUS-1", "x1", "x2", "y1", "y2", "y", "rating": int|null, "kA": int|null, "label_text": "T5"|null} ],
  "symbols": [ {"id": "S1", "kind": "BREAKER" | "TRANSFORMER" | "CIRCLE", "box": [x, y, w, h], "shape_score": 0..1} ],
  "devices": [ {"id": "D1", "symbol": "S1", "type": "ACB"|"MCCB"|"BUS_SECTION"|"TRANSFORMER"|"GENERATOR", "box": [x,y,w,h],
                "rating": int|null, "poles": int|null, "kA": int|null, "dest": str, "label": str, "label_texts": ["S1t1"], "conf": 0..1,
                "on_bus": "BUS-1"|null, "connected": bool} ],
  "notes": {"form": str|null, "ip": str|null, "fault_kA": int|null, "poles_default": int|null, "evidence": ["T7"]} }
```
A trained reader (`reader_v2.py`) must return the same shape; its `symbols[].kind` may add classes; `shape_score` becomes the detector confidence.

**Spec** (`read_spec`): `{"clauses": [{"id": "7.3", "text": str}], "requirements": {standard, form, fault_kA, poles, ip, busbar_material, certificates, other: [{clause, text}]}, "evidence": {field: clause_id}}`.
**Conflicts** (`cross_check`): `[{"field", "drawing", "spec", "drawing_evidence": [ids], "spec_clause": id}]`.
**BoM** (`build_bom`): `{"board": {...}, "lines": [{"line", "part", "description", "qty", "unit_gbp", "total_gbp", "evidence": [ids], "rule": "R-...", "conf", "basis"}], "subtotal_gbp", "margin_gbp", "total_gbp", "issues": [{"kind", "what", "action", "evidence"}]}`.
**Ground truth** (`generate.py`): as written today; add keys, never rename. Device `type` values are the same five as above.
Coordinates are in the drawing frame; `size` is per sheet (1600 to about 4300 px wide); `style.level` and, on scan copies, `style.rotate_deg`
were added on 15 Oct. Training labels must go through `image_box()` so they match the saved scan image.
Device types since 15 Oct: the five above plus FUSE_SWITCH, ISOLATOR, CONTACTOR, MOTOR, CAPACITOR, SPD, EARTH, CT, METER, RELAY and BOARD
(ids such as `OG-3-K` hang below way `OG-3`; `ACB-1-CT`, `-PM`, `-R` sit on incomer ACB-1). Detector classes are 14, one per symbol meaning:
breaker (ACB, MCCB and BUS_SECTION in either drawn style; `style.breaker` keeps the style for a per-style score), fuse_switch, isolator,
contactor, transformer, generator, motor, board, ct, meter, relay, spd, capacitor, earth.

## 6. Known faults to fix first (diagnosed on sheet-0001, 14 Oct)

1. The incomer breaker symbol is found by the shape test (score 1.0) but then discarded as text, because the page-level OCR read its
   diagonal as a backslash and returned one box covering symbol plus label. Fix: in `find_symbols`, apply the text filter only to
   components that do not pass a strong shape test (diagonal score >= 0.9 or two-ring transformer test), and require symbol size >= 18 px
   so letters (12 to 16 px at the generator's fonts) never pass.
2. The transformer (two overlapping circles, component 45x75) is read by OCR as the digit "8" and discarded the same way; the single
   ring test also fails on a two-circle shape. Fix: for components with aspect 1.4 to 2.0, test two rings (centres at w/2, r and w/2, h-r).
3. Bold label letters with diagonals (A, R) pass the breaker test; today only the text filter removes them. The size floor in fix 1 and the
   text filter for weak-scoring components keep them out.
Also missing: bus-section and generator cases are generated but untested in the reader; label-below style untested; `evaluate.py` never run.

## 7. Phase plan with acceptance criteria

| Phase | Work | Done when | Estimate |
|---|---|---|---|
| 1 Reader fixes | Section 6 fixes; run on 30 clean sheets; fix what the per-sheet table shows | Classical reader: symbol precision and recall >= 0.98, rating accuracy >= 0.98, notes accuracy >= 0.95, zero false conflicts on clean sheets | 1 to 2 h |
| 2 IEC symbols and styles | Add a symbol style switch to the generator: `uk_box` (today), `iec` (IEC 60617 switch-with-cross breakers, transformer, generator, motor, fuse, contactor, meter where sensible) drawn from the open library's geometry; parse the library's simple SVG primitives (lines, polylines, circles, rects, paths with M/L/C) into PIL draws, or rasterise with `qlmanage -t -s 256 -o <dir> <file.svg>` (macOS Quick Look) if parsing is slow; add style families: label below, double-line busbar, withdrawable brackets, section boxes, more fonts; keep ground truth shape-agnostic (type comes from the generator, not the look). Attribution block in README (CC BY 3.0, QElectroTech and synergycodes) | Generator renders both styles; classical reader still >= 0.9 on `iec` (it may not; that is the point the detector makes) | 3 to 4 h |
| 3 Trained reader | `sld2quote/export_coco.py`: tiles 640 px with 25% overlap, classes = symbol shapes (`breaker_box`, `breaker_iec`, `transformer`, `generator`, plus any added), boxes from ground truth, splits by sheet with held-out styles; `sld2quote/train_detector.py`: RT-DETR fine-tune on MPS (batch 4 to 8, 20 to 40 min), fallback Faster R-CNN; `sld2quote/reader_v2.py`: detector for `symbols`, then the existing OCR, association, notes and roles code (factor those out of `reader.py` so both readers share them); `--reader classical|trained` in the CLI and evaluate | Trained reader >= classical on every split and >= 0.95 symbol precision and recall on the unseen-style split; mAP@0.5 reported | half a day plus training time |
| 4 Robustness and splits | Scan-like set (15) and unseen-style set (15: one font and one layout family held out of training); tune adaptive threshold and OCR merging; per-sheet failure list by cause | RESULTS.md with clean / scan / unseen columns for both readers | 1 to 2 h |
| 5 Real test set | 10 to 20 genuine SLD sheets from US federal public-domain documents (design manuals and agency reports containing one-line diagrams; record the source URL and page per sheet; no third-party copyrighted figures); a small labelling helper (extend `review.html` with editable boxes that save JSON, or a CLI that takes box lists); evaluate both readers; report separately and honestly | A "real sheets" row in RESULTS.md, however low | 3 to 4 h, mostly labelling |
| 6 Packaging | README rewritten from the evaluation output (what it is, what it is not, numbers, failure causes, how the KTP replaces each module, attribution, AI-assistance line [CONFIRM]); `tests/` with pytest smoke tests (generator determinism; reader on a fixture sheet; rules invariants: never silently lowers a kA, every line has evidence; evaluate runs on two sheets); `run_demo.sh`; `CITATION.cff`; `CHANGELOG.md`; screenshots of `review.html` in `docs/`; LICENSE (MIT proposed); `.gitignore` as now; model weights not committed (training command plus a release asset) | A clean clone runs the demo in under five minutes on a Mac | 2 h |

Order is 1, 2, 3, 4, 6, then 5 if time allows before 22 Oct; 5 is still worth doing afterwards for the interview.

**Order revised 15 Oct 2026 (Adnan):** the trained reader comes first instead of hand-tuning the classical one. (a) Baseline evaluation of
the classical reader, untouched, as the "before" number; (b) Phase 3 on the current symbol set, trained on simple and medium sheets;
(c) evaluate both readers; (d) fix the label, notes and roles faults the per-sheet table shows (shared code, so both readers gain); (e) decide
Phase 2 from the unseen-style numbers. Phase 1's symbol-shape fixes are skipped; the classical reader stays as the baseline. Then 4, 6, 5.

**Phase 2 symbols done 15 Oct (Adnan asked for the wider set before training):** IEC 60617 symbols for the LV items an estimator prices,
placed where they belong, priced by `rules.py` with illustrative catalogue lines; layout audited on 300 sheets (no text over symbols, text or the
page edge; no overlapping symbols) and `tests/test_generate.py` checks every answer-key box against the drawn ink. Still open from Phase 2:
withdrawable brackets, double-line busbar, section boxes, more fonts; a changeover-switch symbol (ATS is drawn as interlocked ACBs, a common
practice); an ANSI/IEEE 315 style, which matters for the US real test set in Phase 5. Rating and tag formats do not vary yet (all "MCCB 250 A 4P"
style); consultants vary them, so that is the next realism step for the label reader.

**Plan after the one-link decision (15 Oct, not started; each step waits for Adnan's go):** what the live site forces: (i) a text reader
that also runs on Linux, since Apple Vision is Mac only (keep Vision as the Mac engine, choose the Linux engine by measuring it on our sheets,
score both); (ii) a detector size that reads a sheet in about 15 s on the free CPU (measure RT-DETR r18vd against r50vd before training;
same model family as decided); (iii) the Space stays private until Adnan says public, because a public Space publishes the code, so README
AI wording and LICENSE come first; (iv) the free tier sleeps when idle and takes about a minute to wake. Steps: 1 baseline; 2 training set;
3 train; 4 reader_v2 and `--reader`; 5 Linux text reader, compared with Vision; 6 compare readers and fix label reading; 7 clean / scan /
unseen scorecard with failures by cause; 8 site: server API and the three-part front end, run locally; 9 Space deployed privately, tested,
public on Adnan's word; 10 package (README from numbers, tests, licence, 0.2.0); 11 real drawings (10 to 20 public-domain sheets, hand-labelled,
scored). All 11 are required before 22 Oct (locked 15 Oct, see Deadline in section 2). Proposed days: Fri 16 steps 1 to 3; Sat 17 steps 4
and 5, and sourcing the real drawings; Sun 18 steps 6 and 7; Mon 19 step 8 and labelling the real drawings; Tue 20 steps 9 (private) and 10,
and finishing the labels; Wed 21 step 11 scoring, final checks and the public switch on Adnan's word; Thu 22 Adnan updates and submits the
application. The older rule above that step 5 "is still worth doing afterwards for the interview" no longer applies.

**Days revised 17 Oct (Friday passed with no go, nothing dropped):** Sat 17 steps 1 to 3 (and the 15 Oct work committed first); Sun 18
steps 4 and 5, and sourcing the real drawings; Mon 19 steps 6 (including the symbol-to-label link above) and 7, and labelling; Tue 20
steps 8 and 9 (private Space); Wed 21 steps 10 and 11, final checks, public switch on Adnan's word; Thu 22 Adnan updates and submits the
application.

**Step 1 done, 17 Oct 14:20.** Baseline of the untouched classical reader on `data/baseline` (30 clean + 30 scan-like, seeds 100 to 129,
generated by the Phase 2 generator), in `eval/RESULTS.md`: symbol precision 0.35 clean / 0.10 scan, recall 0.09 / 0.04 (it knows only box
breakers and circles, and half the sheets use the IEC breaker style); rating accuracy on the few matches 0.72 / 0.44; notes 0.92 / 0.83;
conflicts 20 of 22 found with none false on clean sheets; BoM line recall 0.23 / 0.21; mean price error 0.77 / 0.70; 1.6 s / 2.2 s per sheet.
These are the "before" numbers every later reader is compared with.

**Step 2 done, 17 Oct 15:10.** Pool `data/pool` (900 sheets, seeds 10000 to 10899, clean and scan). `data/coco`: train 400 sheets / 9,176
tiles / 38,332 boxes; val 40 / 936 / 3,934; test_clean and test_scan 30 sheets each / 348 tiles / about 1,510 boxes; test_unseen 20 / 436 /
2,241. Rarest class relay (278 train boxes). At 640 px no tile is empty, so `--empty-keep` had nothing to act on; the notes and title blocks
appear as background inside tiles that also hold symbols. Known skew: test_unseen has 15 sheets with labels below and 7 in Courier New
(2 both), so the layout hold-out dominates; the font and label_pos keys on each tile let the trainer score the two separately. Contact sheet
checked by eye: tight boxes, correct names.

**Step 3 go, 17 Oct 15:35 (Adnan), with the size rule in force:** RT-DETR r18vd trains first (75-minute budget on MPS); it is final if it
reaches at least 0.95 precision and recall on every class on val and on test_unseen at the threshold chosen on val; otherwise r50vd trains
overnight. The CPU benchmark of both sizes decides what the live demo can serve. Smoke run on MPS (32 tiles, 2 epochs) ran end to end.

Bench (M2 Max, `weights/bench/bench.md`, untrained heads, median of 3, tiles batched): seconds per sheet for 1600x1000 (6 tiles) / 3200x1000
(14 tiles): r18vd cpu 8 threads 0.82 / 1.87, cpu 2 threads 1.31 / 3.10, mps 0.23 / 0.47; r50vd cpu 8 threads 1.65 / 3.49, cpu 2 threads
2.45 / 5.90, mps 0.44 / 0.84. r18vd is half of r50vd everywhere. A free Space's 2 shared vCPUs will be slower than 2 M2 threads; the bench
must be rerun on the Space. Training launched 15:30: r18vd, 2 epochs (1,147 steps each at batch 8, 5.7 to 6.5 images/s on MPS, about 25 min
per epoch, so 3 epochs would not fit the 75 min budget), val after each epoch, under caffeinate. One MPS fall-back to cpu:
grid_sampler_2d_backward (deformable attention). MPS training is seeded but not bitwise repeatable.

**Step 3, first run, 17 Oct 16:18 (`weights/rtdetr_r18/card.md`):** r18vd, 2 epochs, 48 min, 6.9 images/s. At the val-chosen threshold
0.15: val mAP@0.5 0.854, mean precision 0.82, mean recall 0.81, worst class precision 0.64 (fuse switch: 845 boxes predicted for 564) and
worst recall 0.48 (earth); test_clean mAP@0.5 0.888, test_scan 0.826, test_unseen 0.849; recall clean 0.90 vs scan 0.76; symbols cut by a tile
edge 0.40 vs whole 0.84; unseen font (Courier New) 0.83 but labels-below 0.72 vs labels-right 0.84. Fails the size rule. Diagnosis from the
error analysis: 92% of false positives are a second box on a symbol already found, and 78% of misses are correct boxes scored below the
threshold; val mAP was flat between epochs 1 and 2 (0.848 to 0.854) because the cosine schedule had ended. A DETR learns to suppress its
own duplicates over a long schedule (RT-DETR's reference is 72 epochs); two is far too few. Class-agnostic NMS at IoU 0.5 on the same
weights: val mAP@0.5 0.933, mean precision 0.95, worst precision 0.81, worst recall 0.63; still short.

**Decision (orchestrator, 16:45):** NMS at IoU 0.5 becomes part of the detector's post-processing, because the reader must de-duplicate
anyway (within a tile and across overlapping tiles) or parts are counted twice in the quote; metrics are reported with and without it.
Overnight queue on this Mac: r18vd with a full schedule (batch 16, epochs sized from the measured rate to fit about 4.5 h), then r50vd
(about 6.5 h), both from the hub weights, both scored on every split; the size rule is applied to the full runs on 18 Oct morning.
Rare-class sampling (earth, relay, SPD) is held back until the full run shows whether those classes still lag.

**Overnight queue launched 17 Oct 17:00** (`scripts/train_overnight.sh`, under caffeinate; log `weights/overnight_queue.log`): r18vd batch 16,
lr 1e-4, warm-up 300, 9 epochs (573 steps each, measured 6.4 images/s, about 25 min per epoch with val), budget 270 min; then r50vd batch 8
(batch 16 thrashed MPS memory at 22 GiB), 6 epochs (1,147 steps, 3.0 images/s, about 53 min per epoch), budget 390 min; then eval of every
split for both and `weights/overnight_summary.md` (per split with NMS and raw, per class with the size rule applied, curves, a plain-language
paragraph). Expected finish about 02:15 on 18 Oct, 04:20 at the latest. `detect()` is now the detector's one inference path (processor
detections at the score floor, then class-agnostic NMS at `--nms-iou`, default 0.5); config.json carries score_threshold,
score_threshold_raw and nms_iou, so the reader inherits them. On the first run's weights the new eval reproduces the NMS experiment exactly
(val mAP@0.5 0.933 at threshold 0.10). 8 trainer tests. The queue script was smoke-tested end to end with 16 tiles and 1 epoch per model.

**Training protocol changed by Adnan, 17 Oct 17:25, queue stopped and relaunched:** early stopping on the validation score (val mAP@0.5
with NMS after every pass; stop when it has not improved for 3 passes; the best pass's weights are the model), with hard caps of 10 passes
for r18vd and 8 for r50vd. If a model reaches the target within its cap, good; otherwise it stops at the cap and the summary reports it as it
is. The cosine schedule is sized to the cap. The 9-epoch r18vd run (25 minutes in) was moved to `weights/aborted_r18_full_9ep_1725`.
More passes past a cap, if Adnan decides so, are a new run from the trained weights with a fresh shorter schedule
(`train --model weights/rtdetr_r18_full --out weights/rtdetr_r18_more --epochs N --lr 5e-5`); the exact command per model goes at the end of
the overnight summary. Results are reported to Adnan decision first, in plain language, with the full tables after.

**Overnight result, 18 Oct 00:21 (`weights/overnight_summary.md`, both runs re-evaluated and reproduced exactly).** Both models stopped
early after pass 6 with pass 3 best; neither meets the size rule (3 of 14 classes each). With NMS at threshold 0.10: r18vd val mAP@0.5 0.929
(mean P 0.94, mean R 0.89), test_clean 0.939, test_scan 0.840, test_unseen 0.926; r50vd val 0.905, test_unseen 0.874, relay recall 0.13 on val
and 0.08 on test_unseen, 2.3 times slower per pass. Val curve r18vd: 0.711, 0.862, 0.929, 0.905, 0.912, 0.913. Reading: (1) the early-stop
rule (patience 3) fired while the cosine schedule, sized to the cap, was still at about 9e-5, so the low-learning-rate tail never ran; the
17 Oct 2-epoch run that annealed to zero scored the same or better with NMS (val 0.933, test_unseen 0.937), so the tail is the missing piece,
not more high-rate passes and not model size; (2) the failing classes are the rare and small ones: relay 0.7% of training boxes (precision
0.68), capacitor 1.8% (recall 0.72 to 0.78), SPD 3.2% (0.78 to 0.81), earth 5.0% and small (0.72 to 0.77); boards on test_unseen rest on
28 boxes from 5 sheets; (3) per-tile scores are a floor: symbols cut by a tile edge are found 0.50 of the time against 0.89 for whole ones,
and every symbol is whole in at least one tile, so the sheet-level reader (step 4, tiles merged) will score higher; (4) with 24 relay boxes in
test_unseen, 0.95 recall allows one miss, so the rule needs a larger unseen set to be a steady verdict (403 held-out-style sheets are unused).
Proposals to Adnan: keep r18vd; run the 1-hour annealing continuation from its best weights now; then a fresh r18vd run with repeat-factor
sampling of rare classes and a schedule that completes its decay (cap plus best pass, no early stop); widen test_unseen to 60 sheets; judge
the rule at sheet level in step 4 as well as per tile. NMS stays: without it false positives on test_unseen are 159 instead of 93.

**18 Oct morning, Adnan's go:** A (3-pass annealing continuation of r18vd, `weights/rtdetr_r18_more`) running; then C (test_unseen widened to 60
sheets, every model re-scored on it) and B (repeat-factor sampling `--rare-repeat`, fresh r18vd, cap 10, no early stop, best pass kept,
`weights/rtdetr_r18_rare`); steps 4 (reader_v2, shared pipeline, `--reader`, drawing-level per-type scorecard on data/eval/test and
data/eval/unseen) and 5 (Linux OCR behind `sld2quote/ocr.py`, chosen by measurement on data/baseline against Vision, `eval/OCR.md`) in
parallel with two more staff agents (three in all). Sourcing the real drawings for step 11 is deferred to the first free agent slot.

**A and C results, 18 Oct 04:40.** A (`weights/rtdetr_r18_more`, best pass 2): val 0.936 but test_unseen 0.912 against the parent's 0.926,
generator recall 0.98 to 0.50 and earth 0.72 to 0.55 on the unseen style; the symbols with a letter inside (G, M, meter units, I>) depend on
the font, and the gentle passes fitted the four seen fonts. The parent `weights/rtdetr_r18_full` stays the current model. C: test_unseen is
now 60 sheets (1,140 tiles, 5,495 boxes; the first 20 sheets and their tiles byte-identical to before; train, val, test byte-identical, checked by
sha256); on it r18vd full mAP@0.5 0.922, mean P 0.93, mean R 0.88, 3 classes pass; continuation 0.918, 0 pass; r50vd 0.887, 2 pass, relay
recall 0.14, twice the false positives. A `val_unseen` split (20 further held-out-style sheets, disjoint from test_unseen; `--val-unseen 20`)
now exists for model selection, so test_unseen is never used for choosing. Selection rule for B (decided 04:45 on the agent's evidence): the
best pass is the one with the highest mean per-class F1 at the operating threshold pooled over val and val_unseen (mean mAP@0.5 would have
preferred the continuation, which test_unseen shows is worse); every pass's weights are kept (`--keep-passes`) with its pooled F1, mean mAP
and worst-class min(P, R) recorded, so any rule can be applied later without retraining. Export command now:
`python3 -m sld2quote.export_coco --pool data/pool --out data/coco --unseen 60 --val-unseen 20`.

**Step 5 done, 18 Oct (`sld2quote/ocr.py`, `sld2quote/evaluate_ocr.py`, `eval/OCR.md`, `tests/test_ocr.py`).** Linux engine: RapidOCR
(rapidocr-onnxruntime 1.4.4, PP-OCRv4 detector with the English PP-OCRv3 recogniser from the hub, CPU, 2 threads). Measured on data/baseline
against the generator's own text ground truth, character accuracy on the label and notes strings, clean and scan pooled: RapidOCR 0.960,
Apple Vision 0.868, Tesseract 0.868, docTR 0.727, EasyOCR 0.732 (and 30 s per sheet); RapidOCR 2.5 s per standard sheet on 2 threads,
Vision 0.5 s. RapidOCR reads the held-out font (Courier New) at 0.98 against Vision's 0.72, and reads the single letters inside symbols that
Vision misses. The bundled Chinese recogniser drops spaces ("ACB-13200A4P"), hence the English one. Vision stays the Mac default;
`SLD2QUOTE_OCR` picks the engine; the engine loads once per process. Findings for other steps: (1) `reader.prepare()` rotates by `-ang`
where it should rotate by `+ang`, so a scan's tilt is doubled instead of removed (0.77 degrees became 1.43); assigned to the step 4 engineer;
(2) with RapidOCR the 3x crop pass is worse than its own page pass (0.914 against 0.964: strokes inside the window are read as text) and
costs 4.75 s per sheet, so the crop pass should become engine-dependent in step 6; (3) on a 10-sheet probe, 26 of 42 wrong ratings had the
label read exactly but attached to another device, which is the step 6 link rule's job; (4) "mm²" reads as "mm2" and scan noise can put a
dot between a number and its unit, so the label regexes must tolerate both; (5) the recogniser downloads on first use, so the Space warms
the engine at start-up (or the file is vendored). Space lines: requirements rapidocr-onnxruntime==1.4.4, onnxruntime==1.31.0,
opencv-python==5.0.0.93; packages.txt libgl1, libglib2.0-0. Homebrew tesseract was installed only to measure it and can be removed.

**Step 11, first half done, 18 Oct (`data/real/`, versioned except the raw PDFs; rebuild with `scripts/real_sheets/build.py` and `spec.json`,
needs pypdfium2, BSD/Apache).** 19 genuine one-line diagrams from 8 US federal documents, public domain under 17 U.S.C. 105 with each
document's own statement recorded in `data/real/SOURCES.md`: UFC 3-540-01 (4 sheets, clean CAD rasters), TM 5-693 (2), TM 5-694 (3, the ones
with the most ratings), VA PG 18-10 (2, 90 dpi), NAVFAC MO-201 (5 scans, 1990), UFC 3-520-01 2002 (1 vector), UFC 3-550-03FA (1 scan),
TM 5-698-1 (1 vector). Mix: 3 vector, 7 clean rasters, 3 low-resolution rasters, 6 scans; all ANSI/IEEE 315 symbols; two are medium-voltage
switchgear; only the TM 5-694, VA and one TM 5-693 sheet carry ampere ratings, so real-sheet scores will be far below the synthetic ones and
are reported as such. Excluded with reasons: figures credited to firms (Pope, Evans and Robbins in TM 5-811-6; Liebert, Exide, HITEC,
McGraw-Hill), national-laboratory reports (contractor-written), NRC licensee drawings, block diagrams, illegible pages; USACE, DTIC and the
Library of Congress were unreachable (HTTP 403). Doubts recorded: the VA manual and MO-201 carry no explicit public-domain sentence (basis is
17 U.S.C. 105 alone); tm5-693-p87 nearly duplicates part of tm5-694-p76. Rendered at 200 dpi (300 where symbols are small), breakers mostly
20 to 80 px. Still to do in step 11: a labelling helper and hand labels (device boxes and types), then scoring both readers.

**Step 4 done, 18 Oct (`sld2quote/reader_v2.py`, shared pipeline in `reader.py`, `--reader classical|trained`, `evaluate.py --make-sets`
builds `data/eval/test` and `data/eval/unseen` from the manifest; `eval/RESULTS.md` holds every set and reader by section).** Drawing-level
symbol precision / recall (same type, IoU 0.3), trained r18_full against classical: test clean 0.934 / 0.961 vs 0.238 / 0.059; test scan
0.928 / 0.880 vs 0.240 / 0.079; unseen clean (60) 0.912 / 0.966 vs 0.428 / 0.133; unseen scan 0.835 / 0.808 vs 0.251 / 0.100; baseline clean
0.933 / 0.967 vs 0.350 / 0.090. Types at or above 0.95 both ways: 7 of 16 on test clean (MCCB, FUSE_SWITCH, ISOLATOR, TRANSFORMER,
GENERATOR, MOTOR, METER), 7 on test scan, 5 on unseen; classical none. Price error 0.29 vs 0.83 (test clean); BoM recall 0.68 vs 0.19. The
deskew sign bug is fixed (rotate by +ang; a scan's residual tilt is now 0.05 degrees, was 1.47) and roughly doubled every scan number; the
continuation weights lose generators at drawing level too (recall 0.40 to 0.65). Seconds per sheet on cpu, 4 threads, machine shared:
trained 9.6 to 11.1 (OCR 6 to 7 of it: the per-symbol crop pass, about 0.3 s each for about 25 symbols; detector 3 to 4), classical 2 to 3.
Faults diagnosed for step 6 (shared code): (1) bus-section role allows 2 px around the busbar band but IEC bus-section boxes sit 4.5 to 6 px
off the line, so 14 of 38 are typed ACB; (2) on scans the 3 px notes and title borders blur to 4 px and pass `find_busbars(min_thick=4)`,
the median "main bus" lands at y about 790 and MCCBs are typed ACB (choose the longest or thickest band); (3) low-score false positives
(0.11 to 0.25): Courier New label blocks as boards, tall boxes over stacked symbols as contactors; (4) ratings: kW and kVAr labels are not
parsed (contactor, motor, capacitor rated 0), the generator's label sits to its left outside the OCR window, meters take the CT ratio text.
Decided 18 Oct: two merge filters at the cross-tile merge (drop boxes cut by an inner tile edge; suppress a box that mostly contains a
better-scoring one), measured at +2 to +6 points of precision for at most half a point of recall (unseen clean 0.919 / 0.974 to 0.952 /
0.974, classes passing 4 to 8 at detector level). For the live site: drop the crop pass when the engine is RapidOCR (step 6) and keep the
detector on r18, so a sheet should take about 6 to 9 s on the Space; rerun the bench there.
Merge filters built and committed (41107bc): drawing-level precision test clean 0.934 to 0.958, unseen clean 0.912 to 0.942, baseline
clean 0.933 to 0.959, recall unchanged; types passing 0.95 on unseen clean 5 to 8. A refinement that keeps a cut box when no whole
counterpart exists in another tile was measured and rejected: it regains about 3 true symbols per set but adds 6 to 15 false boxes (cut-box
detections with no counterpart are mostly false), costing 0.4 to 1.1 points of precision on every set; the unconditional edge rule stands.

**B result, 18 Oct 09:58 (`weights/rtdetr_r18_rare`, best pass 4 of 10 by pooled F1 0.920; every pass kept; `eval/DETECTOR.md` is the
four-run summary).** Per tile on the 60-sheet test_unseen, B against r18_full: mAP@0.5 0.919 / 0.922, mean precision 0.968 / 0.932, mean
recall 0.873 / 0.880, false positives 88 / 166, classes passing the 0.95 rule 4 (fuse switch, transformer, generator, meter) / 3. Rare
classes: capacitor recall 0.67 to 0.81; relay precision 0.74 to 0.97 but recall 0.93 to 0.76; SPD recall 0.80 to 0.66; earth unchanged at
0.71 (in 20% of tiles, so sampling does not reach it); generator passes in both. The remaining misses on the unseen style are the symbols
with a glyph inside (breaker 0.82, motor 0.84, isolator 0.85) and small ones (SPD, earth), consistent with the font hold-out, not with
class frequency. All ten passes ran (294 min); passes 1 to 7 were slowed by the other engineers' jobs. Options not started: more fonts in
the generator and a retrain (a dataset change, about 4 h); per-class thresholds chosen on val and val_unseen; the product decision at
drawing level, which is being measured now by the reader engineer.

**Drawing-level verdict on B, 18 Oct 10:50 (`eval/RESULTS.md`, r18_rare sections):** keep r18_full as the product model. r18_rare is more
precise (unseen clean 0.964 against 0.942, 10 types passing against 8) but no longer detects the box-style bus-section (BUS_SECTION recall
0.714 to 0.333 on unseen clean; the IEC style is still found), so quotes lose the bus-section lines: price error is worse on all six columns
(unseen clean 0.232 to 0.279, baseline clean 0.208 to 0.240) and BoM recall lower on four. Cause: repeat-factor sampling rebalances by
class, and the in-busbar breaker is a rare appearance of the most common class, so it lost weight; r18_full's higher ACB count was partly
those mistyped bus-sections. r18_rare's other gains are real (boards no longer misread on Courier New text, relay precision 1.0, CT and
meter complete). Proposed next runs: sampling with the in-busbar breaker as its own group; and more fonts in the generator (the unseen
font explains the remaining misses on symbols with a glyph inside), with the drawing-level check as the judge.

**18 Oct 11:30, Adnan:** all commits pushed to GitHub (master, private); weights stay on this Mac until no more training is needed, then the
final model goes to his Hugging Face account and to a GitHub release. Night 2 queue (`scripts/train_night2.sh`): run 1 `rtdetr_r18_grouped`
(repeat-factor sampling grouped by device_type, so the in-busbar breaker keeps its weight; data/coco), then run 2 `rtdetr_r18_fonts`
(generator font set "wide", about 15 fonts, new pool seeds 20000 to 20899 at data/pool_fonts, export data/coco_fonts with Courier New and
Tahoma plus labels below held out; no rare sampling, so the two effects are measured apart). Existing seeds must reproduce byte-identically
under the default "classic" font set. Morning: both scored at drawing level against r18_full by the reader engineer; the best becomes the
product model. Step 6 starts after that, with Adnan's go. Errors in any run are handled by diagnosing the root cause, fixing, testing and
recording; no run is restarted more than once without a report.

## 8. Metrics (as implemented in `evaluate.py`; keep the names)

symbol precision and recall (IoU >= 0.3, same type); rating accuracy, poles accuracy, kA accuracy over matched devices; notes accuracy over
form, ip, fault_kA, poles_default; conflicts found / expected / false; BoM line precision and recall over (part, qty) multisets against the
reference BoM built from ground truth by the same rules; price error = |pred total - reference total| / reference total; seconds per sheet.
Add for Phase 3: mAP@0.5 and per-class recall from the detector evaluation; add for Phase 4: a "cause" column in the per-sheet table.

## 9. Conventions

British English in prose; no em-dashes; short functions; no new dependencies without noting them in README; every module keeps the
section 5 contracts; `python3 -m sld2quote.evaluate` before any README number changes; version bump to 0.2.0 after Phase 3; commit after each
phase with a message that names the phase and the numbers; never commit `data/`, `out/`, weights or the OCR binary. Nothing in this repo
references R&B's drawings, parts or prices. Synthetic is said out loud in every output.

## 10. Open items for Adnan

1. README AI-assistance wording (current placeholder: "written with AI coding agents as implementation tools under the author's direction and tests").
2. LICENSE: MIT in Adnan's name, or another.
3. When the repository goes public and whether the link goes into form answer Q6 / Link 4 (only if Phases 1 to 4 and 6 are done by 22 Oct).
4. Whether to request the PETRONAS/Mahidol SLD data from a Salford address after joining.
5. Zenodo DOI after Phase 6.

## 11. First actions for the next session

1. Clone `github.com/CaptiveON/sld2quote` (private) into a fresh directory, or open the local copy at `~/Documents/techproject/experimental/thmillion/practical/sld2quote`; `git log --oneline` shows this handoff as the first commit.
2. Done 17 Oct: `python3 sld2quote/generate.py --out data/baseline --n 30 --seed 100 --scan` then `python3 -m sld2quote.evaluate --data data/baseline --out eval` gave the baseline table in `eval/RESULTS.md`.
3. Superseded on 15 Oct: follow the revised order in section 7 (baseline, then Phase 3). A preview of the training sheets
   (seeds 1001 to 1012) is in `data/preview/`; on its 12 clean sheets the untouched classical reader found 17% of symbols (27% before the
   wider symbol set), so the trained reader is the next step. `reader._runs` now returns plain ints (a busbar merge leaked a numpy int and
   crashed the CLI's JSON output on the new sheets).
4. Fetch the symbol library (`git clone https://github.com/synergycodes/ng-diagram-single-line-diagram` into a scratch directory, not into this repo; copy only the symbol SVGs with their licence file into `assets/symbols/` with an ATTRIBUTION.md), then Phase 2.
5. Phase 3 training on this Mac (MPS); check `torch.backends.mps.is_available()` first; download `PekingU/rtdetr_r50vd` weights once.
