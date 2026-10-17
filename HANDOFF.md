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
| Orchestration | Claude (Fable 5.1) orchestrates as the senior staff ML, AI and computer-vision engineer: defines interfaces, briefs, reviews, integrates and keeps git. Sub-agents are Opus 5.5 at most, at most three at a time, briefed as staff engineers with explicit contracts and acceptance criteria. A Fable sub-agent needs Adnan's permission. Adnan is asked whenever a decision is his | Locked by Adnan on 17 Oct 2026 |

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
| `sld2quote/evaluate.py` | Per-sheet and aggregate metrics (section 8); reference BoM built by the same rules from ground truth | Written, not yet run on a batch |
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
2. `python3 sld2quote/generate.py --out data --n 30 --seed 100 --scan` then `python3 -m sld2quote.evaluate --data data --out eval` to get the baseline table before touching code.
3. Superseded on 15 Oct: follow the revised order in section 7 (baseline, then Phase 3). A preview of the training sheets
   (seeds 1001 to 1012) is in `data/preview/`; on its 12 clean sheets the untouched classical reader found 17% of symbols (27% before the
   wider symbol set), so the trained reader is the next step. `reader._runs` now returns plain ints (a busbar merge leaked a numpy int and
   crashed the CLI's JSON output on the new sheets).
4. Fetch the symbol library (`git clone https://github.com/synergycodes/ng-diagram-single-line-diagram` into a scratch directory, not into this repo; copy only the symbol SVGs with their licence file into `assets/symbols/` with an ATTRIBUTION.md), then Phase 2.
5. Phase 3 training on this Mac (MPS); check `torch.backends.mps.is_available()` first; download `PekingU/rtdetr_r50vd` weights once.
