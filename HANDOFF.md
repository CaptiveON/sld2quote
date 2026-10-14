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
| Deadline | The job application is submitted 20 to 23 Oct 2026; if the repo is not complete by 22 Oct it is not linked from the form and goes to interview instead. The application files are never edited from this repo | |

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
| `sld2quote/generate.py` | Synthetic sheets (1600x1000, grey PNG) with style variation and `--scan` degradation; ground truth JSON with device boxes, busbars, connections, notes, every drawn text with its box, a specification with clauses and deliberate conflicts | Works. Symbols are simplified (square with diagonal; two circles; circle with G). Styles: 5 fonts, 4 sizes, 2 line widths, 3 symbol sizes, labels right or below |
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
3. Phase 1 fixes in `reader.py`; re-run; commit "Phase 1: ...".
4. Fetch the symbol library (`git clone https://github.com/synergycodes/ng-diagram-single-line-diagram` into a scratch directory, not into this repo; copy only the symbol SVGs with their licence file into `assets/symbols/` with an ATTRIBUTION.md), then Phase 2.
5. Phase 3 training on this Mac (MPS); check `torch.backends.mps.is_available()` first; download `PekingU/rtdetr_r50vd` weights once.
