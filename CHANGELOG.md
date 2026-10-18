# Changelog

Synthetic data only; nothing here comes from R&B. Steps are those of the plan in `HANDOFF.md` section 7.

## 0.2.0, 19 October 2026

- **Symbols (Phase 2).** IEC 60617 symbols for the devices an estimator prices, breakers in two drawn styles, simple and
  medium boards, all priced by the rules; seven symbol bodies from QElectroTech via synergycodes (CC BY 3.0). Labels no
  longer overlap symbols (audited); answer keys of scan copies follow the rotation; font sets `classic` and `wide`.
- **Step 1, baseline.** The untouched classical reader measured on 30 drawings, each as a clean and a scan-like copy: the
  "before" in `eval/RESULTS.md`.
- **Step 2, training set.** Shared page tiling (`tiles.py`); COCO export split by whole drawing, with held-out styles,
  `val_unseen`, a 60-drawing `test_unseen` and a contact sheet (`export_coco.py`).
- **Step 3, detector.** RT-DETR fine-tuning, evaluation and speed bench (`train_detector.py`): duplicate suppression in the
  one inference path, best pass chosen on pooled per-class F1, kept passes, repeat-factor sampling, weight averaging, and a
  workaround for a Mac GPU fault in large weight gradients (PROBLEMS 17a). Product model `rtdetr_r18_full`;
  every run in `eval/DETECTOR.md`.
- **Step 4, trained reader.** `reader_v2.py` and `--reader classical|trained`: one pipeline with a swappable symbol finder,
  cross-tile merge filters, a per-type scorecard on whole drawings, the operating threshold chosen on whole validation
  drawings (`select_threshold.py`). Fixed: deskew turned scans the wrong way, doubling their tilt.
- **Step 5, Linux text reader.** OCR behind one function (`ocr.py`); RapidOCR chosen by measuring the candidates on our
  sheets (`evaluate_ocr.py`, `eval/OCR.md`); Apple Vision stays the Mac default.
- **Steps 8 and 9, the site.** A FastAPI back end that runs the pipeline on a generated or uploaded drawing and scores
  generated ones against their answer keys (`site/app.py`, `sld2quote/result.py`), the front end (`site/static/`),
  precomputed samples (`scripts/build_samples.py`), and a Dockerfile and deploy script for a Hugging Face Space.
- **Step 10, packaging.** README with every result taken from the evaluation files, MIT licence, `CITATION.cff`, this
  changelog, `run_demo.sh` with the trained reader, screenshots in `docs/`.
- **Step 11, first half.** 19 public-domain one-line diagrams from US federal documents, with provenance (`data/real/`);
  not labelled yet.
- **Records and tests.** `docs/PROBLEMS.md`, every problem met, in plain language; tests for the generator, export, trainer,
  text readers, readers and site (`tests/`).

Not in 0.2.0: step 6 (the full symbol-to-label rule and the label fixes), step 7 (the scorecard by cause), labels and
scores for the real drawings.

## 0.1.0, 14 October 2026

- Synthetic SLD generator with answer keys, a specification per sheet and scan-like copies; Apple Vision text reader
  (`ocr/ocr.swift`); classical reader; specification reader and cross-check; deterministic rules over an invented
  catalogue; `quotation.md`, `bom.csv` and `review.html`; evaluation against the answer keys; the plan in `HANDOFF.md`.
