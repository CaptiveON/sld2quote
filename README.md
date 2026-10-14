# sld2quote (prototype, v0.1, 14 October 2026)

A small, honest demonstration of the single-line-diagram to quotation pipeline that the University of Salford /
R&B Switchgear KTP project describes: read a drawing and a specification, build a bill of materials with the evidence
behind every line, price it, write the quotation, flag what a person must decide, and measure the whole thing against
ground truth. **Synthetic data only. Nothing here is an R&B drawing, part number or price.**

Status (14 Oct 2026): the pipeline runs end to end on synthetic sheets in about a second; the classical reader is
incomplete (see `eval/RESULTS.md` once it exists; on the first sheet it read 3 of 6 devices). Work is paused pending a
plan review.

## What is here

| Module | What it does | In the KTP it becomes |
|---|---|---|
| `sld2quote/generate.py` | Draws synthetic LV switchboard SLDs with style variation and a scan-like variant, and writes ground truth (devices with boxes, busbars, connections, notes, a specification with deliberate conflicts) | The labelling tool over real sheets, plus a synthetic generator seeded with real topologies |
| `ocr/ocr.swift` | On-device OCR with Apple Vision (no cloud, no install) | The same, or a document OCR model, run inside the company's boundary |
| `sld2quote/reader.py` | Busbars and lines by run-length analysis, symbols by connected components and shape tests, labels by a second OCR pass around each symbol, notes by anchor | A trained symbol detector and layout model; the interface stays |
| `sld2quote/spec.py` | Clauses to requirements with the clause as evidence; drawing-versus-specification cross-check | An LLM extractor with a fixed schema and clause citations |
| `sld2quote/rules.py` + `catalogue.json` | Deterministic selection of parts within the verified envelope; issues raised, never silently fixed; evidence per line | R&B's real design rules and supplier catalogues |
| `sld2quote/quote.py` | `quotation.md`, `bom.csv`, `review.html` with boxes drawn on the sheet | The estimator's workbench |
| `sld2quote/evaluate.py` | Symbol precision/recall, label accuracy, notes, conflicts, BoM line precision/recall, price error, per sheet | The golden set and the regression gate |

## Run

```
python3 sld2quote/generate.py --out data --n 10 --seed 1 --scan
python3 -m sld2quote.cli run data/sheet-0001.png          # writes out/sheet-0001/{quotation.md,bom.csv,review.html,extraction.json}
python3 -m sld2quote.evaluate --data data --out eval      # writes eval/RESULTS.md
```
Requires macOS (Apple Vision for OCR), Python 3.12 with numpy, Pillow and scipy; `swiftc -O ocr/ocr.swift -o ocr/ocr -framework Vision -framework AppKit` rebuilds the OCR tool.

## Honesty notes
- Prices and part numbers are invented and labelled as such. The catalogue exists so the pipeline has somewhere to go.
- The reader is classical computer vision chosen for speed of demonstration; a trained detector is the KTP's first task.
- The specification parser is rule-based; an LLM with citations is the KTP's second task.
- Written with AI coding agents as implementation tools under the author's direction and tests. [CONFIRM wording with the author]
