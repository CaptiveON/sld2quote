# OCR engines measured on our sheets

Written by `python3 -m sld2quote.evaluate_ocr --data data/baseline` from the records in `out/ocr_eval`; synthetic sheets only, nothing here is an R&B drawing. 60 sheets, a clean and a scan-like copy of each drawing; 10 of them (both copies) are standard 1600 x 1000 sheets, the rest are wider (medium boards and long rows of ways).

Scoring: each engine reads every page as saved (scan copies keep their tilt of up to 1.2 degrees; the ground-truth boxes are mapped into the image) through the reader's own OCR call, so the boxes and the look-alike letter map are what the reader gets. A recognised box matches a ground-truth text box one to one when IoU >= 0.3 or either centre lies inside the other. Strings are compared after collapsing whitespace. Character accuracy is 1 minus the edit distance over the longer string. Label strings are those with a number and A, kA, kVA, kW or kVAr (device, busbar and CT labels); notes strings are the notes lines with kA, Form, IP or -pole (the four the reader parses). In the label and notes columns a missed string scores 0, so they combine finding and reading. The crop pass reads a window around every ground-truth device as the reader does around every symbol; text cut by a window's edge is left out of that window's score.

| engine | measured | threads | model load s | load average, start and end | crop pass |
|---|---|---|---|---|---|
| vision | 2026-10-18 03:15 to 2026-10-18 03:22 | set by macOS, not capped (about 3.5 cores busy); not a Linux candidate | 0.16 | 6.5 / 8.4 | yes |
| rapidocr | 2026-10-18 03:38 to 2026-10-18 03:51 | 2: onnxruntime 2 intra-op and 1 inter-op, OpenCV 2 | 1.4 | 8.0 / 9.2 | yes |
| rapidocr-ch | 2026-10-18 03:55 to 2026-10-18 04:09 | 2: onnxruntime 2 intra-op and 1 inter-op, OpenCV 2 | 1.34 | 8.7 / 7.2 | yes |
| tesseract | 2026-10-18 03:51 to 2026-10-18 03:55 | 1 (this Homebrew build has no OpenMP) | 1.22 | 8.8 / 8.7 | yes |
| easyocr | 2026-10-18 04:58 to 2026-10-18 05:08 | 2: torch 2, OpenCV 2, CPU device | 4.26 | 8.7 / 9.2 | yes |
| doctr | 2026-10-18 04:09 to 2026-10-18 04:44 | 2: torch 2, OpenCV 2, CPU device | 5.99 | 7.4 / 12.6 | yes |

Run on the standard sheets only (`--max-width 1600`), to spare the CPU for the training run: easyocr (10 of the 60).

Machine: macOS-27.0.1-arm64-arm-64bit, 12 cores. CPU only: docTR and EasyOCR on torch's CPU device, RapidOCR on onnxruntime's CPU provider, and Vision's default path gave output bitwise identical to a build pinned to the CPU. Every candidate ran at 2 threads (OMP_NUM_THREADS=2, torch.set_num_threads(2), onnxruntime and OpenCV at 2), one engine at a time, while the detector's training run held the GPU and about four to five cores, so the times are a little pessimistic. A free Space's 2 shared vCPUs are slower than 2 M2 Max cores: check the times there.

## Page pass (2x, every text on the sheet)

| engine | copy | sheets | box recall | box precision | exact (matched) | char acc (matched) | label exact | label char | notes exact | notes char | s per sheet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vision | clean | 30 | 0.859 | 0.989 | 0.634 | 0.943 | 0.666 | 0.898 | 0.644 | 0.914 | 0.54 |
| vision | scan | 30 | 0.725 | 0.989 | 0.568 | 0.938 | 0.531 | 0.780 | 0.378 | 0.878 | 0.57 |
| rapidocr | clean | 30 | 0.962 | 0.991 | 0.729 | 0.968 | 0.878 | 0.989 | 0.833 | 0.995 | 3.90 |
| rapidocr | scan | 30 | 0.955 | 0.944 | 0.434 | 0.924 | 0.520 | 0.939 | 0.189 | 0.915 | 4.76 |
| rapidocr-ch | clean | 30 | 0.962 | 0.991 | 0.514 | 0.916 | 0.559 | 0.925 | 0.444 | 0.963 | 4.15 |
| rapidocr-ch | scan | 30 | 0.955 | 0.941 | 0.285 | 0.875 | 0.294 | 0.875 | 0.122 | 0.881 | 4.96 |
| tesseract | clean | 30 | 0.951 | 0.710 | 0.558 | 0.866 | 0.613 | 0.894 | 0.767 | 0.927 | 0.86 |
| tesseract | scan | 30 | 0.942 | 0.722 | 0.276 | 0.763 | 0.260 | 0.752 | 0.256 | 0.898 | 0.91 |
| easyocr | clean | 5 | 0.965 | 0.570 | 0.417 | 0.784 | 0.467 | 0.741 | 0.067 | 0.779 | 30.55 |
| easyocr | scan | 5 | 0.951 | 0.578 | 0.328 | 0.761 | 0.333 | 0.735 | 0.067 | 0.673 | 33.50 |
| doctr | clean | 30 | 0.829 | 0.936 | 0.260 | 0.776 | 0.202 | 0.722 | 0.044 | 0.768 | 4.30 |
| doctr | scan | 30 | 0.798 | 0.949 | 0.214 | 0.765 | 0.158 | 0.687 | 0.022 | 0.731 | 4.19 |

## Crop pass (3x, a window around every ground-truth device, same-line boxes merged)

| engine | copy | sheets | crops per sheet | box recall | box precision | exact (matched) | char acc (matched) | label exact | label char | s per sheet |
|---|---|---|---|---|---|---|---|---|---|---|
| vision | clean | 30 | 21.1 | 0.920 | 0.984 | 0.750 | 0.952 | 0.847 | 0.972 | 5.65 |
| vision | scan | 30 | 21.1 | 0.885 | 0.986 | 0.601 | 0.936 | 0.633 | 0.933 | 5.87 |
| rapidocr | clean | 30 | 21.1 | 0.921 | 0.974 | 0.667 | 0.942 | 0.740 | 0.948 | 9.18 |
| rapidocr | scan | 30 | 21.1 | 0.903 | 0.961 | 0.408 | 0.891 | 0.334 | 0.879 | 9.10 |
| rapidocr-ch | clean | 30 | 21.1 | 0.922 | 0.976 | 0.339 | 0.850 | 0.168 | 0.806 | 9.83 |
| rapidocr-ch | scan | 30 | 21.1 | 0.902 | 0.957 | 0.257 | 0.836 | 0.078 | 0.786 | 10.07 |
| tesseract | clean | 30 | 21.1 | 0.870 | 0.899 | 0.528 | 0.843 | 0.617 | 0.907 | 2.77 |
| tesseract | scan | 30 | 21.1 | 0.824 | 0.857 | 0.224 | 0.710 | 0.238 | 0.733 | 2.80 |
| easyocr | clean | 5 | 12.6 | 0.948 | 0.993 | 0.673 | 0.917 | 0.613 | 0.900 | 23.43 |
| easyocr | scan | 5 | 12.6 | 0.890 | 0.939 | 0.493 | 0.858 | 0.412 | 0.851 | 24.60 |
| doctr | clean | 30 | 21.1 | 0.938 | 0.908 | 0.494 | 0.890 | 0.462 | 0.924 | 30.95 |
| doctr | scan | 30 | 21.1 | 0.914 | 0.943 | 0.389 | 0.862 | 0.347 | 0.889 | 30.22 |

## By font (page pass; label char, then notes char; clean / scan)

| engine | Arial | Arial Narrow | Courier New | Trebuchet MS | Verdana |
|---|---|---|---|---|---|
| vision | 0.96 / 0.96; 0.99 / 0.99 | 0.91 / 0.84; 0.98 / 0.89 | 0.72 / 0.49; 0.54 / 0.60 | 0.94 / 0.72; 0.99 / 0.88 | 0.92 / 0.84; 0.90 / 0.98 |
| rapidocr | 1.00 / 0.96; 1.00 / 0.97 | 0.99 / 0.95; 1.00 / 0.97 | 0.98 / 0.87; 0.98 / 0.57 | 0.98 / 0.95; 1.00 / 0.97 | 0.99 / 0.96; 0.99 / 0.96 |
| rapidocr-ch | 0.94 / 0.89; 0.98 / 0.94 | 0.94 / 0.90; 0.99 / 0.94 | 0.87 / 0.80; 0.95 / 0.55 | 0.92 / 0.87; 0.93 / 0.92 | 0.92 / 0.88; 0.95 / 0.92 |
| tesseract | 0.97 / 0.91; 1.00 / 0.96 | 0.95 / 0.73; 1.00 / 0.94 | 0.56 / 0.51; 0.47 / 0.53 | 0.95 / 0.81; 1.00 / 0.96 | 0.93 / 0.78; 1.00 / 0.97 |
| easyocr | 0.93 / 0.91; 0.98 / 0.89 | 0.87 / 0.80; 0.97 / 0.91 | 0.31 / 0.36; 0.14 / 0.16 | 0.87 / 0.86; 0.90 / 0.70 | n/a / n/a; n/a / n/a |
| doctr | 0.77 / 0.78; 0.81 / 0.82 | 0.69 / 0.68; 0.68 / 0.65 | 0.74 / 0.59; 0.83 / 0.76 | 0.72 / 0.68; 0.84 / 0.78 | 0.76 / 0.73; 0.72 / 0.71 |

Sheets per font (each clean and scan): Arial 5, Arial Narrow 10, Courier New 4, Trebuchet MS 7, Verdana 4.

## The rule

Best character accuracy on the label and notes strings (the mean of the two groups, both read in the page pass, clean and scan pooled), among the engines that read a standard sheet (1600 x 1000) in under about 5 s on 2 CPU threads; if two are close (within 0.01), the lighter install wins. Apple Vision is the Mac engine and the reference, not a Linux candidate.

| engine | sheets | label char | notes char | label and notes | label char, crop pass | s per standard sheet, page | s per standard sheet, crops | within the time rule | install on Linux |
|---|---|---|---|---|---|---|---|---|---|
| vision | 60 | 0.839 | 0.896 | 0.868 | 0.953 | 0.46 | 3.45 | yes | macOS only (Apple Vision), not available on Linux |
| rapidocr | 60 | 0.964 | 0.955 | 0.960 | 0.914 | 2.51 | 4.75 | yes | pip rapidocr-onnxruntime (onnxruntime, OpenCV, 15 MB of bundled models), a 9 MB English recogniser fetched once, apt libgl1 for OpenCV |
| rapidocr-ch | 60 | 0.900 | 0.922 | 0.911 | 0.796 | 2.76 | 4.98 | yes | pip rapidocr-onnxruntime (onnxruntime, OpenCV, 15 MB of bundled models), apt libgl1 for OpenCV |
| tesseract | 60 | 0.823 | 0.913 | 0.868 | 0.820 | 0.59 | 1.48 | yes | apt tesseract-ocr (about 30 MB with the English model), pip pytesseract |
| easyocr | 10 | 0.738 | 0.726 | 0.732 | 0.876 | 32.03 | 24.01 | no | pip easyocr (torch, OpenCV, scikit-image), 94 MB of models fetched once |
| doctr | 60 | 0.705 | 0.749 | 0.727 | 0.907 | 4.23 | 16.73 | yes | pip python-doctr (torch, OpenCV, onnx, h5py and more), 124 MB of models fetched once |

## The choice

**rapidocr**: label and notes accuracy 0.960 (labels 0.964, notes 0.955) in 2.51 s per standard sheet. Apple Vision on the same pages: 0.868 (labels 0.839, notes 0.896) in 0.46 s.

Ranked under the rule: rapidocr 0.960 in 2.51 s, rapidocr-ch 0.911 in 2.76 s, tesseract 0.868 in 0.59 s, doctr 0.727 in 4.23 s.

Over the time rule: easyocr (32.0 s per standard sheet, accuracy 0.732).

For the reader: rapidocr reads labels at 0.964 from the 2x page and 0.914 from the 3x crops (crops add 4.8 s per standard sheet); Vision reads them at 0.839 from the page and 0.953 from the crops, which is why the reader crops today.

## On a Hugging Face Space (Linux, 2 shared vCPUs)

- `requirements.txt`: `rapidocr-onnxruntime==1.4.4`, `onnxruntime==1.31.0` and `opencv-python==5.0.0.93`, the versions measured (rapidocr-onnxruntime also pulls pyclipper, shapely, PyYAML, six and tqdm); `huggingface_hub` is already there for the detector.
- `packages.txt`: `libgl1` and `libglib2.0-0`, because rapidocr-onnxruntime requires opencv-python by name and its cv2 loads libGL.
- Models: the detector and the direction classifier ship inside the wheel (15 MB). The English recogniser (9 MB) is fetched on first use from the Hugging Face hub (`SWHL/RapidOCR`, `PP-OCRv3/en_PP-OCRv3_rec_infer.onnx`, pinned revision) into the hub cache (`~/.cache/huggingface/hub`, or `$HF_HOME/hub`); warm the engine once when the app starts (one `ocr.ocr_image` call on a small blank image loads it and fetches the recogniser), so no request waits.
- Engine: leave `SLD2QUOTE_OCR` unset (with no Vision tool on Linux the default is rapidocr) or set it to `rapidocr`. `ocr.THREADS` is 2, the free Space's vCPUs.
