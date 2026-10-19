#!/bin/sh
# Draws three synthetic sheets, quotes one and prints where the outputs are. Synthetic data only; prices are invented.
set -e
cd "$(dirname "$0")"
if [ "$(uname)" = Darwin ] && [ ! -x ocr/ocr ] && command -v swiftc >/dev/null; then
  swiftc -O ocr/ocr.swift -o ocr/ocr -framework Vision -framework AppKit  # Apple Vision; without it the reader uses RapidOCR
fi
python3 sld2quote/generate.py --out data/demo --n 3 --seed 1 --scan
W=weights/rtdetr_r18_plain_ema_fix_pass4  # the site's detector (README, The trained model)
if [ -f $W/config.json ]; then
  python3 -m sld2quote.cli run data/demo/sheet-0001.png --reader trained --weights $W --device cpu
else
  echo "note: no trained detector in $W (README, The trained model), so the classical reader runs; it finds far fewer symbols"
  python3 -m sld2quote.cli run data/demo/sheet-0001.png
fi
echo "outputs: out/sheet-0001/ (quotation.md, bom.csv, review.html, extraction.json); sheets and answer keys: data/demo/"
echo "open out/sheet-0001/review.html"
