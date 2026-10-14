#!/bin/sh
set -e
cd "$(dirname "$0")"
[ -x ocr/ocr ] || swiftc -O ocr/ocr.swift -o ocr/ocr -framework Vision -framework AppKit
python3 sld2quote/generate.py --out data --n 5 --seed 1 --scan
python3 -m sld2quote.cli run data/sheet-0001.png
echo "open out/sheet-0001/review.html"
