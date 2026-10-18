# The live site's image. The Space builds it from the repo scripts/deploy_space.py assembles (this file at its root, the
# package, the site and its samples). CPU only; the detector weights download at start-up (site/app.py), never baked in.
FROM python:3.12.15-slim-trixie

# libgl1 and libglib2.0-0t64 (libglib2.0-0 before trixie): opencv-python, pulled in by rapidocr-onnxruntime, loads them.
# Fonts: generate.py and quote.py open macOS fonts by file name in /System/Library/Fonts/Supplemental, so those names point
# at the free metric-compatible Liberation faces; Verdana and Trebuchet MS have no such face and fall back to Arial, as
# generate.py does for any missing font. Seeds therefore draw the same layout as on the Mac in slightly different glyphs.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 fonts-liberation fonts-liberation-sans-narrow \
    && rm -rf /var/lib/apt/lists/* \
    && L=/usr/share/fonts/truetype/liberation M=/System/Library/Fonts/Supplemental && mkdir -p "$M" \
    && ln -s $L/LiberationSans-Regular.ttf "$M/Arial.ttf" && ln -s $L/LiberationSans-Bold.ttf "$M/Arial Bold.ttf" \
    && ln -s $L/LiberationSansNarrow-Regular.ttf "$M/Arial Narrow.ttf" && ln -s $L/LiberationMono-Regular.ttf "$M/Courier New.ttf"

# the extra index resolves torch==2.5.1 and torchvision==0.20.1 to their CPU-only builds (+cpu); PyPI's carry CUDA
COPY site/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

# Spaces run the container as uid 1000; /data holds the downloaded weights (SLD2QUOTE_WEIGHTS), a persistent volume if one is added
RUN useradd -m -u 1000 user && mkdir /data && chown user /data
USER user
WORKDIR /home/user/app
COPY --chown=user . .
# the English text recogniser (9 MB, public) into the image's hub cache so a cold start does not fetch it; this also
# proves the OCR stack loads on this image
RUN python -c "from sld2quote.ocr import _rapid; _rapid(True)"
# torch would otherwise size its thread pool from the host's cores, not the Space's 2 vCPUs
ENV OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
CMD ["uvicorn", "app:app", "--app-dir", "site", "--host", "0.0.0.0", "--port", "7860"]
