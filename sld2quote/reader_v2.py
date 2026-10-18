"""Trained reader: the RT-DETR symbol detector finds the symbols, and every other step (page OCR, busbars and lines, OCR
around each symbol, label association, roles, notes) is the classical reader's pipeline, reader.read_sheet, so the two
readers differ only in the symbol finder and are compared like for like.

The detector reads the prepared (deskewed) grey page, so its boxes share one frame with the OCR and the line finders. The
page is cut by tiles.py as the training export cut it (640 px, a quarter overlap, white padding), the tiles go in batches
through train_detector.detect, the detector's one inference path, at the model's own score threshold and per-tile NMS from
its config.json; boxes move back to page pixels by their tile's origin and are merged across tiles by class-agnostic NMS, so
a symbol seen in two overlapping tiles is one symbol. The detector was trained on synthetic sheets drawn by generate.py.
"""
from __future__ import annotations
import functools, os
import torch
from PIL import Image
from . import reader
from .classes import CLASSES
from .tiles import cut_tile, tile_grid
from .train_detector import detect, load_model, load_processor, nms

WEIGHTS = reader.WEIGHTS  # weights/rtdetr_r18_full
TILE, OVERLAP = 640, 0.25  # as the training export cut the sheets (data/coco/manifest.json)
MERGE_IOU = 0.5            # boxes from overlapping tiles that overlap more than this are one symbol
BATCH = 16                 # tiles per forward pass


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@functools.cache
def _load(weights: str, device: str):
    if not os.path.isfile(os.path.join(weights, "config.json")):
        raise FileNotFoundError(f"{weights}: not a trained weights directory (no config.json); see sld2quote/train_detector.py")
    model, proc = load_model(weights).to(device).eval(), load_processor(weights)
    if device != "cpu":  # the first pass of a batch shape compiles kernels (about 10 s on mps): a full batch here, so no sheet's time carries it
        with torch.no_grad():
            detect(model, proc, proc(images=[Image.new("RGB", (TILE, TILE), "white")] * BATCH, return_tensors="pt")["pixel_values"].to(device), [(TILE, TILE)] * BATCH, 0.5, 0)
    return model, proc


def load(weights: str = WEIGHTS, device: str | None = None):
    """model and processor, loaded and warmed up once per process for each weights directory and device"""
    return _load(os.path.abspath(weights), device or default_device())


def merge(origins: list, results: list, iou: float = MERGE_IOU) -> dict:
    """per-tile detections (boxes x0, y0, x1, y1 in tile pixels) moved to page pixels by each tile's (x, y) origin, then
    class-agnostic NMS across tiles: of boxes that overlap by more than iou only the higher score survives"""
    boxes = [r["boxes"].cpu() + torch.tensor([x, y, x, y], dtype=r["boxes"].dtype) for (x, y), r in zip(origins, results)]
    return nms({"boxes": torch.cat(boxes), **{k: torch.cat([r[k].cpu() for r in results]) for k in ("scores", "labels")}}, iou)


@torch.no_grad()
def detect_page(page: Image.Image, model, proc, device: str, score_threshold: float, nms_iou: float) -> dict:
    """the detector on a whole page: its tiles through detect() in batches, merged into page pixels"""
    rgb = page.convert("RGB"); grid = tile_grid(*rgb.size, TILE, OVERLAP); results = []
    for i in range(0, len(grid), BATCH):
        tiles = [cut_tile(rgb, x, y, TILE) for x, y in grid[i:i + BATCH]]
        pv = proc(images=tiles, return_tensors="pt")["pixel_values"].to(device)
        results += detect(model, proc, pv, [(TILE, TILE)] * len(tiles), score_threshold, nms_iou)
    return merge(grid, results)


def to_symbols(r: dict) -> list[dict]:
    """merged detections as the extraction's symbols, in reading order (top to bottom, then left to right) as the classical
    finder gives them: the class name as kind, the box [x, y, w, h] in page pixels, the confidence as shape_score"""
    dets = sorted(zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()), key=lambda d: (round(d[0][1]), round(d[0][0])))
    return [{"id": f"S{i + 1}", "kind": CLASSES[c], "box": [round(x0), round(y0), round(x1) - round(x0), round(y1) - round(y0)], "shape_score": round(s, 2)}
            for i, ((x0, y0, x1, y1), s, c) in enumerate(dets)]


def read_sheet(image_path: str, texts: list[dict] | None = None, weights: str = WEIGHTS, device: str | None = None,
               score_threshold: float | None = None, nms_iou: float | None = None) -> dict:
    """the section 5 extraction with the detector's symbols; score_threshold and nms_iou default to the model's config.json"""
    device = device or default_device()
    model, proc = load(weights, device)
    st = getattr(model.config, "score_threshold", None) if score_threshold is None else score_threshold
    ni = getattr(model.config, "nms_iou", None) if nms_iou is None else nms_iou
    if st is None or ni is None:
        raise ValueError(f"{weights}: config.json has no score_threshold or nms_iou (trained before they were saved); pass both")
    ex = reader.read_sheet(image_path, texts, finder=lambda page, ink, texts: to_symbols(detect_page(page, model, proc, device, st, ni)), reader="trained")
    ex["detector"] = {"weights": weights, "device": device, "threads": torch.get_num_threads(), "score_threshold": st, "nms_iou": ni, "merge_iou": MERGE_IOU}
    return ex
