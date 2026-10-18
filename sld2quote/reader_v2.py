"""Trained reader: the RT-DETR symbol detector finds the symbols, and every other step (page OCR, busbars and lines, OCR
around each symbol, label association, roles, notes) is the classical reader's pipeline, reader.read_sheet, so the two
readers differ only in the symbol finder and are compared like for like.

The detector reads the prepared (deskewed) grey page, so its boxes share one frame with the OCR and the line finders. The
page is cut by tiles.py as the training export cut it (640 px, a quarter overlap, white padding), the tiles go in batches
through train_detector.detect, the detector's one inference path, at the model's own score threshold and per-tile NMS from
its config.json. The merge into page boxes has three settings, all recorded in the extraction's detector provenance: a box
within EDGE_PX of an inner tile border (one that is not the page edge) is dropped, since it is a symbol cut by the tile and
the export's quarter overlap puts every symbol whole in some tile; the rest move back to page pixels by their tile's origin
and are merged across tiles by class-agnostic NMS at MERGE_IOU, so a symbol seen in two overlapping tiles is one symbol;
then a box holding CONTAIN or more of a better-scoring box's area is dropped, such as a tall box spanning two stacked
symbols. The detector was trained on synthetic sheets drawn by generate.py.
"""
from __future__ import annotations
import functools, os
import torch
from PIL import Image
from . import reader
from .classes import CLASSES
from .tiles import cut_tile, tile_grid
from .train_detector import EDGE_PX, detect, load_model, load_processor, nms

WEIGHTS = reader.WEIGHTS  # weights/rtdetr_r18_full
TILE, OVERLAP = 640, 0.25  # as the training export cut the sheets (data/coco/manifest.json)
MERGE_IOU = 0.5            # boxes from overlapping tiles that overlap more than this are one symbol
CONTAIN = 0.8              # a box holding this share of a better box's area or more is a duplicate of it
# EDGE_PX (train_detector's, 2 px): a box this close to an inner tile border is cut by it
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


def drop_cut(origins: list, results: list, size: tuple, edge_px: float = EDGE_PX) -> list:
    """per tile, the detections clear of its inner borders: a box within edge_px of a border that is not the page edge is a
    symbol cut by the tile, and the export's quarter overlap puts that symbol whole in a neighbouring tile"""
    W, H = size; out = []
    for (x, y), r in zip(origins, results):
        b = r["boxes"].cpu()
        cut = (((b[:, 0] <= edge_px) & (x > 0)) | ((b[:, 1] <= edge_px) & (y > 0)) |
               ((b[:, 2] >= TILE - edge_px) & (x + TILE < W)) | ((b[:, 3] >= TILE - edge_px) & (y + TILE < H)))
        out.append({k: v.cpu()[~cut] for k, v in r.items()})
    return out


def drop_containers(r: dict, contain: float = CONTAIN) -> dict:
    """best score first, a box survives unless it holds contain or more of the area of a better box already kept (the
    better box being the smaller): a box spanning two stacked symbols goes, partial overlaps stay"""
    b = r["boxes"]; area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    inter = (torch.min(b[:, None, 2:], b[None, :, 2:]) - torch.max(b[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    holds = (inter > 0) & (inter >= contain * area[None, :]) & (area[None, :] <= area[:, None])  # [i, j]: box i holds box j
    keep = []
    for i in r["scores"].argsort(descending=True).tolist():
        if not any(holds[i, j] for j in keep):
            keep.append(i)
    return {k: v[torch.tensor(keep, dtype=torch.long)] for k, v in r.items()}


def merge(origins: list, results: list, size: tuple, iou: float = MERGE_IOU, edge_px: float | None = EDGE_PX, contain: float | None = CONTAIN) -> dict:
    """per-tile detections (boxes x0, y0, x1, y1 in tile pixels, tiles at their (x, y) origins on a page of size (W, H)) to
    page boxes: those cut by an inner tile border dropped, the rest moved by their tile's origin, class-agnostic NMS across
    tiles (of boxes overlapping by more than iou only the higher score survives), then boxes holding a better one dropped;
    edge_px or contain None turns that filter off"""
    if edge_px is not None:
        results = drop_cut(origins, results, size, edge_px)
    boxes = [r["boxes"].cpu() + torch.tensor([x, y, x, y], dtype=r["boxes"].dtype) for (x, y), r in zip(origins, results)]
    r = nms({"boxes": torch.cat(boxes), **{k: torch.cat([r[k].cpu() for r in results]) for k in ("scores", "labels")}}, iou)
    return drop_containers(r, contain) if contain is not None else r


@torch.no_grad()
def tile_detections(page: Image.Image, model, proc, device: str, score_threshold: float, nms_iou: float) -> tuple:
    """the detector on every tile of a page, in batches through detect(): the tile origins, each tile's detections, the page size"""
    rgb = page.convert("RGB"); grid = tile_grid(*rgb.size, TILE, OVERLAP); results = []
    for i in range(0, len(grid), BATCH):
        tiles = [cut_tile(rgb, x, y, TILE) for x, y in grid[i:i + BATCH]]
        pv = proc(images=tiles, return_tensors="pt")["pixel_values"].to(device)
        results += detect(model, proc, pv, [(TILE, TILE)] * len(tiles), score_threshold, nms_iou)
    return grid, results, rgb.size


def detect_page(page: Image.Image, model, proc, device: str, score_threshold: float, nms_iou: float) -> dict:
    """the detector on a whole page, merged into page pixels"""
    return merge(*tile_detections(page, model, proc, device, score_threshold, nms_iou))


def to_symbols(r: dict) -> list[dict]:
    """merged detections as the extraction's symbols, in reading order (top to bottom, then left to right) as the classical
    finder gives them: the class name as kind, the box [x, y, w, h] in page pixels, the confidence as shape_score"""
    dets = sorted(zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()), key=lambda d: (round(d[0][1]), round(d[0][0])))
    return [{"id": f"S{i + 1}", "kind": CLASSES[c], "box": [round(x0), round(y0), round(x1) - round(x0), round(y1) - round(y0)], "shape_score": round(s, 2)}
            for i, ((x0, y0, x1, y1), s, c) in enumerate(dets)]


def threshold(config, given: float | None = None) -> tuple:
    """the operating score threshold and its source: the caller's, else the one chosen on whole validation drawings
    (select_threshold.py writes score_threshold_drawing), else the per-tile one chosen at training"""
    for value, source in ((given, "argument"), (getattr(config, "score_threshold_drawing", None), "drawing"), (getattr(config, "score_threshold", None), "tile")):
        if value is not None:
            return value, source
    return None, None


def read_sheet(image_path: str, texts: list[dict] | None = None, weights: str = WEIGHTS, device: str | None = None,
               score_threshold: float | None = None, nms_iou: float | None = None, crops: bool | None = None) -> dict:
    """the section 5 extraction with the detector's symbols; score_threshold defaults to threshold()'s choice, nms_iou to
    the model's config.json and crops (the OCR pass around every symbol) to the OCR engine's default (reader.read_sheet)"""
    device = device or default_device()
    model, proc = load(weights, device)
    st, source = threshold(model.config, score_threshold)
    ni = getattr(model.config, "nms_iou", None) if nms_iou is None else nms_iou
    if st is None or ni is None:
        raise ValueError(f"{weights}: config.json has no score_threshold or nms_iou (trained before they were saved); pass both")
    ex = reader.read_sheet(image_path, texts, finder=lambda page, ink, texts: to_symbols(detect_page(page, model, proc, device, st, ni)), reader="trained", crops=crops)
    ex["detector"] = {"weights": weights, "device": device, "threads": torch.get_num_threads(), "score_threshold": st, "score_threshold_source": source, "nms_iou": ni, "merge_iou": MERGE_IOU,
                     "edge_px": EDGE_PX, "contain": CONTAIN}
    return ex
