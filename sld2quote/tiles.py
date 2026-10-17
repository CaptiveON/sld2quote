"""Page tiling for the symbol detector, shared by the COCO export and the trained reader's inference so that both cut a sheet
the same way: square tiles (640 px by default) overlapping by a quarter, the last tile on each axis flush with the page edge,
and a symbol's box kept in a tile only when most of the symbol lies inside it."""
from __future__ import annotations
from PIL import Image


def tile_starts(length: int, tile: int = 640, overlap: float = 0.25) -> list[int]:
    """tile origins along one axis: every stride from 0 while a tile still ends short of the edge, then one flush with the
    edge, so every pixel is covered and neighbours share at least `overlap` of a tile (the stride is rounded to whole pixels);
    a length <= tile gives [0]"""
    return sorted({*range(0, length - tile, round(tile * (1 - overlap))), max(0, length - tile)})


def tile_grid(width: int, height: int, tile: int = 640, overlap: float = 0.25) -> list[tuple[int, int]]:
    """(x, y) origin of every tile on a page, row by row"""
    return [(x, y) for y in tile_starts(height, tile, overlap) for x in tile_starts(width, tile, overlap)]


def clip_box(box: list, x0: int, y0: int, tile: int, min_visible: float = 0.6) -> tuple[list, float] | None:
    """a page box [x, y, w, h] cut to the tile at (x0, y0), in tile pixels, with the share of its area inside the tile;
    None below min_visible, so a sliver of a symbol at a tile edge is not taught as a whole one"""
    x, y, w, h = box
    ix0, iy0, ix1, iy1 = max(x, x0), max(y, y0), min(x + w, x0 + tile), min(y + h, y0 + tile)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    frac = (ix1 - ix0) * (iy1 - iy0) / (w * h)
    return ([ix0 - x0, iy0 - y0, ix1 - ix0, iy1 - iy0], frac) if frac >= min_visible else None


def cut_tile(im: Image.Image, x: int, y: int, tile: int = 640) -> Image.Image:
    """the tile x tile image at (x, y); where the page is shorter than a tile, the rest is white paper"""
    t = Image.new(im.mode, (tile, tile), "white")
    t.paste(im.crop((x, y, min(x + tile, im.width), min(y + tile, im.height))))
    return t
