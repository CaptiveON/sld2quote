"""OCR behind one function: ocr_image(img, scale, tag) -> [{text, conf, box, src}], boxes in the pixels of the image passed in.

"vision" is Apple Vision through ocr/ocr.swift (macOS only, on-device). "rapidocr" is the engine that also runs on Linux,
chosen by measuring the candidates on our own sheets (eval/OCR.md): the PP-OCRv4 text detector with the English PP-OCRv3
recogniser on onnxruntime, CPU only. The other candidates measured stay here so the comparison can be rerun:
"rapidocr-ch" (the package's bundled Chinese recogniser, which drops the spaces in English labels), "tesseract",
"easyocr" and "doctr". SLD2QUOTE_OCR or the engine argument picks one; the default is vision when its tool can run, else
rapidocr. Each engine loads once per process, on first use. The reader maps look-alike letters (reader._ocr_image).
"""
from __future__ import annotations
import functools, json, os, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image

OCR_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ocr", "ocr")
THREADS = 2  # the free Space has 2 vCPUs; left alone, onnxruntime and torch size their pools from every core they can see
# the English recogniser (9 MB, Apache-2.0), fetched once into the Hugging Face cache at a pinned revision
EN_REC = ("SWHL/RapidOCR", "PP-OCRv3/en_PP-OCRv3_rec_infer.onnx", "1cfba2e90fc938db55889873735088de210cc173")


def ocr_image(img: Image.Image, scale: float, tag: str, engine: str | None = None) -> list[dict]:
    """OCR a PIL image at an upscale factor (small CAD text reads far better enlarged); boxes come back in the pixels of
    the image passed in, before any offset the caller adds"""
    if scale != 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    name = engine or os.environ.get("SLD2QUOTE_OCR") or default_engine()
    if name not in ENGINES:
        raise ValueError(f"unknown OCR engine {name!r} (SLD2QUOTE_OCR or the engine argument): one of {', '.join(ENGINES)}")
    run = ENGINES[name]
    return [{"text": t, "conf": float(c), "box": [x / scale, y / scale, w / scale, h / scale], "src": tag} for t, c, x, y, w, h in run(img)]


@functools.cache
def default_engine() -> str:
    return "vision" if vision_tool() else "rapidocr"


@functools.cache
def vision_tool() -> str | None:
    """the Vision tool on macOS, built from ocr/ocr.swift if it is missing; None where it cannot run"""
    if sys.platform != "darwin":
        return None
    if not os.path.exists(OCR_BIN) and shutil.which("swiftc"):
        subprocess.run(["swiftc", "-O", OCR_BIN + ".swift", "-o", OCR_BIN, "-framework", "Vision", "-framework", "AppKit"], check=False)
    return OCR_BIN if os.path.exists(OCR_BIN) else None


def _bbox(pts) -> tuple:
    xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _vision(img: Image.Image) -> list[tuple]:
    """the Swift tool on a temporary PNG, as the reader has always called it"""
    tool = vision_tool()
    if not tool:
        raise RuntimeError("the vision engine needs macOS and ocr/ocr (swiftc -O ocr/ocr.swift -o ocr/ocr -framework Vision -framework AppKit)")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        tmp = fh.name
    try:
        img.save(tmp)
        out = subprocess.run([tool, tmp], capture_output=True, text=True, timeout=180)
    finally:
        os.unlink(tmp)
    if out.returncode != 0:
        raise RuntimeError("ocr failed: " + out.stderr)
    return [(b["text"], b["conf"], b["x"], b["y"], b["w"], b["h"]) for b in json.loads(out.stdout)]


@functools.cache
def _rapid(english: bool):
    import cv2
    from rapidocr_onnxruntime import RapidOCR
    cv2.setNumThreads(THREADS)
    kw = {}
    if english:
        from huggingface_hub import hf_hub_download
        kw["rec_model_path"] = hf_hub_download(EN_REC[0], EN_REC[1], revision=EN_REC[2])
    # max_side_len: the package shrinks any image over 2000 px, which would undo the caller's upscale
    return RapidOCR(max_side_len=100_000, intra_op_num_threads=THREADS, inter_op_num_threads=1, **kw)


def _rapidocr(img: Image.Image, english: bool = True) -> list[tuple]:
    res, _ = _rapid(english)(img)
    return [(t, c, *_bbox(pts)) for pts, t, c in res or []]


def _tesseract(img: Image.Image) -> list[tuple]:
    """words joined into Tesseract's own lines; the sparse-text mode suits labels scattered over a drawing"""
    import pytesseract
    os.environ.setdefault("OMP_THREAD_LIMIT", str(THREADS))
    d = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    lines = {}
    for i, t in enumerate(d["text"]):
        if t.strip():
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
    out = []
    for ix in lines.values():
        x0, y0 = min(d["left"][i] for i in ix), min(d["top"][i] for i in ix)
        x1, y1 = max(d["left"][i] + d["width"][i] for i in ix), max(d["top"][i] + d["height"][i] for i in ix)
        out.append((" ".join(d["text"][i] for i in ix), max(0.0, min(float(d["conf"][i]) for i in ix)) / 100, x0, y0, x1 - x0, y1 - y0))
    return out


@functools.cache
def _easy():
    import cv2, easyocr, torch
    cv2.setNumThreads(THREADS); torch.set_num_threads(THREADS)
    return easyocr.Reader(["en"], gpu=False, verbose=False)  # gpu=False: CPU, never CUDA or MPS


def _easyocr(img: Image.Image) -> list[tuple]:
    a = np.asarray(img.convert("L"))
    # canvas_size: EasyOCR shrinks any image over 2560 px, which would undo the caller's upscale
    return [(t, c, *_bbox(p)) for p, t, c in _easy().readtext(a, canvas_size=max(a.shape))]


@functools.cache
def _doctr_model():
    import cv2, torch
    from doctr.models import ocr_predictor
    cv2.setNumThreads(THREADS); torch.set_num_threads(THREADS)
    return ocr_predictor(pretrained=True).to("cpu")  # fast_base detector (fixed 1024 px input), crnn_vgg16_bn recogniser; CPU only


def _doctr(img: Image.Image) -> list[tuple]:
    W, H = img.size
    lines = [l for b in _doctr_model()([np.asarray(img.convert("RGB"))]).pages[0].blocks for l in b.lines]
    return [(" ".join(w.value for w in l.words), min(w.confidence for w in l.words), l.geometry[0][0] * W, l.geometry[0][1] * H,
             (l.geometry[1][0] - l.geometry[0][0]) * W, (l.geometry[1][1] - l.geometry[0][1]) * H) for l in lines]


ENGINES = {"vision": _vision, "rapidocr": _rapidocr, "rapidocr-ch": functools.partial(_rapidocr, english=False),
           "tesseract": _tesseract, "easyocr": _easyocr, "doctr": _doctr}
