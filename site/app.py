"""The site's back end: precomputed samples, live runs (a sheet generated from a seed, or an uploaded drawing), their
files, the scorecards and the static front end.

  uvicorn app:app --app-dir site --host 0.0.0.0 --port 7860
  (not "site.app:app": the standard library's own `site` module shadows this directory as a package)

Environment: SLD2QUOTE_RUNS (run directories, default /tmp/sld2quote-runs, pruned to the last 50); SLD2QUOTE_WEIGHTS
(default weights/rtdetr_r18_full; when missing, downloaded once at start-up from the model repo SLD2QUOTE_MODEL_REPO
with HF_TOKEN); SLD2QUOTE_OCR (the OCR engine: vision on a Mac with the tool, else rapidocr); SLD2QUOTE_READER=classical
(tests only: the shape-test reader, no weights). The detector and the OCR engine warm up in a thread at start-up; live
runs answer 503 until they are ready, and run one at a time."""
from __future__ import annotations
import contextlib, io, json, os, re, secrets, shutil, sys, threading, time
from typing import Literal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sld2quote import generate, ocr, result  # noqa: E402

SAMPLES, STATIC = os.path.join(HERE, "samples"), os.path.join(HERE, "static")
RUNS = os.environ.get("SLD2QUOTE_RUNS", "/tmp/sld2quote-runs")
WEIGHTS = os.environ.get("SLD2QUOTE_WEIGHTS", os.path.join(ROOT, "weights", "rtdetr_r18_full"))
READER = os.environ.get("SLD2QUOTE_READER", "trained")
ENGINE = os.environ.get("SLD2QUOTE_OCR") or ocr.default_engine()
KEEP = 50                                                     # run directories kept
MAX_UPLOAD, MAX_SPEC, MAX_PIXELS = 12 << 20, 100 << 10, 12_000_000  # bytes, bytes, pixels (a 4000 x 3000 drawing)
SCORECARDS = {"results_md": "eval/RESULTS.md", "ocr_md": "eval/OCR.md", "detector_md": "eval/DETECTOR.md", "problems_md": "docs/PROBLEMS.md"}
ID, NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$"), re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
STATE = {"warm": False, "error": None}
LOCK = threading.Lock()  # ponytail: live runs are serial on the 2-core Space; a queue with positions if the demo gets busy


def ensure_weights():
    """the trained weights, downloaded once from SLD2QUOTE_MODEL_REPO when the directory is missing"""
    if READER != "trained" or os.path.isfile(os.path.join(WEIGHTS, "config.json")):
        return
    repo = os.environ.get("SLD2QUOTE_MODEL_REPO")
    if not repo:
        raise FileNotFoundError(f"{WEIGHTS}: no trained weights, and SLD2QUOTE_MODEL_REPO is not set")
    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=WEIGHTS, token=os.environ.get("HF_TOKEN"), allow_patterns=["*.json", "model.safetensors"])


def warm_up():
    """load the OCR engine and the detector once, so the first request does not pay for it"""
    try:
        ensure_weights()
        im = Image.new("L", (240, 48), 255); ImageDraw.Draw(im).text((8, 12), "ACB-1 2500 A 4P", fill=0)
        ocr.ocr_image(im, 2.0, "warm")
        if READER == "trained":
            from sld2quote import reader_v2
            model, proc = reader_v2.load(WEIGHTS, "cpu")
            reader_v2.detect_page(Image.new("L", (640, 640), 255), model, proc, "cpu", 0.5, 0.5)
        STATE["warm"] = True
    except Exception as e:  # reported by /api/health; live runs stay at 503
        STATE["error"] = f"{type(e).__name__}: {e}"


@contextlib.asynccontextmanager
async def lifespan(_):
    os.makedirs(RUNS, exist_ok=True)
    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(title="sld2quote", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def bad_input(_, e):
    return JSONResponse({"detail": jsonable_encoder(e.errors())}, status_code=400)


def check(value: str, rx: re.Pattern, what: str) -> str:
    if not rx.match(value):
        raise HTTPException(400, f"bad {what}: {value!r}")
    return value


def serve(path: str):
    if not os.path.isfile(path):
        raise HTTPException(404, "no such file")
    return FileResponse(path)


def mapped(res: dict, base: str) -> dict:
    """a Result with its file names turned into URLs under base"""
    return dict(res, images={k: f"{base}/{v}" for k, v in res["images"].items()}, files={k: f"{base}/{v}" for k, v in res["files"].items()})


# ---------------------------------------------------------------- health, samples, scorecards
@app.get("/api/health")
def health():
    return {"ok": STATE["error"] is None, "model": os.path.basename(WEIGHTS) if READER == "trained" else READER, "engine": ENGINE,
            "warm": STATE["warm"], **({"error": STATE["error"]} if STATE["error"] else {})}


@app.get("/api/samples")
def samples():
    p = os.path.join(SAMPLES, "index.json")
    items = json.load(open(p)) if os.path.exists(p) else []
    items = items.get("samples", []) if isinstance(items, dict) else items
    return {"samples": [dict(s, thumb=thumb_url(s)) for s in items]}


def thumb_url(s: dict) -> str:
    """the index's thumb (a path under site/samples, or under the sample's own directory) as a URL"""
    t = str(s.get("thumb") or "thumb.jpg")
    return t if t.startswith("/") else f"/api/samples/{s['id']}/files/{t[len(s['id']) + 1:] if t.startswith(s['id'] + '/') else t}"


@app.get("/api/samples/{sid}")
def sample(sid: str):
    p = os.path.join(SAMPLES, check(sid, ID, "sample id"), "result.json")
    if not os.path.isfile(p):
        raise HTTPException(404, f"no sample {sid!r}")
    return mapped(json.load(open(p)), f"/api/samples/{sid}/files")


@app.get("/api/samples/{sid}/files/{name}")
def sample_file(sid: str, name: str):
    return serve(os.path.join(SAMPLES, check(sid, ID, "sample id"), check(name, NAME, "file name")))


@app.get("/api/runs/{rid}/{name}")
def run_file(rid: str, name: str):
    return serve(os.path.join(RUNS, check(rid, ID, "run id"), check(name, NAME, "file name")))


@app.get("/api/scorecards")
def scorecards():
    return {k: open(os.path.join(ROOT, p), encoding="utf-8").read() if os.path.isfile(os.path.join(ROOT, p)) else "" for k, p in SCORECARDS.items()}


# ---------------------------------------------------------------- live runs
class Generate(BaseModel):
    seed: int = Field(ge=0, le=999999)
    level: Literal["simple", "medium", "mix"] = "mix"
    scan: bool = False


def new_run() -> tuple[str, str]:
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    d = os.path.join(RUNS, rid); os.makedirs(d)
    return rid, d


def prune():
    """keep the last KEEP run directories (their names sort by time)"""
    for name in sorted(os.listdir(RUNS))[:-KEEP]:
        shutil.rmtree(os.path.join(RUNS, name), ignore_errors=True)


def live(rid: str, d: str, image: str, spec: str | None, gt: str | None, source: dict) -> dict:
    with LOCK:
        try:
            res = result.build_result(image, spec, d, gt, READER, WEIGHTS, "cpu", source)
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(500, f"the run failed: {type(e).__name__}: {e}")
    prune()
    return mapped(res, f"/api/runs/{rid}")


def ready():
    if not STATE["warm"]:
        raise HTTPException(503, "the server is still loading the detector and the text reader; try again in a moment" if not STATE["error"] else f"the server could not load its models: {STATE['error']}")


@app.post("/api/generate")
def api_generate(g: Generate):
    ready()
    rid, d = new_run()
    stem = generate.write_sheet(d, g.seed, g.scan, None if g.level == "mix" else g.level)
    return live(rid, d, stem + ".png", stem + "-spec.txt", stem + ".json", {"kind": "generated", "title": f"Generated drawing, seed {g.seed}"})


@app.post("/api/upload")
def api_upload(file: UploadFile = File(...), spec: str | None = Form(None)):
    ready()
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(400, f"the image is over {MAX_UPLOAD >> 20} MB")
    try:
        im = Image.open(io.BytesIO(data)); fmt, (w, h) = im.format, im.size; im.verify()
    except Exception:
        raise HTTPException(400, "not a readable image: send a PNG or JPEG")
    if fmt not in ("PNG", "JPEG"):
        raise HTTPException(400, f"{fmt} is not accepted: send a PNG or JPEG")
    if w * h > MAX_PIXELS:
        raise HTTPException(400, f"the image is {w} x {h} pixels; the limit is {MAX_PIXELS // 1_000_000} megapixels")
    if spec and len(spec.encode("utf-8")) > MAX_SPEC:
        raise HTTPException(400, f"the specification is over {MAX_SPEC >> 10} kB")
    rid, d = new_run()
    img = os.path.join(d, "upload." + ("png" if fmt == "PNG" else "jpg"))
    open(img, "wb").write(data)
    sp = None
    if spec and spec.strip():
        sp = os.path.join(d, "spec.txt"); open(sp, "w", encoding="utf-8").write(spec)
    return live(rid, d, img, sp, None, {"kind": "upload", "title": (file.filename or "uploaded drawing")[:80]})


# ---------------------------------------------------------------- the front end
@app.get("/")
def index():
    p = os.path.join(STATIC, "index.html")
    return FileResponse(p) if os.path.isfile(p) else HTMLResponse("<p>sld2quote: the front end is not built yet; the API answers at /api/health.</p>")


@app.get("/static/{path:path}")
def static(path: str):
    p = os.path.realpath(os.path.join(STATIC, path))
    if not p.startswith(os.path.realpath(STATIC) + os.sep):
        raise HTTPException(400, "bad path")
    return serve(p)
