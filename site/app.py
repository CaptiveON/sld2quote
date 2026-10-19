"""The site's back end: precomputed samples, live runs (a sheet generated from a seed, or an uploaded drawing), their
files, the scorecards and the chart data (eval/charts.json from scripts/build_charts.py), and the static front end. POST
/api/generate and /api/upload with ?stream=1 answer NDJSON: a start
line, one line as each of the five steps finishes, then the Result (or one error line).

  uvicorn app:app --app-dir site --host 0.0.0.0 --port 7860
  (not "site.app:app": the standard library's own `site` module shadows this directory as a package)

Environment: SLD2QUOTE_RUNS (run directories, default /tmp/sld2quote-runs, pruned to the last 50); SLD2QUOTE_WEIGHTS
(default weights/rtdetr_r18_full; when missing, downloaded once at start-up from the model repo SLD2QUOTE_MODEL_REPO
with HF_TOKEN); SLD2QUOTE_OCR (the OCR engine: vision on a Mac with the tool, else rapidocr); SLD2QUOTE_READER=classical
(tests only: the shape-test reader, no weights); SLD2QUOTE_CROPS=0 or 1 (the OCR pass around every symbol; default: the
engine's own, on for vision, off for rapidocr); SLD2QUOTE_DAILY_RUNS (live runs allowed per UTC day per instance, default 16, so two instances stay inside the free tier in the strict worst case; over it
live runs answer 429, samples and scorecards never). The detector and the OCR engine load in the first request that needs
them (/api/health or a live run; the caller waits about 20 s, other callers wait for its lock), since a background thread
gets no CPU under per-request billing; after a load error /api/health and live runs answer 503 for good. Live runs go one
at a time."""
from __future__ import annotations
import contextlib, functools, io, json, os, queue, re, secrets, shutil, sys, threading, time
from typing import Literal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
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
CROPS = {"0": False, "1": True}.get(os.environ.get("SLD2QUOTE_CROPS", ""))  # None: the engine's default
KEEP = 50                                                     # run directories kept
MAX_UPLOAD, MAX_SPEC, MAX_PIXELS = 12 << 20, 100 << 10, 12_000_000  # bytes, bytes, pixels (a 4000 x 3000 drawing)
SCORECARDS = {"results_md": "eval/RESULTS.md", "ocr_md": "eval/OCR.md", "detector_md": "eval/DETECTOR.md", "problems_md": "docs/PROBLEMS.md"}
CHARTS = os.path.join(ROOT, "eval", "charts.json")
ID, NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$"), re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
STATE = {"warm": False, "error": None}
LOCK = threading.Lock()  # ponytail: live runs are serial on the 2-core Space; a queue with positions if the demo gets busy
WARMING = threading.Lock()  # the one warm-up; concurrent requests wait for it
DAILY_RUNS = int(os.environ.get("SLD2QUOTE_DAILY_RUNS", "16"))
TODAY = {"date": "", "runs": 0}  # ponytail: in memory, so the count resets when the instance restarts; a bucket or Firestore counter if it ever matters
OVER = "today's free allowance of live runs is used up; the samples still work; try again tomorrow"
STEPS = [{"step": "text", "label": "Reading the words on the page"}, {"step": "symbols", "label": "Finding the symbols"},
         {"step": "labels", "label": "Pairing labels with symbols"}, {"step": "spec", "label": "Checking the specification"},
         {"step": "price", "label": "Choosing parts and pricing"}]
NDJSON = "application/x-ndjson"


def ensure_weights():
    """the trained weights, downloaded once from SLD2QUOTE_MODEL_REPO when the directory is missing"""
    if READER != "trained" or os.path.isfile(os.path.join(WEIGHTS, "config.json")):
        return
    repo = os.environ.get("SLD2QUOTE_MODEL_REPO")
    if not repo:
        raise FileNotFoundError(f"{WEIGHTS}: no trained weights, and SLD2QUOTE_MODEL_REPO is not set")
    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=WEIGHTS, token=os.environ.get("HF_TOKEN"), allow_patterns=["*.json", "model.safetensors"])


def warm():
    """the models loaded once, in the first request that needs them (a thread would get no CPU between requests on Cloud
    Run); a caller arriving meanwhile waits for the lock; after a load error nothing is retried and every caller sees it"""
    if not (STATE["warm"] or STATE["error"]):
        with WARMING:
            if not (STATE["warm"] or STATE["error"]):
                warm_up()


def warm_up():
    """load the OCR engine and the detector: STATE warm, or error with the reason"""
    try:
        ensure_weights()
        im = Image.new("L", (240, 48), 255); ImageDraw.Draw(im).text((8, 12), "ACB-1 2500 A 4P", fill=0)
        ocr.ocr_image(im, 2.0, "warm")
        if READER == "trained":
            from sld2quote import reader_v2
            model, proc = reader_v2.load(WEIGHTS, "cpu")
            reader_v2.detect_page(Image.new("L", (640, 640), 255), model, proc, "cpu", 0.5, 0.5)
        STATE["warm"] = True
    except Exception as e:  # /api/health and live runs answer 503 with it
        STATE["error"] = f"{type(e).__name__}: {e}"


@contextlib.asynccontextmanager
async def lifespan(_):
    os.makedirs(RUNS, exist_ok=True)
    cards()
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
    """warms the models on the first call (the caller waits, about 20 s), then 200 warm; 503 with the error after a load error"""
    warm()
    body = {"ok": STATE["error"] is None, "model": os.path.basename(WEIGHTS) if READER == "trained" else READER, "engine": ENGINE,
            "warm": STATE["warm"], **({"error": STATE["error"]} if STATE["error"] else {})}
    return JSONResponse(body, status_code=200 if STATE["warm"] else 503)


@app.get("/api/samples")
def samples():
    return {"samples": cards()}


@functools.cache
def cards() -> list[dict]:
    """the gallery cards, read once at start-up: index.json gives the order and titles, each sample's result.json the rest"""
    p = os.path.join(SAMPLES, "index.json")
    items = json.load(open(p)) if os.path.exists(p) else []
    out = []
    for s in (items.get("samples", []) if isinstance(items, dict) else items):
        rp = os.path.join(SAMPLES, s["id"], "result.json")
        r = json.load(open(rp)) if os.path.isfile(rp) else {}
        src, sc = r.get("source", {}), r.get("score")
        out.append({"id": s["id"], "title": s.get("title") or src.get("title", s["id"]), "kind": src.get("kind", s.get("kind")), "copy": src.get("copy", s.get("copy")),
                    "has_answer_key": src.get("has_answer_key", s.get("has_answer_key", False)), "note": src.get("note", s.get("note", "")),
                    "devices": len(r.get("devices", [])), "total_gbp": r.get("bom", {}).get("total_gbp"), "issues": len(r.get("issues", [])),
                    "score": sc and {k: sc.get(k) for k in ("precision", "recall", "price_error", "rating_accuracy")},
                    "seconds": r.get("timing", {}).get("total"), "thumb": thumb_url(s)})
    return out


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


@app.get("/api/charts")
def charts():
    """the chart data, eval/charts.json as scripts/build_charts.py wrote it"""
    if not os.path.isfile(CHARTS):
        raise HTTPException(404, "no chart data: run python3 scripts/build_charts.py")
    return FileResponse(CHARTS, media_type="application/json")


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


def runs_today() -> int:
    """live runs counted so far on this UTC day"""
    day = time.strftime("%Y-%m-%d", time.gmtime())
    if TODAY["date"] != day:
        TODAY.update(date=day, runs=0)
    return TODAY["runs"]


def live(rid: str, d: str, image: str, spec: str | None, gt: str | None, source: dict, progress=None) -> dict:
    with LOCK:
        if runs_today() >= DAILY_RUNS:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(429, OVER)
        TODAY["runs"] += 1
        try:
            res = result.build_result(image, spec, d, gt, READER, WEIGHTS, "cpu", source, CROPS, progress=progress)
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(500, f"the run failed: {type(e).__name__}: {e}")
    prune()
    return mapped(res, f"/api/runs/{rid}")


def ready():
    """a live run on a cold instance warms the models first; 503 only after a load error; 429 over the daily cap"""
    warm()
    if STATE["error"]:
        raise HTTPException(503, f"the server could not load its models: {STATE['error']}")
    if runs_today() >= DAILY_RUNS:  # before any work; live() counts under the lock
        raise HTTPException(429, OVER)


def line(obj: dict) -> str:
    return json.dumps(obj) + "\n"


def stream_run(prepare):
    """a live run as NDJSON: prepare() checks the request and sets the run up, giving (rid, run) with run(progress) -> Result;
    the run goes in a worker thread (serial, under LOCK, in live()) and the stream carries the start line, one line per step
    as it finishes, then the Result or one error line; a refusal in prepare() (400, 429, 503) is the stream's only line"""
    try:
        rid, run = prepare()
    except HTTPException as e:
        return StreamingResponse(iter([line({"event": "error", "status": e.status_code, "message": e.detail})]), media_type=NDJSON)
    q = queue.Queue()

    def work():
        try:
            q.put({"event": "result", "result": run(lambda step, secs: q.put({"event": "step", "step": step, "seconds": secs}))})
        except Exception as e:
            status, msg = (e.status_code, e.detail) if isinstance(e, HTTPException) else (500, f"the run failed: {type(e).__name__}: {e}")
            q.put({"event": "error", "status": status, "message": msg})
        finally:
            q.put(None)

    def lines():
        yield line({"event": "start", "run": rid, "steps": STEPS})
        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            yield line(item)
    return StreamingResponse(lines(), media_type=NDJSON)


@app.post("/api/generate")
def api_generate(g: Generate, stream: bool = False):
    def prepare():
        ready()
        rid, d = new_run()
        stem = generate.write_sheet(d, g.seed, g.scan, None if g.level == "mix" else g.level)
        return rid, lambda progress=None: live(rid, d, stem + ".png", stem + "-spec.txt", stem + ".json", {"kind": "generated", "title": f"Generated drawing, seed {g.seed}"}, progress)
    return stream_run(prepare) if stream else prepare()[1]()


@app.post("/api/upload")
def api_upload(file: UploadFile = File(...), spec: str | None = Form(None), stream: bool = False):
    def prepare():
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
        return rid, lambda progress=None: live(rid, d, img, sp, None, {"kind": "upload", "title": (file.filename or "uploaded drawing")[:80]}, progress)
    return stream_run(prepare) if stream else prepare()[1]()


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


# the page's own assets (style.css, app.js, mock/) are referenced from the root, as on the static Space; last, so the API routes win
if os.path.isdir(STATIC):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="site")
