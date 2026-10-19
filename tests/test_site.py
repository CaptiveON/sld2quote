"""The site's back end (site/app.py) through FastAPI's test client, with the classical reader (SLD2QUOTE_READER=classical,
so no weights are needed) and the runs in a temporary directory: health, the samples list without and with an index, a
live generated sheet scored against its answer key (as JSON and as an NDJSON stream of step events), the upload checks, a run file served, the models warmed by
the first health call and 503 after a load error, the daily cap on live runs (429; samples and scorecards never capped), and the OCR crop pass following the
engine (on for vision, off for rapidocr) unless overridden, in the reader and through the server's SLD2QUOTE_CROPS, and the chart data
(eval/charts.json validated by scripts/build_charts.py's own check, served by /api/charts), and a sample rerun live with its stored
answer key and specification. About 12 s."""
import importlib.util, json, os, sys, tempfile
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.environ["SLD2QUOTE_READER"] = "classical"
os.environ["SLD2QUOTE_RUNS"] = tempfile.mkdtemp(prefix="sld2quote-runs-")
sys.path.insert(0, os.path.join(ROOT, "site"))
import app as site  # noqa: E402
from sld2quote import reader  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

RESULT_KEYS = {"id", "source", "images", "size", "steps", "devices", "board", "bom", "issues", "conflicts", "spec", "score", "timing", "reader", "engine", "model", "files"}


@pytest.fixture(scope="module")
def client():
    with TestClient(site.app) as c:
        yield c


def test_the_first_health_call_warms_the_models(client):
    assert not site.STATE["warm"]
    r = client.get("/api/health"); h = r.json()
    assert r.status_code == 200 and h["ok"] and h["warm"] and h["model"] == "classical" and h["engine"] in ("vision", "rapidocr")
    assert site.STATE["warm"] and client.get("/api/health").status_code == 200


def test_samples_empty_then_with_an_index(client, tmp_path, monkeypatch, request):
    monkeypatch.setattr(site, "SAMPLES", str(tmp_path)); site.cards.cache_clear(); request.addfinalizer(site.cards.cache_clear)
    assert client.get("/api/samples").json() == {"samples": []}
    (tmp_path / "s1").mkdir()
    (tmp_path / "index.json").write_text(json.dumps([{"id": "s1", "title": "Sheet 1", "kind": "sample", "copy": "clean", "has_answer_key": True, "thumb": "s1/thumb.png"}]))
    (tmp_path / "s1" / "result.json").write_text(json.dumps({"id": "s1", "source": {"kind": "sample", "title": "x", "copy": "scan", "has_answer_key": True, "note": "n"}, "images": {"sheet": "sheet.png", "overlay": "overlay.png"},
        "files": {"quotation_md": "quotation.md"}, "devices": [1, 2, 3], "bom": {"total_gbp": 1234.5}, "issues": [1], "timing": {"total": 9.5},
        "score": {"precision": 0.9, "recall": 0.8, "price_error": 0.1, "rating_accuracy": 0.7, "symbols_total": 3}}))
    (tmp_path / "s1" / "thumb.png").write_bytes(b"\x89PNG")
    site.cards.cache_clear()
    card = client.get("/api/samples").json()["samples"][0]
    assert card == {"id": "s1", "title": "Sheet 1", "kind": "sample", "copy": "scan", "has_answer_key": True, "note": "n", "devices": 3, "total_gbp": 1234.5, "issues": 1,
                    "score": {"precision": 0.9, "recall": 0.8, "price_error": 0.1, "rating_accuracy": 0.7}, "seconds": 9.5, "thumb": "/api/samples/s1/files/thumb.png"}
    r = client.get("/api/samples/s1").json()
    assert r["images"]["overlay"] == "/api/samples/s1/files/overlay.png" and r["files"]["quotation_md"] == "/api/samples/s1/files/quotation.md"
    assert client.get("/api/samples/s1/files/thumb.png").content == b"\x89PNG"
    assert client.get("/api/samples/s1/files/..%2Findex.json").status_code in (400, 404)
    assert client.get("/api/samples/nope").status_code == 404


def test_generate_scores_the_sheet_and_serves_its_files(client):
    r = client.post("/api/generate", json={"seed": 7, "level": "simple", "scan": False})
    assert r.status_code == 200, r.text
    res = r.json()
    assert set(res) == RESULT_KEYS and res["reader"] == "classical" and res["source"]["kind"] == "generated" and res["source"]["seed"] == 7
    assert all({"conf", "score", "box"} <= set(d) for d in res["devices"])  # score: the detector's raw score once the reader records it
    assert [s["key"] for s in res["steps"]] == ["text", "symbols", "labels", "spec", "price"] and all(s["count"] >= 0 and "plain" in s for s in res["steps"])
    sc = res["score"]
    assert sc and sc["symbols_total"] > 0 and 0 <= sc["recall"] <= 1 and sc["reference_total_gbp"] > 0 and sc["per_type"]
    assert res["spec"]["clauses"] and res["bom"]["total_gbp"] > 0 and res["images"]["overlay"] == f"/api/runs/{res['id']}/overlay.png"
    f = client.get(res["images"]["overlay"])
    assert f.status_code == 200 and f.content[:4] == b"\x89PNG"
    assert client.get(res["files"]["quotation_md"]).text.startswith("# Quotation")
    assert client.post("/api/generate", json={"seed": -1}).status_code == 400
    assert client.post("/api/generate", json={"seed": 7, "level": "hard"}).status_code == 400


def test_upload_rejects_a_non_image_and_accepts_a_png(client):
    r = client.post("/api/upload", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 400 and "PNG or JPEG" in r.json()["detail"]
    run = sorted(os.listdir(site.RUNS))[-1]
    png = open(os.path.join(site.RUNS, run, "sheet.png"), "rb").read()
    r = client.post("/api/upload", files={"file": ("mine.png", png, "image/png")}, data={"spec": "7.3 The assembly shall withstand a short-circuit current of 50 kA.\n"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert set(res) == RESULT_KEYS and res["score"] is None and res["source"] == dict(res["source"], kind="upload", has_answer_key=False, seed=None)
    assert res["spec"]["clauses"][0]["id"] == "7.3" and res["board"]["fault_kA"]
    assert client.get(res["images"]["sheet"]).status_code == 200


def test_a_load_error_gives_503_with_the_error_for_good(client, monkeypatch):
    monkeypatch.setitem(site.STATE, "warm", False); monkeypatch.setitem(site.STATE, "error", "RuntimeError: no weights")
    r = client.get("/api/health")
    assert r.status_code == 503 and r.json() == dict(r.json(), ok=False, warm=False, error="RuntimeError: no weights")
    r = client.post("/api/generate", json={"seed": 1})
    assert r.status_code == 503 and "no weights" in r.json()["detail"]
    assert client.get("/api/samples").status_code == 200 and client.get("/api/scorecards").status_code == 200


def test_the_daily_cap_on_live_runs(client, monkeypatch):
    monkeypatch.setattr(site, "DAILY_RUNS", 0)
    r = client.post("/api/generate", json={"seed": 1})
    assert r.status_code == 429 and "allowance" in r.json()["detail"]
    assert client.get("/api/samples").status_code == 200 and client.get("/api/scorecards").status_code == 200
    monkeypatch.setattr(site, "DAILY_RUNS", site.runs_today() + 1)  # one more run today, then the cap
    assert client.post("/api/generate", json={"seed": 2, "level": "simple"}).status_code == 200
    assert client.post("/api/generate", json={"seed": 2, "level": "simple"}).status_code == 429


def test_scorecards_and_index(client):
    assert set(client.get("/api/scorecards").json()) == {"results_md", "ocr_md", "detector_md", "problems_md"}
    assert client.get("/").status_code == 200 and client.get("/static/../app.py").status_code in (400, 404)


def test_the_crop_pass_follows_the_engine_unless_overridden(tmp_path, monkeypatch):
    from PIL import Image
    page = tmp_path / "blank.png"; Image.new("L", (400, 300), 255).save(page)
    tags = []
    monkeypatch.setattr(reader, "_ocr_image", lambda img, scale, tag: tags.append(tag) or [])
    one = lambda page_img, ink, texts: [{"id": "S1", "kind": "BREAKER", "box": [100, 100, 30, 30], "shape_score": 1.0}]
    for engine, crops, expect in [("vision", None, True), ("rapidocr", None, False), ("rapidocr", True, True), ("vision", False, False)]:
        monkeypatch.setenv("SLD2QUOTE_OCR", engine); tags.clear()
        ex = reader.read_sheet(str(page), finder=one, crops=crops)
        assert ex["crops"] is expect and ("crop" in tags) is expect and tags[0] == "page", (engine, crops, tags)
    assert reader.choose("classical", crops=0).keywords == {"crops": False}


def test_the_server_passes_crops_through(client, monkeypatch):
    monkeypatch.setattr(site, "CROPS", False)
    res = client.post("/api/generate", json={"seed": 3, "level": "simple"}).json()
    assert client.get(res["files"]["extraction_json"]).json()["crops"] is False


def test_a_streamed_run_yields_start_the_five_steps_then_the_result(client):
    r = client.post("/api/generate?stream=1", json={"seed": 5, "level": "simple"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson"), r.text
    ev = [json.loads(l) for l in r.text.splitlines()]
    assert ev[0]["event"] == "start" and ev[0]["run"] and [s["step"] for s in ev[0]["steps"]] == ["text", "symbols", "labels", "spec", "price"] and all(s["label"] for s in ev[0]["steps"])
    assert [(e["event"], e["step"]) for e in ev[1:6]] == [("step", s) for s in ("text", "symbols", "labels", "spec", "price")] and all(e["seconds"] >= 0 for e in ev[1:6])
    counts = {"text": {"texts"}, "symbols": {"symbols"}, "labels": {"linked", "devices"}, "spec": {"clauses", "conflicts"}, "price": {"lines", "total_gbp"}}
    assert all(set(e) == {"event", "step", "seconds"} | counts[e["step"]] for e in ev[1:6]) and ev[4]["clauses"] > 0 and ev[5]["lines"] > 0 and ev[5]["total_gbp"] > 0
    res = ev[6]["result"]
    assert ev[6]["event"] == "result" and len(ev) == 7 and set(res) == RESULT_KEYS and res["id"] == ev[0]["run"] and res["images"]["overlay"] == f"/api/runs/{res['id']}/overlay.png"


def test_a_refused_streamed_run_is_one_error_line(client, monkeypatch):
    ev = [json.loads(l) for l in client.post("/api/upload?stream=1", files={"file": ("x.txt", b"no", "text/plain")}).text.splitlines()]
    assert ev == [{"event": "error", "status": 400, "message": "not a readable image: send a PNG or JPEG"}]
    monkeypatch.setattr(site, "DAILY_RUNS", 0)
    r = client.post("/api/generate?stream=1", json={"seed": 5})
    assert r.status_code == 200 and json.loads(r.text) == {"event": "error", "status": 429, "message": site.OVER}
    monkeypatch.setitem(site.STATE, "warm", False); monkeypatch.setitem(site.STATE, "error", "boom")
    ev = [json.loads(l) for l in client.post("/api/upload?stream=1", files={"file": ("x.png", b"\x89PNG", "image/png")}).text.splitlines()]
    assert ev == [{"event": "error", "status": 503, "message": "the server could not load its models: boom"}]


def test_the_chart_data_validates_and_is_served(client, monkeypatch):
    spec = importlib.util.spec_from_file_location("build_charts", os.path.join(ROOT, "scripts", "build_charts.py"))
    bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
    doc = json.load(open(os.path.join(ROOT, "eval", "charts.json")))
    bc.validate(doc)
    r = client.get("/api/charts")
    assert r.status_code == 200 and r.json() == doc
    monkeypatch.setattr(site, "CHARTS", "/nowhere/charts.json")
    assert client.get("/api/charts").status_code == 404


def test_a_sample_reruns_live_with_its_stored_answer_key(client, tmp_path, monkeypatch, request):
    from sld2quote.generate import write_sheet
    monkeypatch.setattr(site, "SAMPLES", str(tmp_path)); site.cards.cache_clear(); request.addfinalizer(site.cards.cache_clear)
    d = tmp_path / "s9"; d.mkdir(); stem = write_sheet(str(d), 9, level="simple")
    for src, name in ((stem + ".png", "sheet.png"), (stem + ".json", "answer.json"), (stem + "-spec.txt", "spec.txt")):
        os.rename(src, d / name)
    (tmp_path / "index.json").write_text(json.dumps([{"id": "s9", "title": "Sample nine", "kind": "sample", "copy": "clean", "has_answer_key": True, "note": "n", "thumb": "thumb.jpg"}]))
    r = client.post("/api/samples/s9/run")
    assert r.status_code == 200, r.text
    res = r.json()
    assert set(res) == RESULT_KEYS and res["source"] == dict(res["source"], kind="sample", sample="s9", title="Sample nine", seed=9, copy="clean", has_answer_key=True, note="n")
    assert res["score"]["symbols_total"] > 0 and res["spec"]["clauses"] and res["images"]["sheet"] == f"/api/runs/{res['id']}/sheet.png"
    assert os.path.isdir(os.path.join(site.RUNS, res["id"]))
    ev = [json.loads(l) for l in client.post("/api/samples/s9/run?stream=1").text.splitlines()]
    assert [e["event"] for e in ev] == ["start"] + ["step"] * 5 + ["result"] and ev[-1]["result"]["source"]["sample"] == "s9"
    assert client.post("/api/samples/nope/run").status_code == 404 and client.post("/api/samples/nope/run?stream=1").json()["status"] == 404


def test_every_static_page_is_served_at_its_directory(client):
    pages = [d for d, _, fs in os.walk(site.STATIC) if "index.html" in fs]
    assert pages, "site/static has no index.html"
    for d in pages:
        rel = os.path.relpath(d, site.STATIC); url = "/" if rel == "." else f"/{rel}/"
        r = client.get(url + "?sample=sheet-10833")
        assert r.status_code == 200 and r.text == open(os.path.join(d, "index.html"), encoding="utf-8").read(), url
        if url != "/":
            r = client.get(url.rstrip("/"), follow_redirects=False)
            assert r.status_code in (301, 307, 308) and r.headers["location"].endswith(url), url
