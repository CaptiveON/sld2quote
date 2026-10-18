"""The site's back end (site/app.py) through FastAPI's test client, with the classical reader (SLD2QUOTE_READER=classical,
so no weights are needed) and the runs in a temporary directory: health, the samples list without and with an index, a
live generated sheet scored against its answer key, the upload checks, a run file served, and the OCR crop pass following the
engine (on for vision, off for rapidocr) unless overridden, in the reader and through the server's SLD2QUOTE_CROPS. About 10 s."""
import json, os, sys, tempfile, time
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
        for _ in range(600):  # the warm-up thread
            if c.get("/api/health").json()["warm"]:
                break
            time.sleep(0.1)
        yield c


def test_health(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["warm"] and h["model"] == "classical" and h["engine"] in ("vision", "rapidocr")


def test_samples_empty_then_with_an_index(client, tmp_path, monkeypatch):
    monkeypatch.setattr(site, "SAMPLES", str(tmp_path))
    assert client.get("/api/samples").json() == {"samples": []}
    (tmp_path / "s1").mkdir()
    (tmp_path / "index.json").write_text(json.dumps([{"id": "s1", "title": "Sheet 1", "kind": "sample", "copy": "clean", "has_answer_key": True, "thumb": "s1/thumb.png"}]))
    (tmp_path / "s1" / "result.json").write_text(json.dumps({"id": "s1", "images": {"sheet": "sheet.png", "overlay": "overlay.png"}, "files": {"quotation_md": "quotation.md"}}))
    (tmp_path / "s1" / "thumb.png").write_bytes(b"\x89PNG")
    assert client.get("/api/samples").json()["samples"][0]["thumb"] == "/api/samples/s1/files/thumb.png"
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


def test_live_runs_answer_503_while_warming(client, monkeypatch):
    monkeypatch.setitem(site.STATE, "warm", False)
    assert client.post("/api/generate", json={"seed": 1}).status_code == 503


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
