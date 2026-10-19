#!/usr/bin/env python3
"""Deploy the static fallback of the site to a private Hugging Face Space: the precomputed samples and scorecards with no
server (the live site runs on Cloud Run, scripts/deploy_cloudrun.sh); with --model, the detector to its model repo first.

    python3 scripts/deploy_space.py                  push the static Space, then print its URL and status (--static: the same)
    python3 scripts/deploy_space.py --model          upload weights/rtdetr_r18_full to the model repo first
    python3 scripts/deploy_space.py --public         print what making the Space public would do; add --confirm to do it

Idempotent: an unchanged tree makes no commit. Uses this machine's huggingface_hub login; a missing repo is created
private. The Space repo is site/static at its root, site/samples as samples/ and the scorecards at their own paths
(eval/*.md, eval/charts.json, docs/PROBLEMS.md), with README; its samples/index.json holds the gallery cards exactly as the server's
/api/samples returns them (site/app.py cards()), so the gallery shows device counts, prices and scores there too.
"""
import argparse, glob, json, os, sys
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC, MODEL = "itsmoadnan/sld2quote-samples", "itsmoadnan/sld2quote-detector"
WEIGHTS = "weights/rtdetr_r18_plain_ema_fix_pass4"  # the site's model, at the model repo's root (MODEL in scripts/deploy_cloudrun.sh)
MODEL_FILES = ["config.json", "model.safetensors", "preprocessor_config.json", "ORIGIN.md"]
ARCHIVE = ("rtdetr_r18_full", ["config.json", "model.safetensors", "preprocessor_config.json", "card.md", "threshold_drawing.json"])  # the first site model, kept
# (local glob, local prefix, Space prefix): a match is published at the Space prefix plus its path after the local prefix
RULES = [("site/static/**", "site/static/", ""), ("site/samples/**", "site/samples/", "samples/"), ("eval/*.md", "", ""), ("eval/charts.json", "", ""),
         ("docs/PROBLEMS.md", "", "")]
README = b"""---
title: sld2quote samples
sdk: static
license: mit
short_description: sld2quote precomputed results (no live runs)
---
The precomputed results of [sld2quote](https://github.com/CaptiveON/sld2quote), a single-line diagram read into a priced
bill of materials with the evidence for every line, served as a static page: making or uploading a drawing needs the
live server (scripts/deploy_cloudrun.sh in the repository). Synthetic drawings with their answer-key scores, and
public-domain real drawings, not scored; an invented catalogue with illustrative prices.
"""
CARD_HEAD = """---
license: apache-2.0
library_name: transformers
pipeline_tag: object-detection
base_model: PekingU/rtdetr_r18vd
tags:
- synthetic-data
---
Synthetic data only. Source, generator and evaluation: [CaptiveON/sld2quote](https://github.com/CaptiveON/sld2quote).
Operating threshold on whole drawings: `score_threshold_drawing` {t} in config.json, which also carries `score_calibration`
(how often a box with a given score was right on the validation drawings).

"""


def upload_model(api: HfApi) -> None:
    """the site's model at the repo root with its card; the first model kept under its own folder; other root files removed"""
    api.create_repo(MODEL, private=True, exist_ok=True)
    src, (old, old_files) = os.path.join(ROOT, WEIGHTS), ARCHIVE
    with open(os.path.join(src, "config.json")) as fh, open(os.path.join(src, "ORIGIN.md")) as o:
        readme = (CARD_HEAD.format(t=json.load(fh)["score_threshold_drawing"]) + "\n".join(l.lstrip("# ") for l in o.read().splitlines()) +
                  f"\n\nThe first site model, {old}, is kept under [{old}/]({old}) with its card (from_pretrained with subfolder=\"{old}\").\n")
    have = set(api.list_repo_files(MODEL))
    ops = [CommitOperationAdd("README.md", readme.encode())] + [CommitOperationAdd(f, os.path.join(src, f)) for f in MODEL_FILES]
    if f"{old}/config.json" not in have:
        ops += [CommitOperationAdd(f"{old}/{f}", os.path.join(ROOT, "weights", old, f)) for f in old_files]
    ops += [CommitOperationDelete(f) for f in sorted(have) if "/" not in f and f not in MODEL_FILES + ["README.md", ".gitattributes"]]
    print("model:", api.create_commit(MODEL, ops, commit_message=f"The site model {os.path.basename(WEIGHTS)} at the root; {old} kept under {old}/").commit_url)


def cards() -> list:
    """the gallery cards as the server's /api/samples gives them"""
    sys.path.insert(0, os.path.join(ROOT, "site"))
    import app  # site/app.py
    return app.cards()


def space_files() -> dict:
    """Space repo path -> local path relative to ROOT, or the file's bytes"""
    out = {"README.md": README}
    for pat, src, dst in RULES:
        for p in glob.glob(pat, root_dir=ROOT, recursive=True):
            if os.path.isfile(os.path.join(ROOT, p)) and "__pycache__" not in p:
                out[dst + p[len(src):]] = p
    out["samples/index.json"] = json.dumps({"samples": cards()}, indent=1).encode()
    return out


def push_space(api: HfApi) -> None:
    if not api.repo_exists(STATIC, repo_type="space"):
        api.create_repo(STATIC, repo_type="space", space_sdk="static", private=True)
        print("created", STATIC, "(private, static)")
    want = space_files()
    stale = set(api.list_repo_files(STATIC, repo_type="space")) - set(want) - {".gitattributes"}
    ops = [CommitOperationAdd(k, v if isinstance(v, bytes) else os.path.join(ROOT, v)) for k, v in sorted(want.items())]
    ops += [CommitOperationDelete(p) for p in sorted(stale)]
    print(f"space: {len(want)} files, {len(stale)} removed:", api.create_commit(STATIC, ops, commit_message="Deploy the static site", repo_type="space").commit_url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static", action="store_true", help="the default (the static Space is the only one); kept for older commands")
    ap.add_argument("--model", action="store_true", help=f"upload {WEIGHTS} to {MODEL} first")
    ap.add_argument("--public", action="store_true", help="make the Space public (only prints, unless --confirm)")
    ap.add_argument("--confirm", action="store_true", help="with --public: really do it")
    a = ap.parse_args()
    api = HfApi()
    if a.public:
        if not a.confirm:
            print(f"would make https://huggingface.co/spaces/{STATIC} public: its pages, samples and scorecards become visible to everyone.\n"
                  "Run again with --public --confirm to do it.")
            return
        api.update_repo_visibility(STATIC, private=False, repo_type="space")
        print("public:", f"https://huggingface.co/spaces/{STATIC}")
        return
    for f in ("site/static/index.html", "site/samples/index.json"):
        if not os.path.isfile(os.path.join(ROOT, f)):
            sys.exit(f"{f} is missing: nothing to deploy")
    if a.model:
        upload_model(api)
    push_space(api)
    rt = api.get_space_runtime(STATIC)
    print(f"https://huggingface.co/spaces/{STATIC} (page: https://{STATIC.replace('/', '-')}.static.hf.space): {rt.stage}")


if __name__ == "__main__":
    main()
