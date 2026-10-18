#!/usr/bin/env python3
"""Deploy the site to a private Hugging Face Space: the live Docker Space, or with --static the free fallback that serves
the precomputed samples with no server; with --model, the detector to its model repo first.

    python3 scripts/deploy_space.py                      push the live (Docker) Space, then print its URL and status
    python3 scripts/deploy_space.py --static             push the static Space instead
    python3 scripts/deploy_space.py --model              upload weights/rtdetr_r18_full to the model repo first
    python3 scripts/deploy_space.py [--static] --public  print what making that Space public would do; add --confirm to do it

Idempotent: an unchanged tree makes no commit, so no rebuild. Uses this machine's huggingface_hub login. A missing repo is
created private. A new Docker Space gets cpu-basic and the login token as its HF_TOKEN secret (the app downloads the
private model with it; replace it in the Space settings, the script never touches it again), and its SLD2QUOTE_* variables
are kept in sync. The Docker Space repo is site/Dockerfile and site/space/README.md at its root plus the DOCKER files at
their own paths; the static Space repo is site/static at its root, site/samples as samples/ and the scorecards at their
own paths (eval/*.md, docs/PROBLEMS.md), with STATIC_README.
"""
import argparse, glob, json, os, sys
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, get_token

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE, STATIC, MODEL, WEIGHTS = "itsmoadnan/sld2quote", "itsmoadnan/sld2quote-samples", "itsmoadnan/sld2quote-detector", "weights/rtdetr_r18_full"
MODEL_FILES = ["config.json", "model.safetensors", "preprocessor_config.json", "card.md", "threshold_drawing.json"]
VARIABLES = {"SLD2QUOTE_MODEL_REPO": MODEL, "SLD2QUOTE_OCR": "rapidocr", "SLD2QUOTE_WEIGHTS": "/data/weights/rtdetr_r18_full"}
SKIP = ("__pycache__", "site/Dockerfile", "site/space/")
SCORECARDS = [("eval/*.md", "", ""), ("docs/PROBLEMS.md", "", "")]
# (local glob, local prefix, Space prefix): a match is published at the Space prefix plus its path after the local prefix
DOCKER = [("sld2quote/**", "", ""), ("site/**", "", ""), ("assets/**", "", ""), ("data/preview/*", "", "")] + SCORECARDS
STATIC_RULES = [("site/static/**", "site/static/", ""), ("site/samples/**", "site/samples/", "samples/")] + SCORECARDS
STATIC_README = b"""---
title: sld2quote samples
sdk: static
license: mit
short_description: sld2quote precomputed results (no live runs)
---
The precomputed results of [sld2quote](https://github.com/CaptiveON/sld2quote), a single-line diagram read into a priced
bill of materials with the evidence for every line, served as a static page: making or uploading a drawing needs the
live server (scripts/deploy_cloudrun.sh in the repository). Synthetic drawings with their answer-key scores, and public-domain real drawings, not scored;
an invented catalogue with illustrative prices.
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
Operating threshold on whole drawings: `score_threshold_drawing` {t} in config.json, chosen in threshold_drawing.json.

"""


def upload_model(api: HfApi) -> None:
    api.create_repo(MODEL, private=True, exist_ok=True)
    src = os.path.join(ROOT, WEIGHTS)
    with open(os.path.join(src, "config.json")) as fh, open(os.path.join(src, "card.md")) as card:
        readme = CARD_HEAD.format(t=json.load(fh)["score_threshold_drawing"]) + card.read()
    ops = [CommitOperationAdd("README.md", readme.encode())] + [CommitOperationAdd(f, os.path.join(src, f)) for f in MODEL_FILES]
    print("model:", api.create_commit(MODEL, ops, commit_message=f"Upload {WEIGHTS}").commit_url)


def ensure_space(api: HfApi, static: bool) -> None:
    repo = STATIC if static else SPACE
    if not api.repo_exists(repo, repo_type="space"):
        if static:
            api.create_repo(repo, repo_type="space", space_sdk="static", private=True)
        else:
            api.create_repo(repo, repo_type="space", space_sdk="docker", space_hardware="cpu-basic", private=True,
                            space_secrets=[{"key": "HF_TOKEN", "value": get_token()}])
        print("created", repo, "(private, static)" if static else "(private, cpu-basic, HF_TOKEN secret set from this machine's login)")
    if static:
        return
    have = api.get_space_variables(repo)
    for k, v in VARIABLES.items():
        if k not in have or have[k].value != v:
            api.add_space_variable(repo, k, v)


def space_files(static: bool) -> dict:
    """Space repo path -> local path relative to ROOT, or the file's bytes"""
    out = {"README.md": STATIC_README} if static else {"Dockerfile": "site/Dockerfile", "README.md": "site/space/README.md"}
    for pat, src, dst in STATIC_RULES if static else DOCKER:
        for p in glob.glob(pat, root_dir=ROOT, recursive=True):
            if os.path.isfile(os.path.join(ROOT, p)) and not any(s in p for s in SKIP):
                out[dst + p[len(src):]] = p
    return out


def push_space(api: HfApi, static: bool) -> None:
    repo, want = STATIC if static else SPACE, space_files(static)
    stale = set(api.list_repo_files(repo, repo_type="space")) - set(want) - {".gitattributes"}
    ops = [CommitOperationAdd(k, v if isinstance(v, bytes) else os.path.join(ROOT, v)) for k, v in sorted(want.items())]
    ops += [CommitOperationDelete(p) for p in sorted(stale)]
    print(f"space: {len(want)} files, {len(stale)} removed:", api.create_commit(repo, ops, commit_message="Deploy the site", repo_type="space").commit_url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static", action="store_true", help=f"the static Space {STATIC} (samples only, no server) instead of {SPACE}")
    ap.add_argument("--model", action="store_true", help=f"upload {WEIGHTS} to {MODEL} first")
    ap.add_argument("--public", action="store_true", help="make the Space public (only prints, unless --confirm)")
    ap.add_argument("--confirm", action="store_true", help="with --public: really do it")
    a = ap.parse_args()
    api, repo = HfApi(), STATIC if a.static else SPACE
    if a.public:
        if not a.confirm:
            print(f"would make https://huggingface.co/spaces/{repo} public: its " + ("pages, samples and scorecards" if a.static else "code, samples and README") +
                  " become visible to everyone" + ("." if a.static else f";\n{MODEL} stays private (the Space reads it with its HF_TOKEN secret, so replace that with a read-only token first).") +
                  "\nRun again with " + ("--static " if a.static else "") + "--public --confirm to do it.")
            return
        api.update_repo_visibility(repo, private=False, repo_type="space")
        print("public:", f"https://huggingface.co/spaces/{repo}")
        return
    need = ("site/static/index.html", "site/samples/index.json") if a.static else ("site/Dockerfile", "site/space/README.md", "site/app.py", "site/requirements.txt")
    for f in need:
        if not os.path.isfile(os.path.join(ROOT, f)):
            sys.exit(f"{f} is missing: nothing to deploy")
    if a.model:
        upload_model(api)
    ensure_space(api, a.static)
    push_space(api, a.static)
    rt = api.get_space_runtime(repo)
    host = repo.replace("/", "-") + (".static.hf.space" if a.static else ".hf.space")
    print(f"https://huggingface.co/spaces/{repo} (page: https://{host}): {rt.stage}, {rt.hardware or rt.requested_hardware}")


if __name__ == "__main__":
    main()
