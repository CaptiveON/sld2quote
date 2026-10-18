#!/usr/bin/env python3
"""Deploy the live site to its Hugging Face Space and, with --model, the detector to its model repo; both private.

    python3 scripts/deploy_space.py                    push the Space files, then print its URL and status
    python3 scripts/deploy_space.py --model            upload weights/rtdetr_r18_full to the model repo first
    python3 scripts/deploy_space.py --public           print what making the Space public would do; add --confirm to do it

Idempotent: an unchanged tree makes no commit, so no rebuild. Uses this machine's huggingface_hub login. A missing repo is
created private; a new Space gets cpu-basic and the login token as its HF_TOKEN secret (the app downloads the private model
with it; replace it in the Space settings, the script never touches it again). The SLD2QUOTE_* variables are kept in sync.
The Space repo is site/Dockerfile and site/space/README.md at its root plus the SHIP files at their own paths.
"""
import argparse, glob, json, os, sys
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, get_token

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE, MODEL, WEIGHTS = "itsmoadnan/sld2quote", "itsmoadnan/sld2quote-detector", "weights/rtdetr_r18_full"
MODEL_FILES = ["config.json", "model.safetensors", "preprocessor_config.json", "card.md", "threshold_drawing.json"]
VARIABLES = {"SLD2QUOTE_MODEL_REPO": MODEL, "SLD2QUOTE_OCR": "rapidocr", "SLD2QUOTE_WEIGHTS": "/data/weights/rtdetr_r18_full"}
SHIP = ["sld2quote/**", "site/**", "assets/**", "eval/*.md", "docs/PROBLEMS.md", "data/preview/*"]
SKIP = ("__pycache__", "site/Dockerfile", "site/space/")
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


def ensure_space(api: HfApi) -> None:
    if not api.repo_exists(SPACE, repo_type="space"):
        api.create_repo(SPACE, repo_type="space", space_sdk="docker", space_hardware="cpu-basic", private=True,
                        space_secrets=[{"key": "HF_TOKEN", "value": get_token()}])
        print("created", SPACE, "(private, cpu-basic, HF_TOKEN secret set from this machine's login)")
    have = api.get_space_variables(SPACE)
    for k, v in VARIABLES.items():
        if k not in have or have[k].value != v:
            api.add_space_variable(SPACE, k, v)


def space_files() -> dict:
    """Space repo path -> local path, relative to ROOT"""
    out = {"Dockerfile": "site/Dockerfile", "README.md": "site/space/README.md"}
    for pat in SHIP:
        for p in glob.glob(pat, root_dir=ROOT, recursive=True):
            if os.path.isfile(os.path.join(ROOT, p)) and not any(s in p for s in SKIP):
                out[p] = p
    return out


def push_space(api: HfApi) -> None:
    want = space_files()
    stale = set(api.list_repo_files(SPACE, repo_type="space")) - set(want) - {".gitattributes"}
    ops = [CommitOperationAdd(k, os.path.join(ROOT, v)) for k, v in sorted(want.items())] + [CommitOperationDelete(p) for p in sorted(stale)]
    print(f"space: {len(want)} files, {len(stale)} removed:", api.create_commit(SPACE, ops, commit_message="Deploy the site", repo_type="space").commit_url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="store_true", help=f"upload {WEIGHTS} to {MODEL} first")
    ap.add_argument("--public", action="store_true", help="make the Space public (only prints, unless --confirm)")
    ap.add_argument("--confirm", action="store_true", help="with --public: really do it")
    a = ap.parse_args()
    api = HfApi()
    if a.public:
        if not a.confirm:
            print(f"would make https://huggingface.co/spaces/{SPACE} public: its code, samples and README become visible to everyone;\n"
                  f"{MODEL} stays private (the Space reads it with its HF_TOKEN secret, so replace that with a read-only token first).\n"
                  "Run again with --public --confirm to do it.")
            return
        api.update_repo_visibility(SPACE, private=False, repo_type="space")
        print("public:", f"https://huggingface.co/spaces/{SPACE}")
        return
    for f in ("site/Dockerfile", "site/space/README.md", "site/app.py", "site/requirements.txt"):
        if not os.path.isfile(os.path.join(ROOT, f)):
            sys.exit(f"{f} is missing: nothing to deploy")
    if a.model:
        upload_model(api)
    ensure_space(api)
    push_space(api)
    rt = api.get_space_runtime(SPACE)
    print(f"https://huggingface.co/spaces/{SPACE} (app: https://{SPACE.replace('/', '-')}.hf.space): {rt.stage}, {rt.hardware or rt.requested_hardware}")


if __name__ == "__main__":
    main()
