"""Command line: python3 -m sld2quote.cli run <sheet.png> [--spec sheet-spec.txt] [--out out/sheet]"""
from __future__ import annotations
import argparse, json, os, sys, time
from . import __version__
from .reader import read_sheet
from .spec import read_spec, cross_check
from .rules import build_bom
from .quote import write_outputs


def run(image: str, spec_path: str | None, out_dir: str) -> dict:
    t0 = time.time()
    ex = read_sheet(image)
    sp = read_spec(spec_path) if spec_path and os.path.exists(spec_path) else None
    conflicts = cross_check(ex["notes"], sp) if sp else []
    bom = build_bom(ex, sp, conflicts)
    meta = {"version": __version__, "seconds": round(time.time() - t0, 1)}
    write_outputs(out_dir, image, ex, sp, conflicts, bom, meta)
    return {"devices": len(ex["devices"]), "lines": len(bom["lines"]), "total_gbp": bom["total_gbp"], "issues": len(bom["issues"]), "conflicts": len(conflicts), "seconds": meta["seconds"], "out": out_dir}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sld2quote")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="read one sheet and write the quotation outputs")
    r.add_argument("image"); r.add_argument("--spec"); r.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        stem = os.path.splitext(os.path.basename(a.image))[0]
        spec = a.spec or os.path.join(os.path.dirname(a.image), stem.replace("-scan", "") + "-spec.txt")
        out = a.out or os.path.join("out", stem)
        res = run(a.image, spec, out)
        print(json.dumps(res))


if __name__ == "__main__":
    main()
