"""windbench CLI: fetch | build | validate | score | baseline."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import io

MAN = Path(__file__).resolve().parent.parent / "manifest"
CROP_SETS = {"v1": MAN / "crops.json", "v1.1": MAN / "crops_v11.json"}
CROP_SET = "v1"


def crops() -> list[dict]:
    return json.loads(CROP_SETS[CROP_SET].read_text())["crops"]


def cmd_fetch(a):
    t = time.time()
    n = io.fetch_meshes()
    print(json.dumps({"bytes_pulled": n, "seconds": round(time.time() - t, 1)}))


def cmd_build(a):
    from .build import build_crop
    from .rasterise import load_meshes
    meshes = load_meshes()
    sel = [c for c in crops() if not a.crop or c["id"] in a.crop]
    for c in sel:
        spec = {"id": c["id"], "origin": c["origin"]}
        rec = build_crop(meshes, spec, force=a.force)
        print(json.dumps({k: rec[k] for k in ("spec", "n_sites", "sites_by_type", "bytes_pulled", "seconds")}))


def cmd_validate(a):
    from .validate import validate
    res = validate([c["id"] for c in crops()], compressed_abs=14.4 if CROP_SET == "v1.1" else None)
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["criteria"], indent=1, default=float))


def cmd_score(a):
    from .score import score_dir
    t = time.time()
    res = score_dir(Path(a.pred), [c["id"] for c in crops()])
    res["seconds"] = round(time.time() - t, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["summary"], indent=1, default=float))


def cmd_zcheck(a):
    from .zcheck import zcheck
    res = zcheck(crops())
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("z_A_um_median", "z_B_um_median", "verdict", "native_bytes_pulled", "seconds")}))


def cmd_baseline(a):
    from .baseline import spiral_prior
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in crops():
        np.savez_compressed(out / f"{c['id']}.npz", k=spiral_prior(c["id"]))
        print(c["id"], "done")


def main(argv=None):
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("WINDBENCH_THREADS", "6"))
    p = argparse.ArgumentParser(prog="windbench")
    p.add_argument("--set", default="v1", choices=sorted(CROP_SETS), help="crop set (default v1)")
    sp = p.add_subparsers(required=True)
    s = sp.add_parser("fetch", help="download the 19 GT meshes (~880 MB)")
    s.set_defaults(fn=cmd_fetch)
    s = sp.add_parser("build", help="build crops listed in manifest/crops.json (idempotent)")
    s.add_argument("--crop", nargs="*")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_build)
    s = sp.add_parser("validate", help="benchmark integrity checks C2-C7")
    s.add_argument("--out", default="validation.json")
    s.set_defaults(fn=cmd_validate)
    s = sp.add_parser("score", help="score a prediction directory (<crop_id>.npz per crop)")
    s.add_argument("pred")
    s.add_argument("--out", default="scores.json")
    s.set_defaults(fn=cmd_score)
    s = sp.add_parser("baseline", help="write the spiral-prior baseline predictions")
    s.add_argument("out")
    s.set_defaults(fn=cmd_baseline)
    s = sp.add_parser("select-v11", help="v1.1: pick compressed crops (GT-only rule) and build them")
    s.set_defaults(fn=lambda a: __import__("windbench.select_v11", fromlist=["run"]).run())
    s = sp.add_parser("zcheck", help="v1.1 Z1: transferred vs native mesh offset on their own scans")
    s.add_argument("--out", default="zcheck.json")
    s.set_defaults(fn=cmd_zcheck)
    a = p.parse_args(argv)
    global CROP_SET
    CROP_SET = a.set
    a.fn(a)


if __name__ == "__main__":
    main()
