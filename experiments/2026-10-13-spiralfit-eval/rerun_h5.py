"""Rerun H5 only, after the fix to the sheet-gap search (commit 3d32d3f).

H1-H4 do not use the own-sheet exclusion and are unchanged, so only the H5 entries of
results.json are replaced. The first-seen old entry of each fit goes to
``results_before_gap_fix.json``, not into results.json: run.py cannot recompute it, so it
must not sit among the numbers the documented command reproduces.

    PY=python    # same environment as run.py; one fit peaks at ~6 GB, 21 fits take ~40 min
    $PY experiments/2026-10-13-spiralfit-eval/rerun_h5.py [--fits C,A2] [--n-sample 200000]
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import FITS, PARIS4, PHERC0826  # noqa: E402  (also puts the tools on sys.path)

from spiralfiteval.fit import check_seams, load_fit, read_grids  # noqa: E402
from spiralfiteval.score import sheet_spacing  # noqa: E402

BEFORE_NOTE = ("H5 output of the sheet-gap search that commit 3d32d3f (H5 gap-search fix) replaced; "
               "not reproducible from this tree, kept for the record only.")


def sources():
    for name in PARIS4:
        yield "paris4", name, FITS / "jashann_v1.1" / f"grids_Paris4_{name}.npz", False
    for name in PHERC0826:
        yield "pherc0826", name, FITS / "jashann_v1.1" / f"grids_PHerc0826_{name}.npz", False
    for spliced in (False, True):
        yield ("pherc0211", "gaona" + ("_spliced" if spliced else ""),
               FITS / "pherc0211_gaona" / "meshes", spliced)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=None, help="comma list of fit names (default all 21)")
    ap.add_argument("--n-sample", type=int, default=200_000)
    ap.add_argument("--out", default=None, help="default: update results.json in place")
    args = ap.parse_args(argv)
    want = set(args.fits.split(",")) if args.fits else None
    path = HERE / "results.json"
    res = json.loads(path.read_text())
    before_path = HERE / "results_before_gap_fix.json"
    before = json.loads(before_path.read_text()) if before_path.exists() else {"_note": BEFORE_NOTE}
    log = open(HERE / "rerun_h5.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    gaps = {}
    for scroll, name, src, spliced in sources():
        if want is not None and name not in want:
            continue
        t0 = time.time()
        seams = check_seams(read_grids(src, spliced=spliced))
        fit = load_fit(src, spliced=spliced, name=name, column_sense=seams["sense"])
        h5 = sheet_spacing(fit, n_sample=args.n_sample)
        gap, idx = h5.pop("_gap"), h5.pop("_idx")
        if name in ("A2", "C"):
            gaps[name] = (gap, idx)
        h5["seconds"] = round(time.time() - t0, 1)
        h5["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        r = res[scroll][name]
        old = before.setdefault(scroll, {}).setdefault(name, r["H5"])
        r["H5"] = h5
        say(f"{scroll} {name}: median {old.get('median', float('nan')):.2f} -> {h5.get('median', float('nan')):.2f} | "
            f"p10 {old.get('p10', float('nan')):.2f} -> {h5.get('p10', float('nan')):.2f} | "
            f"no_neighbour {old['no_neighbour']:.4f} -> {h5['no_neighbour']:.4f} | "
            f"duplicate {old['duplicate']:.4f} -> {h5['duplicate']:.4f} | {h5['seconds']} s, {h5['peak_rss_gb']} GB")
        del fit
    out = Path(args.out) if args.out else path
    out.write_text(json.dumps(res, indent=1, default=float))
    if args.out is None:
        before_path.write_text(json.dumps(before, indent=1, default=float) + "\n")
    arrays = HERE / "arrays_A2_C.npz"
    if gaps and arrays.exists() and args.out is None:
        a = dict(np.load(arrays))
        for name, (gap, idx) in gaps.items():
            a[f"{name}_h5_gap"], a[f"{name}_h5_idx"] = gap, idx
        np.savez_compressed(arrays, **a)
    say(f"done {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
