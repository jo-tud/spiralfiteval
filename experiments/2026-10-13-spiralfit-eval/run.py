"""Score published spiral fits with tools/spiralfiteval. See README.md.

    PY=python    # Python >= 3.11 with numpy, scipy, tifffile
    $PY experiments/2026-10-13-spiralfit-eval/run.py [--fits A2,C]

Inputs (fetched if missing, all public): Jashann's vesuvius-scrolling v1.1 grids (~400 MB),
Armando Gaona's PHerc. 0211 meshes (~60 MB), the Paris 4 human winding files and eval fibers.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in ("tools/spiralfiteval/src", "tools/windingcheck/src", "tools/windbench"):
    sys.path.insert(0, str(REPO / p))

from spiralfiteval.data import annotation_points, eval_fibers, fetch, scroll_dir  # noqa: E402
from spiralfiteval.evaluate import evaluate  # noqa: E402
from spiralfiteval.fit import check_seams, load_fit, read_grids  # noqa: E402
from spiralfiteval.score import _switch_steps  # noqa: E402
from windbench.metrics import densify  # noqa: E402
from windingcheck.chains import load_pcl  # noqa: E402
from windingcheck.score import build_truth  # noqa: E402


def main_data_root() -> Path:
    """Inputs go in the repository's gitignored data/ (one copy shared by all git worktrees)."""
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute",
                          "--git-common-dir"], capture_output=True, text=True).stdout.strip()
    return Path(out).parent / "data" if out else REPO / "data"


DATA = main_data_root()
FITS = DATA / "spiralfits"
JASHANN = "https://github.com/Jashann/vesuvius-scrolling/releases/download/v1.1"
GAONA = "https://github.com/armando-gaona/pherc0211-first-letters-free-compute.git"
GAONA_COMMIT = "7b63279f6fcd5ce7f2f0db509abeb9ba81b211ed"

BAND1, BAND2 = (8400, 9400), (11000, 12000)
# Jashann's report (v1.1, section 2b): slips per wrap, 95% ladder-bootstrap CI.
# Input sets from the same report; "human" = took the organisers' human winding annotations.
PARIS4 = {
    "A2": (BAND1, 0.203, (0.170, 0.249), "auto"),
    "A2_seed2": (BAND1, 0.221, (0.183, 0.276), "auto"),
    "B3w10": (BAND1, 0.093, (0.062, 0.127), "auto+PCU"),
    "B3w10_seed2": (BAND1, 0.088, (0.057, 0.124), "auto+PCU"),
    "A": (BAND1, 0.159, (0.104, 0.214), "patches"),
    "B": (BAND1, 0.156, (0.085, 0.225), "patches+PCU"),
    "C": (BAND1, 0.030, (0.007, 0.056), "patches+human"),
    "B2": (BAND1, None, None, "auto+PCU"),
    "B3w40": (BAND1, None, None, "auto+PCU"),
    "B4w4": (BAND1, None, None, "auto+PCU"),
    "B4w20": (BAND1, None, None, "auto+PCU"),
    "A2_z11000": (BAND2, 0.306, (0.247, 0.371), "auto"),
    "B3w10_z11000": (BAND2, 0.286, (0.198, 0.379), "auto+PCU"),
}
PHERC0826 = ["A0", "B0", "B1w10", "B1w20", "C0", "C1"]


def fetch_fits():
    d = FITS / "jashann_v1.1"
    d.mkdir(parents=True, exist_ok=True)
    for name in [f"Paris4_{k}" for k in PARIS4] + [f"PHerc0826_{k}" for k in PHERC0826]:
        f = d / f"grids_{name}.npz"
        if not f.exists():
            subprocess.run(["curl", "-sfL", "-o", str(f), f"{JASHANN}/grids_{name}.npz"], check=True)
    g = FITS / "pherc0211_gaona"
    if not (g / "meshes").exists():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", GAONA, str(g)], check=True)
        subprocess.run(["git", "-C", str(g), "sparse-checkout", "set", "meshes"], check=True)
        subprocess.run(["git", "-C", str(g), "checkout", "-q", GAONA_COMMIT], check=True)


def positive_control_i4(fit, truth, chain_ids, tol_n=10.0, flicker=0.0, bridge=1e9):
    """Switches along human radial chains (segment by segment) vs the human wraps crossed.
    The defaults are the settings the first run used; H2 itself counts with flicker=10 and
    bridge=10, and under those this check fails (found in a code review)."""
    sw = wraps = 0
    ch = truth.chains
    for c in chain_ids:
        idx = np.flatnonzero(ch.chain == c)
        for a, b in zip(idx[:-1], idx[1:]):
            p = densify(np.vstack([ch.xyz[a], ch.xyz[b]]), 1.0)
            f, _ = fit.project(p, tol_n)
            if not (np.isfinite(f[0]) and np.isfinite(f[-1])):
                continue
            s_idx, _, _ = _switch_steps(p, f, jump=0.5, flicker=flicker, bridge=bridge)
            sw += len(s_idx)
            wraps += abs(ch.level[b] - ch.level[a])
    return {"switches": int(sw), "human_wraps": float(wraps),
            "ratio": float(sw / wraps) if wraps else float("nan")}


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=None, help="comma list of Paris 4 run names (default all)")
    ap.add_argument("--no-others", action="store_true", help="skip PHerc0826 / PHerc0211")
    args = ap.parse_args(argv)
    fetch()
    fetch_fits()
    log = open(HERE / "run.log", "a")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    truth = build_truth(load_pcl(scroll_dir() / "relative_windings.json"))
    ann = annotation_points()
    fibers = {b: [f["xyz"] for f in eval_fibers(z_range=b)] for b in (BAND1, BAND2)}
    say(f"truth {len(truth.chains)} clicks; annotations {len(ann)}; fibers band1 {len(fibers[BAND1])} band2 {len(fibers[BAND2])}")

    names = args.fits.split(",") if args.fits else list(PARIS4)
    results = {"paris4": {}, "pherc0826": {}, "pherc0211": {}}
    arrays = {}
    for name in names:
        band, ref, ci, inputs = PARIS4[name]
        src = FITS / "jashann_v1.1" / f"grids_Paris4_{name}.npz"
        t0 = time.time()
        grids = read_grids(src)
        seams = check_seams(grids)
        fit = load_fit(src, name=name, column_sense=seams["sense"])
        r = evaluate(fit, truth=truth, fibers=fibers[band], annotations=ann, z_range=band,
                     keep_arrays=name in ("A2", "C"))
        if name in ("A2", "C"):
            arrays[name] = r.pop("_arrays")
        if name == "C":
            z = truth.chains.xyz[:, 2]
            inb = (z >= band[0]) & (z < band[1])
            ids = [c for c in np.unique(truth.chains.chain[inb]) if inb[truth.chains.chain == c].all()]
            r["I4"] = positive_control_i4(fit, truth, ids)
        r.update(band=list(band), inputs=inputs, held_out_H1="human" not in inputs, seams=seams,
                 jashann_slips=ref, jashann_ci=list(ci) if ci else None,
                 seconds_total=round(time.time() - t0, 1),
                 peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
        results["paris4"][name] = strip(r)
        h1 = r["H1"]
        say(f"{name}: slips {h1['slips_per_wrap']:.3f} {h1.get('slips_ci95')} (Jashann {ref}) "
            f"adj {h1['adjacent_tests']} | H2 {r['H2']['per_1000_vox']:.2f}/1000vox | "
            f"H3 med {r['H3'].get('median', float('nan')):.2f} w4 {r['H3']['within_4']:.3f} | "
            f"H5 dup {r['H5']['duplicate']:.4f} | {r['seconds_total']} s")
        del fit
    if not args.no_others:
        for name in PHERC0826:
            src = FITS / "jashann_v1.1" / f"grids_PHerc0826_{name}.npz"
            seams = check_seams(read_grids(src))
            fit = load_fit(src, name=name, column_sense=seams["sense"])
            r = strip(evaluate(fit))
            r.update(seams=seams)
            results["pherc0826"][name] = r
            say(f"PHerc0826 {name}: H5 dup {r['H5']['duplicate']:.4f} gap med {r['H5'].get('median')}")
            del fit
        for spliced in (False, True):
            src = FITS / "pherc0211_gaona" / "meshes"
            seams = check_seams(read_grids(src, spliced=spliced))
            fit = load_fit(src, spliced=spliced, name="gaona" + ("_spliced" if spliced else ""),
                           column_sense=seams["sense"])
            r = strip(evaluate(fit))
            r.update(seams=seams)
            results["pherc0211"][fit.name] = r
            say(f"PHerc0211 {fit.name}: H5 dup {r['H5']['duplicate']:.4f} gap med {r['H5'].get('median')}")
            del fit

    results["criteria"] = criteria(results)
    out = HERE / ("results.json" if not args.fits else f"results_{'_'.join(names)}.json")
    out.write_text(json.dumps(results, indent=1, default=float))
    np.savez_compressed(HERE / "arrays_A2_C.npz", **{f"{n}_{k}": v for n, a in arrays.items()
                                                     for k, v in _flat(a).items()})
    say("criteria", json.dumps(results["criteria"], indent=1, default=float))


def _flat(a, prefix=""):
    out = {}
    for k, v in a.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}_"))
        elif isinstance(v, np.ndarray):
            out[prefix + k] = v
    return out


def criteria(res):
    """Checks fixed before the run, keyed as in results.json: I2 = H1 inside Jashann's published
    95 % CI, I3 = fit C has the lowest H1, I4 = H2 counted along the human chains against the
    human wraps crossed, I5 = eval fibers on fit C (scale check), I6 = cost per fit."""
    p4 = res["paris4"]
    c = {}
    refs = {n: r for n, r in p4.items() if r.get("jashann_ci")}
    inside = {n: bool(r["jashann_ci"][0] <= r["H1"]["slips_per_wrap"] <= r["jashann_ci"][1])
              for n, r in refs.items()}
    c["I2"] = {"inside_ci": inside, "n_inside": sum(inside.values()), "n": len(inside),
               "pass": len(inside) == 9 and sum(inside.values()) >= 7}
    b1 = {n: r["H1"]["slips_per_wrap"] for n, r in p4.items() if r["band"] == list(BAND1)}
    if "C" in b1:
        c["I3"] = {"C": b1["C"], "min_other": min(v for n, v in b1.items() if n != "C") if len(b1) > 1 else None,
                   "pass": len(b1) == 11 and all(b1["C"] < v for n, v in b1.items() if n != "C")}
    if "C" in p4 and "I4" in p4["C"]:
        i4 = p4["C"]["I4"]
        c["I4"] = dict(i4, **{"pass": bool(0.8 <= i4["ratio"] <= 1.2)})
        h3 = p4["C"]["H3"]
        c["I5"] = {"on_fit": 1 - h3["not_on_fit"], "median": h3.get("median"),
                   "pass": bool(1 - h3["not_on_fit"] >= 0.8 and h3.get("median", 99) <= 4)}
    c["I6"] = {"max_seconds": max(r["seconds_total"] for r in p4.values()),
               "peak_rss_gb": max(r["peak_rss_gb"] for r in p4.values()),
               "pass": bool(max(r["seconds_total"] for r in p4.values()) < 600
                            and max(r["peak_rss_gb"] for r in p4.values()) < 16)}
    return c


if __name__ == "__main__":
    main()
