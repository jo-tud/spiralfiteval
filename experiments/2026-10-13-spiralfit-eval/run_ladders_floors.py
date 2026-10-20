"""H1 on Jashann's ladder sets next to his per-ladder results, H5 per winding, windings per
labelled wrap (villa #2048 check), and every reference metric against its floor. See README.md.

    PY=python   # any Python with numpy, scipy and this repository's three packages; see README.md
    $PY experiments/2026-10-13-spiralfit-eval/run_ladders_floors.py [--fits C,A2] [--seeds 5] [--no-others]

Inputs as run.py, plus four small files from Jashann/vesuvius-scrolling at a pinned commit: his
per-ladder slip counts for the Paris 4 and PHerc0826 fits and the 8 PHerc0826 gold slices held
out of every 0826 fit.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402  (sets sys.path, data roots, fit lists)

os.environ.setdefault("WINDINGCHECK_DATA", str(base.DATA / "spiral_datasets"))

from spiralfiteval.data import annotation_points, eval_fibers, fetch, scroll_dir  # noqa: E402
from spiralfiteval.evaluate import evaluate  # noqa: E402
from spiralfiteval.fit import check_seams, load_fit, read_grids  # noqa: E402
from spiralfiteval.ladders import load_ladders  # noqa: E402
from windingcheck.chains import load_pcl  # noqa: E402
from windingcheck.score import build_truth  # noqa: E402

REPO_RAW = "https://raw.githubusercontent.com/Jashann/vesuvius-scrolling"
COMMIT = "4f4997adcccaa9124ab5af02605233f7e3388c69"   # main, 2026-09-29, release v1.1 grids
FILES = {"paris4": "results/fit/e31b_paris4_bootstrap.json",
         "paris4_seed2": "results/fit/e31b_paris4_seed2_bootstrap.json",
         "paris4_z11000": "results/fit/e31b_paris4_z11000_bootstrap.json",
         "pherc0826": "results/fit/e31b_pherc0826_bootstrap.json",
         "gold0826": "data/gold/gold0826_heldout8.json",
         "scorer": "exp/e31b_bootstrap.py"}
JDIR = base.FITS / "jashann_repo" / COMMIT[:7]
BAND0826 = (8500, 9500)
# run name here -> (his results file, his fit key)
HIS = {"A2": ("paris4", "A2"), "A2_seed2": ("paris4_seed2", "A2_s2"), "B2": ("paris4", "B2"),
       "A": ("paris4", "A"), "B": ("paris4", "B"), "C": ("paris4", "C"), "B4w4": ("paris4", "B4w4"),
       "B3w10": ("paris4", "B3w10"), "B3w10_seed2": ("paris4_seed2", "B3w10_s2"),
       "B4w20": ("paris4", "B4w20"), "B3w40": ("paris4", "B3w40"),
       "A2_z11000": ("paris4_z11000", "A2"), "B3w10_z11000": ("paris4_z11000", "B3w10")}


def fetch_jashann():
    JDIR.mkdir(parents=True, exist_ok=True)
    for k, rel in FILES.items():
        f = JDIR / rel
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-sfL", "-o", str(f), f"{REPO_RAW}/{COMMIT}/{rel}"], check=True)
    # his scorer reads the ladder files through pcu.data; this stub serves the local copies
    (JDIR / "pcu").mkdir(exist_ok=True)
    (JDIR / "pcu" / "__init__.py").write_text("")
    (JDIR / "pcu" / "data.py").write_text(
        "import json, os\nP4_SPIRAL = os.environ['P4_SPIRAL_DIR']\n"
        "def fetch_json(url):\n    return json.load(open(url))\n")
    return {k: JDIR / rel for k, rel in FILES.items()}


def run_his_scorer(kind: str, fits: dict, env: dict) -> dict:
    """Jashann's exp/e31b_bootstrap.py, unchanged, on the same grids: per-ladder slips S and
    assigned pairs T per fit (his JSON output keeps only the totals, so read his ``res``)."""
    out = JDIR / f"his_{kind}_{'_'.join(fits)}.json"
    code = ("import json, runpy, sys; s = sys.argv[1]; o = sys.argv[2]; sys.argv = [s] + sys.argv[3:]; "
            "g = runpy.run_path(s, run_name='__main__'); json.dump(g['res'], open(o, 'w'))")
    args = [sys.executable, "-c", code, str(JDIR / FILES["scorer"]), str(out), kind,
            *[f"{n}={p}" for n, p in fits.items()], "--out", str(out) + ".bootstrap.json", "--nboot", "10"]
    subprocess.run(args, check=True, cwd=JDIR, env=dict(os.environ, **env), stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())


def compare(tool: dict, his: dict, rerun: dict) -> dict:
    """This tool's ladder H1 next to his published totals, and ladder by ladder against his scorer
    re-run on the same grid (same ladder order)."""
    S_h, T_h = np.asarray(rerun["S"]), np.asarray(rerun["T"])
    T_o = np.asarray(tool["T"])
    out = {"his_rate": his["rate"], "his_ci95": his.get("ci95"), "his_slips": his["slips"],
           "his_pairs": his["pairs"], "his_unassigned": his.get("unassigned"),
           "rerun_matches_published": bool(rerun["slips"] == his["slips"] and rerun["pairs"] == his["pairs"]),
           "tool_pairs": int(T_o.sum()), "tool_unassigned": tool["unassigned_points"]}
    for rule in ("published", "tool"):
        S_o = np.asarray(tool[rule]["S"])
        out[rule] = {"rate": tool[rule]["rate"], "ci95": tool[rule].get("ci95"),
                     "delta_points": 100 * (tool[rule]["rate"] - his["rate"]),
                     "ladders_identical": int(((S_o == S_h) & (T_o == T_h)).sum())}
    # his ladders where both scorers assign the same pairs: slips there differ only by rule/projector
    same_T = T_o == T_h
    out["ladders_same_pairs"] = int(same_T.sum())
    out["ladders"] = len(T_h)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=None, help="comma list of Paris 4 run names (default all)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-others", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    fetch()
    base.fetch_fits()
    jf = fetch_jashann()
    his = {k: json.loads(jf[k].read_text()) for k in FILES if k not in ("gold0826", "scorer")}
    log = open(HERE / "run_ladders_floors.log", "a")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    d = scroll_dir()
    truth = build_truth(load_pcl(d / "relative_windings.json"))
    ann = annotation_points()
    files = [d / "abs_winding.json", d / "relative_windings.json"]
    bands = (base.BAND1, base.BAND2)
    fibers = {b: [f["xyz"] for f in eval_fibers(z_range=b)] for b in bands}
    ladders = {b: load_ladders(files, b) for b in bands}
    gold = load_ladders(jf["gold0826"], BAND0826, nmax=3000, seed=0)
    npz = base.FITS / "jashann_v1.1"
    names = args.fits.split(",") if args.fits else list(base.PARIS4)
    rerun = {}
    for band in bands:
        sel = {n: npz / f"grids_Paris4_{n}.npz" for n in names if base.PARIS4[n][0] == band}
        if sel:
            rerun.update(run_his_scorer("human", sel, {"P4_SPIRAL_DIR": str(d), "Z0": str(band[0]),
                                                       "Z1": str(band[1])}))
    if not args.no_others:
        rerun0826 = run_his_scorer("gold", {n: npz / f"grids_PHerc0826_{n}.npz" for n in base.PHERC0826},
                                   {"LADDERS": str(jf["gold0826"]), "Z0": str(BAND0826[0]),
                                    "Z1": str(BAND0826[1]), "P4_SPIRAL_DIR": str(d)})
    say(f"start {time.strftime('%Y-%m-%dT%H:%M:%S')} ladders band1 {len(ladders[bands[0]])} "
        f"band2 {len(ladders[bands[1]])} gold0826 {len(gold)}; seeds {args.seeds}")

    res = {"jashann_commit": COMMIT, "paris4": {}, "pherc0826": {}, "pherc0211": {}}
    for name in names:
        band, _, _, inputs = base.PARIS4[name]
        src = base.FITS / "jashann_v1.1" / f"grids_Paris4_{name}.npz"
        t0 = time.time()
        seams = check_seams(read_grids(src))
        fit = load_fit(src, name=name, column_sense=seams["sense"])
        r = evaluate(fit, truth=truth, fibers=fibers[band], annotations=ann, ladders=ladders[band],
                     z_range=band, floor_seeds=args.seeds)
        if name == "C":  # the H2 chain check (I4) under the settings first used and under H2's own
            z = truth.chains.xyz[:, 2]
            inb = (z >= band[0]) & (z < band[1])
            ids = [c for c in np.unique(truth.chains.chain[inb]) if inb[truth.chains.chain == c].all()]
            r["I4"] = {"first_run": base.positive_control_i4(fit, truth, ids),
                       "h2_counter": base.positive_control_i4(fit, truth, ids, flicker=10.0, bridge=10.0),
                       "h2_flicker_no_bridge": base.positive_control_i4(fit, truth, ids, flicker=10.0)}
        hk = HIS[name]
        r["vs_jashann"] = compare(r["H1_ladders"], his[hk[0]][hk[1]], rerun[name])
        r.update(band=list(band), inputs=inputs, held_out_H1="human" not in inputs,
                 seconds_total=round(time.time() - t0, 1),
                 peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
        res["paris4"][name] = base.strip(r)
        say(f"{name}: done in {r['seconds_total']} s, peak {r['peak_rss_gb']} GB")
        del fit
    if not args.no_others:
        for name in base.PHERC0826:
            src = base.FITS / "jashann_v1.1" / f"grids_PHerc0826_{name}.npz"
            t0 = time.time()
            seams = check_seams(read_grids(src))
            fit = load_fit(src, name=name, column_sense=seams["sense"])
            r = evaluate(fit, ladders=gold, floor_seeds=args.seeds)
            r["vs_jashann"] = compare(r["H1_ladders"], his["pherc0826"][name], rerun0826[name])
            r.update(seconds_total=round(time.time() - t0, 1))
            res["pherc0826"][name] = base.strip(r)
            say(f"PHerc0826 {name}: done in {r['seconds_total']} s")
            del fit
        src = base.FITS / "pherc0211_gaona" / "meshes"
        seams = check_seams(read_grids(src))
        fit = load_fit(src, name="gaona", column_sense=seams["sense"])
        res["pherc0211"]["gaona"] = base.strip(evaluate(fit))
        say("PHerc0211 gaona: done")
        del fit
    out = Path(args.out) if args.out else HERE / ("results_ladders_floors.json" if not args.fits
                                                  else f"results_ladders_floors_{'_'.join(names)}.json")
    out.write_text(json.dumps(res, indent=1, default=float))
    say(f"wrote {out.name} {time.strftime('%Y-%m-%dT%H:%M:%S')}")


if __name__ == "__main__":
    main()
