"""spiralfiteval fetch | score FIT [FIT ...] | compare FIT FIT [FIT ...]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _summary(r: dict) -> str:
    out = [f"{r['fit']}: {r['vertices']:,} vertices, windings w{r['windings'][0]:03d}-w{r['windings'][1]:03d}, "
           f"grid step {r['grid_step_vox']:.1f} vx, {r['seconds']} s"]
    if "H1" in r:
        h = r["H1"]
        ci = h.get("slips_ci95")
        out.append(f"  H1 winding identity: slips per wrap {h['slips_per_wrap']:.1%}"
                   + (f" (95% CI {ci[0]:.1%}-{ci[1]:.1%})" if ci else "")
                   + f" on {h['adjacent_tests']} adjacent pairs of {h['chains']} chains; "
                   f"all pairs exact {h.get('exact', float('nan')):.1%}, covered {h['human_pairs_covered']:.0%}")
    if "H2" in r:
        h = r["H2"]
        out.append(f"  H2 sheet switches: {h['per_1000_vox']:.2f} per 1000 voxels of fiber "
                   f"({h['switches']} in {h['voxels']:.0f} vx); fibers with a switch {h['traces_with_switch']:.0%}; "
                   f"clean (fit within 3 vx on both sides) {h['post_hoc_clean_per_1000_vox']:.2f}")
    if "H3" in r:
        h = r["H3"]
        out.append(f"  H3 distance to eval fibers: median {h.get('median', float('nan')):.2f} vx, "
                   f"p90 {h.get('p90', float('nan')):.2f}, within 4 vx {h['within_4']:.0%}, not on fit {h['not_on_fit']:.0%}")
    if "H4" in r:
        w = r["H4"]["within_4"]
        out.append("  H4 within-4 share by distance to annotations: "
                   + ", ".join(f"{b['lo']:.0f}-{b['hi']:.0f}: {b['mean']:.0%}" for b in w))
    if "H1_ladders" in r:
        h = r["H1_ladders"]
        out.append(f"  H1 on ladder set ({h['ladders']} ladders, {h['pairs_assigned']}/{h['pairs']} pairs assigned): "
                   f"slips per wrap {h['tool']['rate']:.1%} (round Δf), {h['published']['rate']:.1%} (winding label, "
                   f"per-ladder sign); fit windings per labelled wrap {r['windings_per_wrap']['ratio']:.2f}")
    h = r["H5"]
    out.append(f"  H5 sheet spacing: median gap {h.get('median', float('nan')):.1f} vx, "
               f"duplicate (< {h['dup_threshold']} vx) {h['duplicate']:.2%}, no neighbour within {h.get('max_gap', 60):.0f} vx {h['no_neighbour']:.2%}")
    if "H5_by_winding" in r:
        h = r["H5_by_winding"]
        top = sorted(h["per_winding"].items(), key=lambda kv: -kv[1]["duplicate"])[:3]
        out.append(f"     per winding: mean {h['mean_over_windings']:.2%} (area-weighted {h['area_weighted']:.2%}); "
                   "most duplicated " + ", ".join(f"w{int(w):03d} {v['duplicate']:.0%}" for w, v in top))
    for key, label in (("H1", "H1 slips/wrap"), ("H1_ladders", "H1 ladders"), ("H2", "H2 per 1000 vx")):
        fl = r.get("floors", {}).get(key)
        if fl:
            j = fl["jitter"]
            out.append(f"  floor {label}: {j['value']:.3g} vs phase-jitter {j['floor_mean']:.3g} "
                       f"(sd {j['floor_sd']:.2g}); shift curve " + " ".join(f"{float(k):+g}:{v:.3g}" for k, v in fl["shift"].items()))
    fl = r.get("floors", {}).get("H3")
    if fl:
        out.append(f"  floor H3 median: {fl['jitter']['value']:.2f} vx vs phase-jitter {fl['jitter']['floor_mean']:.2f}, "
                   f"random {fl['random']['floor_mean']:.2f}; best shift {fl['best_shift']:+g} vx, rise at ±4 {fl['rise_at_4']:.2f} vx")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spiralfiteval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="human winding files + eval fibers for a scroll (PHercParis4)")
    f.add_argument("--scroll", default="PHercParis4")
    s = sub.add_parser("score", help="score one or more fits")
    s.add_argument("fits", nargs="+", help="fit folder (wNNN/ tifxyz meshes) or .npz of wNNN grids")
    s.add_argument("--spliced", action="store_true", help="score the wNNN_spliced meshes")
    s.add_argument("--xyz-order", default="xyz", help="channel order of .npz grids")
    s.add_argument("--references", default=None,
                   help="scroll with public winding references (PHercParis4): adds H1-H4")
    s.add_argument("--z", type=float, nargs=2, default=None, metavar=("Z0", "Z1"),
                   help="z band of the fit, in the fit's voxels (clips references)")
    s.add_argument("--tol", type=float, default=10.0, help="max normal distance to a fit sheet")
    s.add_argument("--ladders", nargs="+", default=None,
                   help="ladder files (VC3D point collections with wind_a) for H1 on a published set")
    s.add_argument("--ladder-nmax", type=int, default=None, help="random subset of ladders (seed 0)")
    s.add_argument("--floors", type=int, default=0, metavar="SEEDS",
                   help="add each reference metric's floor (phase jitter, SEEDS draws) and shift curve")
    s.add_argument("--chains", default="all", choices=("all", "odd", "even"),
                   help="H1 on chains of this parity only; 'odd' is windingcheck's held-out half")
    s.add_argument("--name", action="append", default=None,
                   help="label for one fit, in the order the fits are given (repeatable); "
                        "default is the shortest path suffix that separates them")
    s.add_argument("--out", default=None, help="write all numbers to this JSON file")
    c = sub.add_parser("compare", help="paired H1 difference between fits, CI over chains")
    c.add_argument("fits", nargs="+", help="two or more fits, as for score")
    c.add_argument("--spliced", action="store_true")
    c.add_argument("--xyz-order", default="xyz")
    c.add_argument("--references", default="PHercParis4")
    c.add_argument("--z", type=float, nargs=2, default=None, metavar=("Z0", "Z1"))
    c.add_argument("--tol", type=float, default=10.0)
    c.add_argument("--chains", default="all", choices=("all", "odd", "even"))
    c.add_argument("--boot", type=int, default=10000, help="bootstrap draws over chains")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--name", action="append", default=None,
                   help="label for one fit, in the order the fits are given (repeatable); "
                        "default is the shortest path suffix that separates them. The results "
                        "are keyed by it, so the labels must be distinct")
    c.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        from .data import fetch
        print(fetch(args.scroll))
        return 0

    if args.cmd == "compare":
        return _compare(args)

    from .evaluate import evaluate
    from .fit import check_seams, load_fit, read_grids, unique_names
    truth = fibers = ann = None
    if args.references:
        from windingcheck.chains import load_pcl
        from windingcheck.score import build_truth
        from .data import annotation_points, eval_fibers, scroll_dir
        d = scroll_dir(args.references)
        if not (d / "relative_windings.json").exists():
            sys.exit(f"no references under {d}; run: spiralfiteval fetch --scroll {args.references}")
        truth = build_truth(load_pcl(d / "relative_windings.json"))
        fibers = [x["xyz"] for x in eval_fibers(args.references, z_range=args.z)]
        ann = annotation_points(args.references)
    ladders = None
    if args.ladders:
        from .ladders import load_ladders
        if args.z is None:
            sys.exit("--ladders needs --z")
        ladders = load_ladders(args.ladders, args.z, nmax=args.ladder_nmax)
    results = []
    names = unique_names(args.fits, args.name)
    for src, nm in zip(args.fits, names):
        grids = read_grids(src, spliced=args.spliced, xyz_order=args.xyz_order)
        seams = check_seams(grids)
        fit = load_fit(src, spliced=args.spliced, xyz_order=args.xyz_order,
                       column_sense=seams["sense"], name=nm)
        del grids
        r = evaluate(fit, truth=truth, fibers=fibers, annotations=ann, ladders=ladders,
                     tol_n=args.tol, z_range=args.z, floor_seeds=args.floors,
                     chain_parity=args.chains)
        r["seams"] = seams
        print(_summary(r), flush=True)
        results.append(r)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=float))
    return 0


def _band_chain_ids(truth, z_range, parity):
    """The chain ids H1 would use: inside the band (every click), then of this parity."""
    from .paired import chain_parity_ids
    ids = None
    if z_range is not None:
        z = truth.chains.xyz[:, 2]
        inb = (z >= z_range[0]) & (z < z_range[1])
        ids = [int(c) for c in np.unique(truth.chains.chain[inb])
               if inb[truth.chains.chain == c].all()]
    return chain_parity_ids(truth, parity, ids)


def _compare(args) -> int:
    from windingcheck.chains import load_pcl
    from windingcheck.score import build_truth

    from .data import scroll_dir
    from .fit import check_seams, load_fit, read_grids, unique_names
    from .paired import paired_slips

    d = scroll_dir(args.references)
    if not (d / "relative_windings.json").exists():
        sys.exit(f"no references under {d}; run: spiralfiteval fetch --scroll {args.references}")
    truth = build_truth(load_pcl(d / "relative_windings.json"))
    ids = _band_chain_ids(truth, args.z, args.chains)
    fits = []
    names = unique_names(args.fits, args.name)
    for src, nm in zip(args.fits, names):
        grids = read_grids(src, spliced=args.spliced, xyz_order=args.xyz_order)
        seams = check_seams(grids)
        fits.append((nm, load_fit(src, spliced=args.spliced, xyz_order=args.xyz_order,
                                  column_sense=seams["sense"], name=nm)))
        del grids
    res = paired_slips(fits, truth, chain_ids=ids, tol_n=args.tol,
                       n_boot=args.boot, seed=args.seed)
    res["chain_parity"] = args.chains
    res["z"] = args.z
    print(f"paired H1, {args.chains} chains: {res['chains']} chains, "
          f"{res['common_adjacent_tests']} adjacent pairs covered by every fit, "
          f"{res['n_boot']} draws over chains")
    for name, v in res["own"].items():
        ci = v["slips_ci95"]
        print(f"  {name}: slips per wrap {v['slips_per_wrap']:.1%}"
              + (f" (95% CI {ci[0]:.1%}-{ci[1]:.1%})" if ci else "")
              + f" on its own {v['adjacent_tests']} pairs; common-test "
              f"{res['common'][name]['slips_per_wrap']:.1%}; clicks on fit {v['clicks_on_fit']:.0%}")
    for k, v in res["differences"].items():
        print(f"  {k}: {v['point']:+.1%} (95% CI {v['ci95'][0]:+.1%} to {v['ci95'][1]:+.1%}), "
              f"{'excludes' if v['excludes_zero'] else 'includes'} 0, "
              f"{v['share_below_zero']:.0%} of draws below 0")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
