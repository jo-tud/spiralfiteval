"""Run every metric that the available references allow on one fit."""

from __future__ import annotations

import time

import numpy as np

from .fit import Fit
from .ladders import ladder_slips, windings_per_wrap
from .score import (by_evidence_distance, evidence_distance, sheet_spacing, sheet_spacing_by_winding,
                    trace_switches, winding_identity, _switch_steps)
from windbench.metrics import densify

H4_BINS = (0, 25, 50, 100, 200, 400, np.inf)
# winding blocks for where duplicates sit: core, middle, and the outer windings above PHerc0826's
# count of about 90 (villa PR #2048), split at 110
H5_BLOCKS = ((10, 29), (30, 59), (60, 90), (91, 109), (110, 129))


def evaluate(fit: Fit, *, truth=None, fibers=None, annotations=None, ladders=None,
             tol_n: float = 10.0, z_range=None, keep_arrays: bool = False,
             floor_seeds: int = 0, h5_blocks=H5_BLOCKS, chain_parity: str = "all") -> dict:
    """``truth``: windingcheck Truth (human chains); ``fibers``: list of (N, 3) same-sheet
    traces; ``annotations``: (M, 3) human annotation points for H4; ``ladders``: a published
    ladder set (``ladders.load_ladders``) for H1 next to the published numbers. Any of them may
    be None; H5 needs nothing. Clicks are restricted to ``z_range`` (the fit's band) if given.
    ``floor_seeds`` > 0 adds every reference metric's floor and shift curve (``floors``).
    ``chain_parity`` ("all", "odd", "even") keeps only chains of that parity for H1:
    "odd" is windingcheck's held-out half, the chains its shipped models never saw."""
    t0 = time.time()
    res = {"fit": fit.name, "vertices": int(len(fit.f)), "windings": [min(fit.windings), max(fit.windings)],
           "grid_step_vox": fit.step}
    arrays = {}
    ids = None
    if truth is not None:
        if z_range is not None:
            z = truth.chains.xyz[:, 2]
            inb = (z >= z_range[0]) & (z < z_range[1])
            ids = np.unique(truth.chains.chain[inb])
            # a chain counts if all its clicks are in the band (radial chains live in one slice)
            ids = [c for c in ids if inb[truth.chains.chain == c].all()]
        if chain_parity != "all":
            from .paired import chain_parity_ids
            ids = chain_parity_ids(truth, chain_parity, ids)
        h1, tests = winding_identity(fit, truth, tol_n=tol_n, chain_ids=ids, return_tests=True)
        h1["chains"] = len(ids) if ids is not None else int(truth.chains.n_chains)
        h1["chain_parity"] = chain_parity
        res["H1"] = h1
        arrays["h1"] = tests
    if ladders:
        res["H1_ladders"] = ladder_slips(fit, ladders, tol_n=tol_n)
        res["windings_per_wrap"] = windings_per_wrap(fit, ladders, tol_n=tol_n, blocks=h5_blocks)
    if fibers is not None and len(fibers):
        h2 = trace_switches(fit, fibers, tol_n=tol_n)
        arrays["switch_sites"] = h2.pop("sites")
        res["H2"] = h2
        pts = np.concatenate([densify(np.asarray(t, np.float64), 1.0) for t in fibers])
        f, dn = fit.project(pts, tol_n)
        res["H3"] = surface_distance_from(dn)
        arrays.update(fiber_pts=pts, fiber_dn=dn)
        if annotations is not None and len(annotations):
            d_ann = evidence_distance(pts, annotations)
            within4 = np.where(np.isfinite(dn) & (dn <= 4), 1.0, 0.0)
            res["H4"] = {"bins": [list(map(float, H4_BINS[:-1])), [float(b) for b in H4_BINS[1:]]],
                         "within_4": by_evidence_distance(within4, d_ann, H4_BINS),
                         "switches_per_1000_vox": _switches_by_bin(fit, fibers, annotations, tol_n)}
            arrays["fiber_d_ann"] = d_ann
    h5 = sheet_spacing(fit)
    arrays["h5_gap"], arrays["h5_idx"] = h5.pop("_gap"), h5.pop("_idx")
    res["H5"] = h5
    res["H5_by_winding"] = sheet_spacing_by_winding(fit, blocks=h5_blocks,
                                                    fit_median_gap=h5.get("median"))
    if floor_seeds:
        res["floors"] = floors(fit, truth=truth, chain_ids=ids if truth is not None else None,
                               fibers=fibers, annotations=annotations, ladders=ladders,
                               gap=h5["median"], tol_n=tol_n, seeds=floor_seeds)
    res["seconds"] = round(time.time() - t0, 1)
    if keep_arrays:
        res["_arrays"] = arrays
    return res


def surface_distance_from(dn: np.ndarray) -> dict:
    n = len(dn)
    hit = np.isfinite(dn)
    out = {"n": n, "not_on_fit": float(1 - hit.mean()) if n else float("nan"),
           "within_2": float((dn <= 2).sum() / n) if n else float("nan"),
           "within_4": float((dn <= 4).sum() / n) if n else float("nan")}
    if hit.any():
        out.update(median=float(np.median(dn[hit])), p90=float(np.percentile(dn[hit], 90)))
    return out


def _switches_by_bin(fit: Fit, fibers, annotations, tol_n, bins=H4_BINS):
    """H2 split by the distance of each trace voxel to the nearest annotation."""
    sw = np.zeros(len(bins) - 1)
    vox = np.zeros(len(bins) - 1)
    for tr in fibers:
        p = densify(np.asarray(tr, np.float64), 1.0)
        f, _ = fit.project(p, tol_n)
        d = evidence_distance(p, annotations)
        b = np.clip(np.digitize(d, bins) - 1, 0, len(bins) - 2)
        idx, _, _ = _switch_steps(p, f, jump=0.5, flicker=10.0, bridge=10.0)
        np.add.at(sw, b[idx], 1)
        ok = np.isfinite(f[:-1]) & np.isfinite(f[1:])
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        np.add.at(vox, b[:-1][ok], seg[ok])
    return [{"lo": float(lo), "hi": float(hi), "voxels": float(v),
             "per_1000_vox": float(1000 * s / v) if v else float("nan")}
            for lo, hi, s, v in zip(bins[:-1], bins[1:], sw, vox)]


def floors(fit: Fit, *, truth=None, chain_ids=None, fibers=None, annotations=None,
           ladders=None, gap: float, tol_n: float = 10.0, seeds: int = 5,
           shifts=None) -> dict:
    """Every reference metric next to its floor (see ``floors.py``): the score of the references
    shifted radially by each of ``shifts`` voxels, and the mean and spread over ``seeds`` draws
    of a radial phase jitter of ± ``gap`` / 2 (``gap`` = the fit's median H5 gap). H3 also gets
    uniform random points in the fibers' bounding box. H1 is jittered per click, H2 per trace
    (a trace moves as a whole, so its own continuity is kept), H3 per point."""
    from dataclasses import replace
    from windingcheck.score import build_truth
    from . import floors as F
    from .ladders import ladder_slips
    from .score import surface_distance

    shifts = F.SHIFTS if shifts is None else shifts
    half = gap / 2
    out = {"jitter_half_vox": half, "seeds": seeds, "shifts": list(shifts)}
    if truth is not None:
        ch = truth.chains
        dirs = F.radial(fit, ch.xyz)

        def h1(xyz):
            t = build_truth(replace(ch, xyz=xyz))
            return winding_identity(fit, t, tol_n=tol_n, chain_ids=chain_ids, n_boot=0)["slips_per_wrap"]
        val = h1(ch.xyz)
        out["H1"] = {"shift": {str(s): h1(F.shift(ch.xyz, dirs, s)) for s in shifts},
                     "jitter": F.summarise(val, [h1(F.jitter(ch.xyz, dirs, half, np.random.default_rng(k)))
                                                 for k in range(seeds)], better="lower")}
    if ladders:
        xyz = np.concatenate([l["xyz"] for l in ladders])
        dirs = F.radial(fit, xyz)

        def h1l(p):
            return ladder_slips(fit, ladders, f=fit.project(p, tol_n)[0], n_boot=0)["tool"]["rate"]
        val = h1l(xyz)
        out["H1_ladders"] = {"shift": {str(s): h1l(F.shift(xyz, dirs, s)) for s in shifts},
                             "jitter": F.summarise(val, [h1l(F.jitter(xyz, dirs, half, np.random.default_rng(k)))
                                                         for k in range(seeds)], better="lower")}
    if fibers is not None and len(fibers):
        tr = [np.asarray(t, np.float64) for t in fibers]
        dirs = [F.radial(fit, t) for t in tr]

        def h2(trs):
            return trace_switches(fit, trs, tol_n=tol_n)["per_1000_vox"]

        def tj(rng):
            return [t + rng.uniform(-half, half) * d for t, d in zip(tr, dirs)]
        out["H2"] = {"shift": {str(s): h2([t + s * d for t, d in zip(tr, dirs)]) for s in shifts},
                     "jitter": F.summarise(h2(tr), [h2(tj(np.random.default_rng(k))) for k in range(seeds)],
                                           better="lower")}
        pts = np.concatenate([densify(t, 1.0) for t in tr])
        pd = F.radial(fit, pts)
        d0 = surface_distance(fit, pts, tol_n=tol_n)
        sh = {str(s): surface_distance(fit, F.shift(pts, pd, s), tol_n=tol_n) for s in shifts}
        jit = [surface_distance(fit, F.jitter(pts, pd, half, np.random.default_rng(k)), tol_n=tol_n)
               for k in range(seeds)]
        rnd = [surface_distance(fit, F.random_box(pts, 200_000, np.random.default_rng(100 + k)), tol_n=tol_n)
               for k in range(seeds)]
        med = {s: r.get("median", np.nan) for s, r in sh.items()}
        best = min(med, key=lambda s: med[s] if np.isfinite(med[s]) else np.inf)
        out["H3"] = {"shift_median": med,
                     "shift_within_4": {s: r.get("within_4", np.nan) for s, r in sh.items()},
                     "best_shift": float(best),
                     "rise_at_4": float(np.nanmean([med.get("-4.0", np.nan), med.get("4.0", np.nan)])
                                        - med.get("0.0", np.nan)),
                     "jitter": F.summarise(d0["median"], [r["median"] for r in jit], better="lower"),
                     "jitter_within_4": F.summarise(d0["within_4"], [r["within_4"] for r in jit], better="higher"),
                     "random": F.summarise(d0["median"], [r["median"] for r in rnd], better="lower"),
                     "random_within_4": F.summarise(d0["within_4"], [r["within_4"] for r in rnd], better="higher")}
        if truth is not None:
            clicks = truth.chains.xyz if chain_ids is None else \
                truth.chains.xyz[np.isin(truth.chains.chain, np.asarray(list(chain_ids)))]
            out["near_clicks"] = F.near_reference_shift(fit, pts, clicks, tol_n=tol_n)
        if annotations is not None and len(annotations):
            q = F.jitter(pts, pd, half, np.random.default_rng(0))
            _, dq = fit.project(q, tol_n)
            out["H4_jitter_within_4"] = by_evidence_distance(
                np.where(np.isfinite(dq) & (dq <= 4), 1.0, 0.0), evidence_distance(pts, annotations), H4_BINS)
    return out
