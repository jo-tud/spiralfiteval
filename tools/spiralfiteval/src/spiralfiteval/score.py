"""The four spiral-fit metrics. Inputs are a loaded ``Fit`` and plain arrays.

H1  winding identity  - human relative-winding chains: is the fit's wrap count between two
                        clicks of one chain the human count?
H2  sheet switches    - along same-sheet traces (fibers, patch grid lines), how often does the
                        fit's spiral time jump by about one turn between neighbouring points?
H3  distance          - how far is each reference-surface point from the nearest fit sheet,
                        measured along the fit normal?
H5  sheet spacing     - no annotations needed: along each fit vertex's normal, how far is the
                        fit's *next* winding? Near zero means two windings sit on one sheet
                        (duplicated coverage, which an ink-pixel objective rewards).
H4  conformity        - H2/H3 split by distance to the nearest input the fit was given, so the
                        "does the fit hold up where there are no annotations" question has a
                        number.

Coordinates are xyz level-0 voxels (VC3D order) throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from windbench.metrics import densify
from windingcheck.score import Truth, _coverage, _metrics

from .fit import Fit


def winding_identity(fit: Fit, truth: Truth, *, tol_n: float = 10.0, chain_ids=None,
                     n_boot: int = 1000, seed: int = 0, return_tests=False) -> dict:
    """H1. Each click takes the spiral time of its fit sheet (the nearest one along the normal,
    within ``tol_n`` voxels; 10 is the convention of Jashann's slips-per-wrap metric). Each pair
    of clicks in one chain with both clicks on a fit sheet is a test: predicted Δw =
    round(f_b - f_a) against the human Δw. Clicks with no fit sheet in tolerance are uncovered,
    not wrong; coverage is reported next to the rates."""
    f, dn = fit.project(truth.chains.xyz, tol_n)
    ia, ib = truth.pairs()
    if chain_ids is not None:
        m = np.isin(truth.chains.chain[ia], np.asarray(list(chain_ids)))
        ia, ib = ia[m], ib[m]
    ok = np.isfinite(f[ia]) & np.isfinite(f[ib])
    ta, tb = ia[ok], ib[ok]
    true_dw = truth.chains.level[tb] - truth.chains.level[ta]
    pred_dw = np.rint(f[tb] - f[ta])
    res = _metrics(true_dw, pred_dw)
    cov = _coverage(truth, ta, tb)
    if chain_ids is not None:  # coverage over the selected chains only
        n_sel = int(len(ia))
        cov["human_pairs"] = n_sel
        cov["human_pairs_covered"] = float(ok.mean()) if n_sel else 0.0
        # the adjacent (Δw = 1) share is H1's own denominator, so it is recomputed over the
        # selection rather than dropped: in earlier versions this branch popped it and left only
        # the all-pairs share, which a caller could mistake for H1 coverage.
        adj_sel = (truth.chains.level[ib] - truth.chains.level[ia]) == 1
        cov["human_adjacent_pairs"] = int(adj_sel.sum())
        cov["human_adjacent_covered"] = float(ok[adj_sel].mean()) if adj_sel.any() else 0.0
    res.update(cov)
    clicks = np.isin(truth.chains.chain, np.asarray(list(chain_ids))) if chain_ids is not None \
        else np.ones(len(f), bool)
    res["clicks"] = int(clicks.sum())
    res["click_dn_median"] = float(np.nanmedian(dn[clicks])) if np.isfinite(dn[clicks]).any() else float("nan")
    # slips per wrap: adjacent human pairs (Δw = 1) with a wrong predicted Δw, CI over chains
    adj = true_dw == 1
    err = pred_dw[adj] != 1
    ch = truth.chains.chain[ta[adj]]
    res["adjacent_tests"] = int(adj.sum())
    res["slips_per_wrap"] = float(err.mean()) if adj.any() else float("nan")
    if adj.any() and n_boot:
        ids, inv = np.unique(ch, return_inverse=True)
        n_c = np.bincount(inv, minlength=len(ids))
        e_c = np.bincount(inv, weights=err, minlength=len(ids))
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, len(ids), size=(n_boot, len(ids)))
        rate = e_c[draw].sum(1) / np.maximum(n_c[draw].sum(1), 1)
        res["slips_ci95"] = [float(np.percentile(rate, 2.5)), float(np.percentile(rate, 97.5))]
    res["clicks_on_fit"] = float(np.isfinite(f[clicks]).mean()) if clicks.any() else float("nan")
    if return_tests:
        return res, {"a": ta, "b": tb, "true_dw": true_dw, "pred_dw": pred_dw, "f": f, "dn": dn}
    return res


def _switch_steps(p: np.ndarray, f: np.ndarray, *, jump: float, flicker: float,
                  bridge: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Switches along one trace. Returns (switch point indices, switch signs, voxels on fit).

    Points off the fit are skipped; a gap of at most ``bridge`` voxels of trace is bridged so a
    fit that leaves the sheet briefly and comes back on another wrap still counts. A jump
    (|Δf| > ``jump``) undone by an opposite jump within ``flicker`` voxels is dropped (both)."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    k = np.flatnonzero(np.isfinite(f))
    if len(k) < 2:
        return np.zeros(0, int), np.zeros(0, int), 0.0
    ds = np.diff(s[k])
    link = ds <= bridge + 1e-9
    df = np.diff(f[k])
    on = float(ds[link & (np.diff(k) == 1)].sum())
    is_j = link & (np.abs(df) > jump)
    idx = k[:-1][is_j]
    sg = np.sign(df[is_j]).astype(int)
    pos = s[idx]
    keep = np.ones(len(idx), bool)
    i = 0
    while i < len(idx) - 1:
        if keep[i] and sg[i + 1] == -sg[i] and pos[i + 1] - pos[i] <= flicker:
            keep[i] = keep[i + 1] = False
            i += 2
        else:
            i += 1
    return idx[keep], sg[keep], on


def trace_switches(fit: Fit, traces, *, tol_n: float = 10.0, max_step: float = 1.0,
                   jump: float = 0.5, flicker: float = 10.0, bridge: float = 10.0,
                   clean_dn: float = 3.0) -> dict:
    """H2. Densify each trace to <= ``max_step`` voxel steps (windbench convention), give every
    point the spiral time of its fit sheet (within ``tol_n``), and count switches: jumps of
    |Δf| > ``jump`` (half a turn) that are not undone within ``flicker`` voxels. Seams between
    windings do not count: f is continuous across them.

    Returns switches, voxels (trace length on the fit), per_1000_vox, coverage (share of trace
    points on a fit sheet), traces_with_switch, and switch sites (xyz)."""
    switches = clean = 0
    vox = on_pts = total = 0.0
    traced = with_sw = 0
    sites = []
    for tr in traces:
        p = densify(np.asarray(tr, np.float64), max_step)
        f, dn = fit.project(p, tol_n)
        total += len(f)
        on_pts += int(np.isfinite(f).sum())
        idx, _, on = _switch_steps(p, f, jump=jump, flicker=flicker, bridge=bridge)
        if on <= 0:
            continue
        traced += 1
        vox += on
        switches += len(idx)
        clean += _clean(idx, dn, p, window=flicker, near=clean_dn)
        if len(idx):
            with_sw += 1
            sites.append(p[idx])
    return {"switches": switches, "voxels": vox,
            "per_1000_vox": 1000.0 * switches / vox if vox else float("nan"),
            "coverage": on_pts / total if total else float("nan"),
            "traces_scored": traced,
            "traces_with_switch": with_sw / traced if traced else float("nan"),
            "sites": np.concatenate(sites) if sites else np.zeros((0, 3)),
            # added after the first trial run, not part of the original metric set: only
            # switches with the fit within ``clean_dn`` of the trace on both sides, so a fit
            # lying half a wrap off the trace does not count
            "post_hoc_clean_switches": clean,
            "post_hoc_clean_per_1000_vox": 1000.0 * clean / vox if vox else float("nan")}


def _clean(idx, dn, p, *, window: float, near: float) -> int:
    """Switches whose median normal distance over ``window`` voxels before and after is
    <= ``near`` on both sides."""
    if len(idx) == 0:
        return 0
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    n = 0
    for i in idx:
        before = dn[(s >= s[i] - window) & (s <= s[i])]
        after = dn[(s > s[i]) & (s <= s[i] + window)]
        if (np.isfinite(before).any() and np.isfinite(after).any()
                and np.nanmedian(before) <= near and np.nanmedian(after) <= near):
            n += 1
    return n


def surface_distance(fit: Fit, pts: np.ndarray, *, tol_n: float = 20.0) -> dict:
    """H3. Normal distance from each reference point to the nearest fit sheet (within
    ``tol_n``; farther points count as missed). Reports the median, p90, share within 2 and 4
    voxels (over all points: missed points are outside), and the share missed."""
    f, dn = fit.project(pts, tol_n)
    n = len(dn)
    hit = np.isfinite(dn)
    out = {"n": n, "missed": float(1 - hit.mean()) if n else float("nan")}
    if hit.any():
        d = dn[hit]
        out.update(median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                   within_2=float((dn <= 2).sum() / n), within_4=float((dn <= 4).sum() / n))
    return out


def evidence_distance(pts: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest fit input point (voxels)."""
    if len(evidence) == 0:
        return np.full(len(pts), np.inf)
    d, _ = cKDTree(np.asarray(evidence, np.float64)).query(np.asarray(pts, np.float64), k=1)
    return d


def by_evidence_distance(values: np.ndarray, dist: np.ndarray, bins) -> list[dict]:
    """H4 helper: mean of a per-point 0/1 (or real) value in distance-to-evidence bins."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dist >= lo) & (dist < hi) & np.isfinite(values)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "mean": float(values[m].mean()) if m.any() else float("nan")})
    return out


def sheet_spacing(fit: Fit, *, n_sample: int = 200_000, max_gap: float = 60.0,
                  dup: float = 2.0, seed: int = 0) -> dict:
    """H5. For a random sample of fit vertices, the normal distance to the nearest vertex of a
    different winding (|Δf| >= 0.5), up to ``max_gap``. Reports the gap median / p10, the share
    of vertices with a neighbouring winding closer than ``dup`` voxels (duplicated or crossing
    windings), and the share with no neighbour within ``max_gap``. The search widens past the
    vertex's own sheet until the answer is exact (before commit 3d32d3f it stopped at 48 neighbours,
    which cut off gaps above ~16 voxels and inflated the no-neighbour share)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fit.f), size=min(n_sample, len(fit.f)), replace=False)
    _, gap = fit.project(fit.xyz[idx], max_gap, exclude_f=fit.f[idx])
    has = np.isfinite(gap)
    out = {"n": int(len(idx)), "no_neighbour": float(1 - has.mean()),
           "duplicate": float((gap < dup).sum() / len(idx)), "dup_threshold": dup,
           "max_gap": max_gap}
    if has.any():
        out.update(median=float(np.median(gap[has])), p10=float(np.percentile(gap[has], 10)))
    out["_gap"] = gap
    out["_idx"] = idx
    return out


def sheet_spacing_by_winding(fit: Fit, *, per_winding: int = 2000, max_gap: float = 60.0,
                             dup: float = 2.0, seed: int = 0, blocks=None,
                             fit_median_gap: float | None = None) -> dict:
    """H5 per winding: the same gap, on up to ``per_winding`` vertices drawn from each winding
    (so small inner windings are not drowned by the large outer ones). Returns a row per
    winding (n, duplicate share, median gap, median radius from the axis), the duplicate share
    as a plain mean over windings, the share of the area-weighted duplicates that falls in each
    ``blocks`` range, and the outermost run of windings whose median gap is under half the fit's
    median gap (``fit_median_gap``; windings stacked on the outer shell)."""
    from .floors import axis_xy
    rng = np.random.default_rng(seed)
    rows = {}
    for w in sorted(set(int(x) for x in np.unique(fit.winding))):
        ids = np.flatnonzero(fit.winding == w)
        idx = rng.choice(ids, size=min(per_winding, len(ids)), replace=False)
        _, gap = fit.project(fit.xyz[idx], max_gap, exclude_f=fit.f[idx], k=48)
        has = np.isfinite(gap)
        r = np.linalg.norm(fit.xyz[idx, :2] - axis_xy(fit, fit.xyz[idx, 2]), axis=1)
        rows[w] = {"n": int(len(idx)), "vertices": int(len(ids)),
                   "duplicate": float((gap < dup).mean()),
                   "median_gap": float(np.median(gap[has])) if has.any() else float("nan"),
                   "median_radius": float(np.median(r))}
    dups = np.array([r["duplicate"] for r in rows.values()])
    area = np.array([r["vertices"] for r in rows.values()], np.float64)
    out = {"per_winding": rows, "mean_over_windings": float(dups.mean()),
           "area_weighted": float((dups * area).sum() / area.sum())}
    dup_area = dups * area
    if blocks:
        ws = np.array(list(rows))
        out["share_of_duplicates"] = {
            f"w{lo:03d}-{hi:03d}": float(dup_area[(ws >= lo) & (ws <= hi)].sum() / max(dup_area.sum(), 1e-12))
            for lo, hi in blocks}
        out["duplicate_in_block"] = {
            f"w{lo:03d}-{hi:03d}": float(dup_area[(ws >= lo) & (ws <= hi)].sum()
                                         / max(area[(ws >= lo) & (ws <= hi)].sum(), 1e-12))
            for lo, hi in blocks}
    if fit_median_gap:
        run = []
        for w in sorted(rows, reverse=True):
            if rows[w]["median_gap"] < 0.5 * fit_median_gap:
                run.append(w)
            else:
                break
        ws = np.array(list(rows))
        m = np.isin(ws, run)
        out["stacked_outer"] = {"windings": [min(run), max(run)] if run else None, "n": len(run),
                                "share_of_duplicates": float(dup_area[m].sum() / max(dup_area.sum(), 1e-12))}
    return out
