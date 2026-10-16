"""Winding metrics. Pure numpy/scipy, no I/O, so spiralfiteval and other scorers can call them on any arrays.

Conventions
-----------
- Volumes are (z, y, x) arrays in one crop's voxel grid.
- Wrap-index volumes are integers with UNKNOWN (-1) for "no label" (GT) or "no prediction".
- Traces are (N, 3) float arrays of (z, y, x) voxel coordinates in the same grid.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

UNKNOWN = -1


def wrap_accuracy(k_pred: np.ndarray, k_gt: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Wrap-index accuracy up to one global offset.

    Over GT-known voxels (k_gt != UNKNOWN, and `mask` if given):
    acc = max_c mean(k_pred - k_gt == c). Voxels with no prediction count as wrong.
    Returns acc, the best offset c, coverage (share of evaluated voxels with a prediction), n.
    """
    if k_pred.shape != k_gt.shape:
        raise ValueError(f"shape mismatch {k_pred.shape} vs {k_gt.shape}")
    sel = k_gt != UNKNOWN
    if mask is not None:
        sel &= mask
    n = int(sel.sum())
    if n == 0:
        return {"acc": float("nan"), "offset": None, "coverage": float("nan"), "n": 0}
    kp = k_pred[sel].astype(np.int64)
    kg = k_gt[sel].astype(np.int64)
    has = kp != UNKNOWN
    if not has.any():
        return {"acc": 0.0, "offset": None, "coverage": 0.0, "n": n}
    diff = kp[has] - kg[has]
    vals, counts = np.unique(diff, return_counts=True)
    best = int(np.argmax(counts))
    return {"acc": float(counts[best]) / n, "offset": int(vals[best]),
            "coverage": float(has.sum()) / n, "n": n}


def densify(trace: np.ndarray, max_step: float = 1.0) -> np.ndarray:
    """Insert points so that consecutive points are at most `max_step` voxels apart."""
    t = np.asarray(trace, dtype=np.float64)
    if len(t) < 2:
        return t
    seg = np.linalg.norm(np.diff(t, axis=0), axis=1)
    reps = np.maximum(1, np.ceil(seg / max_step).astype(int))
    out = [t[:1]]
    for a, b, r in zip(t[:-1], t[1:], reps):
        s = (np.arange(1, r + 1) / r)[:, None]
        out.append(a + s * (b - a))
    return np.concatenate(out)


def _lookup(vol: np.ndarray, pts: np.ndarray) -> np.ndarray:
    idx = np.rint(pts).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
    out = np.full(len(pts), UNKNOWN, dtype=np.int64)
    out[inside] = vol[idx[inside, 0], idx[inside, 1], idx[inside, 2]]
    return out


def switch_rate(traces: list[np.ndarray], k_volume: np.ndarray, max_step: float = 1.0) -> dict:
    """Sheet switches per 1000 steps along traces.

    Each trace is densified to <= `max_step` voxel steps; the wrap index is looked up at the
    nearest voxel. A step counts when both ends are known; it is a switch when the index changes.
    Use (GT traces, k_pred) to score a k-volume, or (predicted-surface traces, k_gt) to score a
    surface. Returns switches, steps, per_1000.
    """
    switches = steps = 0
    for tr in traces:
        k = _lookup(k_volume, densify(tr, max_step))
        ok = (k[:-1] != UNKNOWN) & (k[1:] != UNKNOWN)
        steps += int(ok.sum())
        switches += int((ok & (k[:-1] != k[1:])).sum())
    return {"switches": switches, "steps": steps,
            "per_1000": 1000.0 * switches / steps if steps else float("nan")}


def _components(pred) -> list[np.ndarray]:
    """Predicted defects -> list of point sets. Bool volume: 26-connected components. Points: one each."""
    if isinstance(pred, np.ndarray) and pred.dtype == bool:
        lab, n = ndimage.label(pred, structure=np.ones((3, 3, 3), bool))
        if n == 0:
            return []
        idx = np.argwhere(lab)
        order = np.argsort(lab[tuple(idx.T)], kind="stable")
        idx = idx[order]
        cuts = np.cumsum(np.bincount(lab[tuple(idx.T)], minlength=n + 1)[1:])[:-1]
        return np.split(idx.astype(np.float64), cuts)
    pts = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    return [p[None] for p in pts]


def defect_pr(pred, site_labels: np.ndarray, site_types: dict[int, str] | None = None,
              r: float = 20.0) -> dict:
    """Defect precision / recall against GT error sites.

    pred: bool volume (defect voxels) or (M, 3) points, crop voxel coords.
    site_labels: int volume, 0 = no site, i > 0 = site id i.
    site_types: optional {site id: type} for per-type recall.
    recall = share of sites with a predicted defect within r of any site voxel.
    precision = share of predicted components within r of any site voxel.
    """
    comps = _components(pred)
    site_idx = np.argwhere(site_labels > 0)
    site_ids = site_labels[tuple(site_idx.T)] if len(site_idx) else np.zeros(0, int)
    all_sites = np.unique(site_ids)
    res = {"n_sites": int(len(all_sites)), "n_pred": len(comps), "r": r}
    if len(all_sites) == 0 or len(comps) == 0:
        res.update(recall=float("nan") if len(all_sites) == 0 else 0.0,
                   precision=float("nan") if len(comps) == 0 else 0.0, per_type={})
        return res
    pred_pts = np.concatenate(comps)
    d_site, _ = cKDTree(pred_pts).query(site_idx, k=1, distance_upper_bound=r)
    hit_sites = np.unique(site_ids[np.isfinite(d_site)])
    site_tree = cKDTree(site_idx)
    tp_pred = sum(bool(np.isfinite(site_tree.query(c, k=1, distance_upper_bound=r)[0]).any()) for c in comps)
    res["recall"] = len(hit_sites) / len(all_sites)
    res["precision"] = tp_pred / len(comps)
    per_type = {}
    if site_types:
        for t in sorted(set(site_types.values())):
            ids = [i for i in all_sites if site_types.get(int(i)) == t]
            if ids:
                per_type[t] = {"n": len(ids), "recall": float(np.isin(ids, hit_sites).mean())}
    res["per_type"] = per_type
    return res
