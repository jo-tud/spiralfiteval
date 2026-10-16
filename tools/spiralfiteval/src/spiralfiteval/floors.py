"""Floors: what each reference metric scores when the reference carries no sheet information.

A metric computed from reference points (human clicks, eval fibers) is only informative if the
true points score clearly better than the same points moved off their sheet. Two checks, the
same for every metric:

- **Shift curve.** Move the whole reference set radially (in xy, away from the fit's axis) by
  s voxels and score again. A fit that sits on the reference sheets has its best score at s = 0
  and gets worse by half a sheet spacing; a flat curve means the metric is at its floor.
- **Phase jitter.** Move each reference unit (a click, a trace, or a single point) by an
  independent radial offset drawn uniformly from ± half the fit's median sheet gap (H5), so its
  position *between* sheets is random but its wrap is kept on average. Repeated over seeds;
  the mean is the floor, the spread across seeds its noise.

H3 also gets the **random-point floor**: uniform points in the bounding box of the references.

**Near-click registration** (added after a code review): over the whole fiber set H3 sits at its
floor, so a frame offset between the fibers and the fit cannot show. Close to the human clicks a
fit that took them sits on the clicked sheet, so there the shift curve of fiber points can find a
sub-sheet offset between the two references.
"""

from __future__ import annotations

import numpy as np

from .fit import Fit

SHIFTS = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)
NEAR_SHIFTS = tuple(float(s) for s in range(-8, 9))


def axis_xy(fit: Fit, z: np.ndarray, *, bin_z: float = 50.0) -> np.ndarray:
    """Scroll axis at each height: the xy centroid of the fit's innermost winding, in z bins
    (linearly interpolated, held constant past the ends)."""
    m = fit.winding == min(fit.windings)
    zz, xy = fit.xyz[m, 2].astype(np.float64), fit.xyz[m, :2].astype(np.float64)
    edges = np.arange(np.floor(zz.min()), zz.max() + bin_z, bin_z)
    b = np.digitize(zz, edges) - 1
    cz, cx, cy = [], [], []
    for k in np.unique(b):
        s = b == k
        if s.sum() >= 10:
            cz.append(zz[s].mean())
            cx.append(xy[s, 0].mean())
            cy.append(xy[s, 1].mean())
    z = np.asarray(z, np.float64)
    if not cz:
        return np.broadcast_to(xy.mean(0), (len(z), 2)).copy()
    return np.stack([np.interp(z, cz, cx), np.interp(z, cz, cy)], axis=1)


def radial(fit: Fit, pts: np.ndarray) -> np.ndarray:
    """Unit xy vectors from the fit's axis to each point (z component 0)."""
    pts = np.asarray(pts, np.float64)
    v = pts[:, :2] - axis_xy(fit, pts[:, 2])
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return np.c_[v, np.zeros(len(v))]


def shift(pts, dirs, s):
    return np.asarray(pts, np.float64) + s * dirs


def jitter(pts, dirs, half, rng, groups=None):
    """Independent uniform radial offset in [-half, half] per point, or per group label."""
    if groups is None:
        off = rng.uniform(-half, half, len(pts))
    else:
        _, inv = np.unique(groups, return_inverse=True)
        off = rng.uniform(-half, half, inv.max() + 1)[inv]
    return np.asarray(pts, np.float64) + off[:, None] * dirs


def random_box(pts, n, rng):
    lo, hi = np.min(pts, 0), np.max(pts, 0)
    return rng.uniform(lo, hi, (n, 3))


def summarise(value: float, floor_values, *, better: str) -> dict:
    """Value next to its floor. ``margin`` > 0 means better than the floor."""
    fv = np.asarray(floor_values, np.float64)
    mean = float(np.nanmean(fv))
    sd = float(np.nanstd(fv, ddof=1)) if np.isfinite(fv).sum() > 1 else float("nan")
    margin = (mean - value) if better == "lower" else (value - mean)
    return {"value": float(value), "floor_mean": mean, "floor_sd": sd, "floor_values": fv.tolist(),
            "margin": float(margin), "better": better}


def near_reference_shift(fit: Fit, pts, ref, *, radius: float = 15.0, tol_n: float = 10.0,
                         shifts=NEAR_SHIFTS) -> dict:
    """Shift curve of the H3 median for the points of ``pts`` within ``radius`` voxels of a
    point of ``ref`` (the human clicks), next to the clicks' own median distance to the fit.
    A fit on the clicked sheet with fibers registered to it is lowest at s = 0."""
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, np.float64)
    ref = np.asarray(ref, np.float64)
    d, _ = cKDTree(ref).query(pts, k=1, distance_upper_bound=radius)
    near = pts[np.isfinite(d)]
    _, dref = fit.project(ref, tol_n)
    out = {"radius": radius, "n": int(len(near)),
           "click_median": float(np.nanmedian(dref)) if np.isfinite(dref).any() else float("nan")}
    if len(near) == 0:
        return out
    dirs = radial(fit, near)
    med = {}
    for s in shifts:
        _, dn = fit.project(shift(near, dirs, s), tol_n)
        med[str(s)] = float(np.nanmedian(dn)) if np.isfinite(dn).any() else float("nan")
    best = min(med, key=lambda k: med[k] if np.isfinite(med[k]) else np.inf)
    out.update(median=med, best_shift=float(best), at_0=med.get("0.0", float("nan")),
               at_best=med[best])
    return out
