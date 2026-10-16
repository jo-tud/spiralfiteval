"""Local H5 -> H1: are winding slips more frequent where the fit duplicates a winding?

Unit of analysis: an H1 slip test, i.e. a pair of clicks in one human chain that the human
labels one wrap apart (Δw = 1), with both clicks on a fit sheet. The pair **slips** if the fit's
predicted Δw = round(f_b - f_a) is not 1; an **over-count** is a predicted Δw >= 2 (an extra
fit winding between the clicks, what a duplicated winding produces), an **under-count** <= 0.

Exposure: the **local duplicate share** D of the pair, i.e. the H5 duplicate flag (normal gap to
another winding < 2 voxels) averaged over fit vertices within ``radius`` voxels of the straight
segment between the two clicks. At most ``cap`` vertices per segment are scored, chosen by a
fixed random key per vertex, so the same vertex is picked whenever two segments overlap.

Statistic: the AUROC of D for slip vs no slip (ties count one half), per fit.

Null: **rotation about the scroll axis.** Every click is turned by the same angle φ about the
fit's axis (``floors.axis_xy``) at its own height, so each segment keeps its radius, height and
length (and so its winding block, where duplicates pile up) but lands on another part of the
sheet. D is measured around the turned segment and scored against the pair's own slip label.
One φ per draw, the same for every pair and every fit, so the spatial clustering of clicks and
of duplicates is kept. Pairs with fewer than ``min_vertices`` fit vertices in the tube are not
scored in that configuration (the same rule for the real and the turned positions).
"""

from __future__ import annotations

import numpy as np

from .fit import Fit
from .floors import axis_xy
from .score import winding_identity


def slip_tests(fit: Fit, truth, *, chain_ids=None, tol_n: float = 10.0) -> dict:
    """The H1 slip tests (human Δw = 1, both clicks on the fit) with their outcomes."""
    _, t = winding_identity(fit, truth, tol_n=tol_n, chain_ids=chain_ids, n_boot=0,
                            return_tests=True)
    adj = t["true_dw"] == 1
    a, b, pred = t["a"][adj], t["b"][adj], t["pred_dw"][adj]
    return {"a": a, "b": b, "pred_dw": pred, "slip": pred != 1, "over": pred >= 2,
            "under": pred <= 0, "chain": truth.chains.chain[a],
            "A": truth.chains.xyz[a].astype(np.float64), "B": truth.chains.xyz[b].astype(np.float64)}


class GapCache:
    """Exact H5 gap per fit vertex, computed on first use (``Fit.project`` with ``exclude_f``)."""

    def __init__(self, fit: Fit, *, max_gap: float = 60.0, dup: float = 2.0):
        self.fit, self.max_gap, self.dup = fit, max_gap, dup
        self.gap = np.full(len(fit.f), np.nan, np.float32)
        self.done = np.zeros(len(fit.f), bool)

    def __call__(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, np.int64)
        new = np.unique(idx[~self.done[idx]])
        if len(new):
            _, g = self.fit.project(self.fit.xyz[new], self.max_gap, exclude_f=self.fit.f[new])
            self.gap[new] = g
            self.done[new] = True
        return self.gap[idx].astype(np.float64)

    @property
    def computed(self) -> int:
        return int(self.done.sum())

    def duplicate(self, idx) -> np.ndarray:
        g = self(idx)
        return np.isfinite(g) & (g < self.dup)


def vertex_key(n: int, seed: int = 0) -> np.ndarray:
    """A fixed random priority per vertex (lower = picked first when a tube is capped)."""
    return np.random.default_rng(seed).random(n).astype(np.float32)


def tube(fit: Fit, A: np.ndarray, B: np.ndarray, radius: float) -> np.ndarray:
    """Indices of fit vertices within ``radius`` of the segment A-B."""
    L = float(np.linalg.norm(B - A))
    n = max(2, int(np.ceil(L / (0.5 * radius))) + 1)
    probes = A + np.linspace(0, 1, n)[:, None] * (B - A)
    reach = np.hypot(radius, 0.5 * L / (n - 1)) + 1e-6
    cand = np.unique(np.concatenate([np.asarray(i, np.int64)
                                     for i in fit.tree.query_ball_point(probes, reach)]))
    if not len(cand):
        return cand
    P = fit.xyz[cand].astype(np.float64)
    d = B - A
    s = np.clip(((P - A) @ d) / max(d @ d, 1e-12), 0, 1)
    dist = np.linalg.norm(P - (A + s[:, None] * d), axis=1)
    return cand[dist <= radius]


def local_duplicate_share(fit: Fit, A: np.ndarray, B: np.ndarray, radius: float, gaps: GapCache,
                          key: np.ndarray, *, cap: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """(D, n): local duplicate share and the number of scored vertices, per segment."""
    tubes = []
    for i in range(len(A)):
        idx = tube(fit, A[i], B[i], radius)
        if len(idx) > cap:
            idx = idx[np.argsort(key[idx], kind="stable")[:cap]]
        tubes.append(idx)
    n = np.array([len(t) for t in tubes], np.int64)
    D = np.full(len(A), np.nan)
    if n.sum():
        dup = gaps.duplicate(np.concatenate(tubes))  # one batched gap search per configuration
        ends = np.cumsum(n)
        for i, (s, e) in enumerate(zip(ends - n, ends)):
            if e > s:
                D[i] = float(dup[s:e].mean())
    return D, n


def rotate(fit: Fit, pts: np.ndarray, phi: float) -> np.ndarray:
    """Turn points by ``phi`` radians about the fit's axis at their own height."""
    pts = np.asarray(pts, np.float64)
    c = axis_xy(fit, pts[:, 2])
    v = pts[:, :2] - c
    cs, sn = np.cos(phi), np.sin(phi)
    out = pts.copy()
    out[:, 0] = c[:, 0] + cs * v[:, 0] - sn * v[:, 1]
    out[:, 1] = c[:, 1] + sn * v[:, 0] + cs * v[:, 1]
    return out


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a positive > score of a negative) + 1/2 P(tie). nan without both classes."""
    score = np.asarray(score, np.float64)
    label = np.asarray(label, bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def exposure(fit: Fit, tests: dict, phis, *, radius: float, gaps: GapCache, key: np.ndarray,
             cap: int = 64) -> dict:
    """D and tube size for every pair at every rotation angle (φ = 0 is the real position)."""
    D, N = [], []
    for phi in phis:
        A = tests["A"] if phi == 0 else rotate(fit, tests["A"], phi)
        B = tests["B"] if phi == 0 else rotate(fit, tests["B"], phi)
        d, n = local_duplicate_share(fit, A, B, radius, gaps, key, cap=cap)
        D.append(d)
        N.append(n)
    return {"phis": np.asarray(phis, np.float64), "D": np.asarray(D), "n": np.asarray(N)}


def score_exposure(D: np.ndarray, n: np.ndarray, label: np.ndarray, *, min_vertices: int = 10,
                   among=None) -> dict:
    """AUROC of D for ``label`` per configuration (rows of D), over pairs with a full enough
    tube. ``among`` restricts the pairs (e.g. over-counts vs correct pairs only)."""
    keep = np.ones(D.shape[1], bool) if among is None else np.asarray(among, bool)
    out = {"auroc": [], "pairs": [], "exposed": [], "positives": []}
    for d, k in zip(D, n):
        m = keep & (k >= min_vertices) & np.isfinite(d)
        out["auroc"].append(auroc(d[m], label[m]))
        out["pairs"].append(int(m.sum()))
        out["exposed"].append(int((d[m] > 0).sum()))
        out["positives"].append(int(label[m].sum()))
    return {k: np.asarray(v) for k, v in out.items()}


def relative_risk(d: np.ndarray, n: np.ndarray, label: np.ndarray, *, min_vertices: int = 10) -> dict:
    """Slip rate in exposed (D > 0) vs unexposed pairs, and the share of all slips that the
    exposed pairs hold (what a reviewer who checks only flagged pairs would catch)."""
    m = (n >= min_vertices) & np.isfinite(d)
    e = m & (d > 0)
    u = m & ~(d > 0)
    re = float(label[e].mean()) if e.any() else float("nan")
    ru = float(label[u].mean()) if u.any() else float("nan")
    return {"pairs": int(m.sum()), "exposed": int(e.sum()), "rate_exposed": re, "rate_unexposed": ru,
            "rr": re / ru if ru > 0 else float("nan"),
            "share_of_pairs_flagged": float(e.sum() / max(m.sum(), 1)),
            "share_of_slips_flagged": float(label[e].sum() / max(label[m].sum(), 1))}


def rotation_p(observed: float, null) -> float:
    """One-sided p: (1 + #null >= observed) / (1 + #null), nan draws dropped."""
    null = np.asarray(null, np.float64)
    null = null[np.isfinite(null)]
    return float((1 + (null >= observed).sum()) / (1 + len(null)))
