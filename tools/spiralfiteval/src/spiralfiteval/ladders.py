"""H1 on a published ladder set, so this tool's numbers sit next to the published ones pair for pair.

A *ladder* is a line of clicks across consecutive wraps in one slice, each with a winding label
``wind_a``. ``load_ladders`` applies the selection of Jashann's ``exp/e31b_bootstrap.py``
(vesuvius-scrolling, release v1.1): per collection, the points that carry ``wind_a``, sorted by
it; at least two points; z spread <= 2; mean z in [z0 + 20, z1 - 20). On Paris 4 band
8400-9400 (``abs_winding.json`` then ``relative_windings.json``) this gives his 32 human ladders;
on a PCU gold file it gives his gold ladders, randomly cut to ``nmax`` as he does.

``ladder_slips`` counts slips on consecutive ladder points with two rules:

- ``published``: Jashann's rule. Each point takes the *winding label* of its fit sheet;
  a step is a slip if sign * Δlabel != Δwind_a, with the sign fixed per ladder by the median
  Δlabel (so a ladder clicked inward is not all slips).
- ``tool``: this tool's H1 rule. Δw = round(Δf) of the continuous spiral time, no sign flip.

Both use this tool's projector (``Fit.project``: nearest sheet along the normal, within ``tol_n``), so the
difference between a published rate and this tool's under the published rule is the projector alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fit import Fit


def load_ladders(paths, z_range, *, margin: float = 20.0, max_dz: float = 2.0,
                 nmax: int | None = None, seed: int = 0) -> list[dict]:
    """Ladders from VC3D point-collection files, in file then collection order."""
    z0, z1 = z_range
    out = []
    for path in ([paths] if isinstance(paths, (str, Path)) else paths):
        doc = json.loads(Path(path).read_text())
        for cid, col in doc["collections"].items():
            P = sorted((p for p in col["points"].values() if p.get("wind_a") is not None),
                       key=lambda p: p["wind_a"])
            if len(P) < 2:
                continue
            xyz = np.array([p["p"] for p in P], np.float64)
            if np.ptp(xyz[:, 2]) > max_dz or not (z0 + margin <= xyz[:, 2].mean() < z1 - margin):
                continue
            out.append({"name": col.get("name", cid), "source": Path(path).name, "xyz": xyz,
                        "w": np.array([p["wind_a"] for p in P], np.float64)})
    if nmax is not None and len(out) > nmax:
        keep = sorted(np.random.default_rng(seed).choice(len(out), nmax, replace=False))
        out = [out[i] for i in keep]
    return out


def ladder_slips(fit: Fit, ladders: list[dict], *, tol_n: float = 10.0, f=None,
                 n_boot: int = 2000, seed: int = 1) -> dict:
    """Slips per wrap on consecutive ladder points under both rules, per ladder and pooled,
    with a ladder-level bootstrap CI (Jashann's: ``default_rng(1)``, ladders resampled).
    ``f`` (spiral time per point, all ladders concatenated) skips the projection, for floors."""
    if not ladders:
        return {"ladders": 0}
    xyz = np.concatenate([l["xyz"] for l in ladders])
    if f is None:
        f, _ = fit.project(xyz, tol_n)
    cuts = np.cumsum([0] + [len(l["w"]) for l in ladders])
    S = {"published": [], "tool": []}
    T = []
    for l, a, b in zip(ladders, cuts[:-1], cuts[1:]):
        fl = f[a:b]
        dw = np.diff(l["w"])
        ok = np.isfinite(fl[1:]) & np.isfinite(fl[:-1])
        T.append(int(ok.sum()))
        if not ok.any():
            S["published"].append(0)
            S["tool"].append(0)
            continue
        dlab = np.diff(np.floor(fl))
        sg = np.sign(np.nanmedian(dlab)) or 1.0
        S["published"].append(int(np.sum(sg * dlab[ok] != dw[ok])))
        S["tool"].append(int(np.sum(np.rint(np.diff(fl))[ok] != dw[ok])))
    T = np.asarray(T)
    res = {"ladders": len(ladders), "pairs": int(sum(len(l["w"]) - 1 for l in ladders)),
           "pairs_assigned": int(T.sum()), "unassigned_points": int((~np.isfinite(f)).sum()),
           "T": T.tolist()}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ladders), (n_boot, len(ladders))) if n_boot else None
    for rule, s in S.items():
        s = np.asarray(s)
        r = {"slips": int(s.sum()), "rate": float(s.sum() / max(T.sum(), 1)), "S": s.tolist()}
        if draw is not None:
            rates = s[draw].sum(1) / np.maximum(T[draw].sum(1), 1)
            r["ci95"] = [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]
        res[rule] = r
    return res


def windings_per_wrap(fit: Fit, ladders: list[dict], *, tol_n: float = 10.0, f=None,
                      blocks=None) -> dict:
    """Fit windings per labelled wrap: sum |Δf| / sum |Δwind_a| over consecutive ladder points
    on the fit. 1.0 = the fit has one winding per sheet; > 1 = surplus windings squeezed between
    the labelled sheets; < 1 = sheets skipped. ``blocks`` = [(w_lo, w_hi), ...] splits the pairs
    by the fit winding of the inner point."""
    xyz = np.concatenate([l["xyz"] for l in ladders])
    if f is None:
        f, _ = fit.project(xyz, tol_n)
    cuts = np.cumsum([0] + [len(l["w"]) for l in ladders])
    df, dw, w_in = [], [], []
    for l, a, b in zip(ladders, cuts[:-1], cuts[1:]):
        fl = f[a:b]
        ok = np.isfinite(fl[1:]) & np.isfinite(fl[:-1])
        df.append(np.abs(np.diff(fl))[ok])
        dw.append(np.abs(np.diff(l["w"]))[ok])
        w_in.append(np.floor(np.minimum(fl[1:], fl[:-1]))[ok])
    df, dw, w_in = (np.concatenate(x) if x else np.zeros(0) for x in (df, dw, w_in))

    def one(m):
        return {"pairs": int(m.sum()),
                "ratio": float(df[m].sum() / dw[m].sum()) if m.any() and dw[m].sum() else float("nan"),
                "median_dfdw": float(np.median(df[m] / dw[m])) if m.any() else float("nan")}
    out = one(np.ones(len(df), bool))
    if blocks:
        out["blocks"] = {f"w{lo:03d}-{hi:03d}": one((w_in >= lo) & (w_in <= hi)) for lo, hi in blocks}
    return out
