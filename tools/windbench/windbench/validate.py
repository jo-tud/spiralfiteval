"""Benchmark integrity checks (README). Run after `windbench build`; does not change any data."""
from __future__ import annotations

import json
import zlib

import numpy as np
from scipy import ndimage

from .build import SIZE, crop_dir, lasagna_l2, load_crop
from .rasterise import load_meshes

SHIFTS = range(-3, 4)
N_SAMPLE = 1_000_000


def _shift_scan(vol, idx_a, idx_b=None, vals=None):
    """For each integer shift s in [-3,3]^3: (mean vol[idx_a+s] - mean vol[idx_b+s]) if idx_b given,
    else Pearson r between vol[idx_a+s] and vals. Returns {s: score}."""
    out = {}
    hi = np.array(vol.shape) - 1
    for dz in SHIFTS:
        for dy in SHIFTS:
            for dx in SHIFTS:
                s = np.array([dz, dy, dx])
                a = np.clip(idx_a + s, 0, hi)
                va = vol[a[:, 0], a[:, 1], a[:, 2]].astype(np.float64)
                if idx_b is not None:
                    b = np.clip(idx_b + s, 0, hi)
                    out[(dz, dy, dx)] = float(va.mean() - vol[b[:, 0], b[:, 1], b[:, 2]].astype(np.float64).mean())
                else:
                    out[(dz, dy, dx)] = float(np.corrcoef(va, vals)[0, 1])
    return out


def _best(scores):
    s = max(scores, key=scores.get)
    return list(s), all(abs(x) <= 1 for x in s)


def crop_rng(crop_id: str) -> np.random.Generator:
    """One generator per crop, seeded from the crop id, so a crop's sample does not depend on which
    crops ran before it (bug review finding #6). crc32, not hash(): stable across processes."""
    return np.random.default_rng(zlib.crc32(crop_id.encode()))


def _sample(mask, rng, n=N_SAMPLE):
    """Up to n voxel indices (m, 3) of mask; same draw as np.argwhere(mask)[choice] without
    holding the full (N, 3) index array."""
    flat = np.flatnonzero(mask)
    if len(flat) > n:
        flat = flat[rng.choice(len(flat), n, replace=False)]
    return np.stack(np.unravel_index(flat, mask.shape), -1)


def round_trip(meshes, crop, tol=0.15):
    """f (trilinear) at GT vertices whose 8 neighbours are known, vs the vertex label."""
    f = crop["f"]
    o = np.array(crop["origin"], float)
    errs = []
    for m in meshes.values():
        L = m.label
        sel = np.isfinite(L) & np.all((m.P >= o) & (m.P < o + SIZE - 1), -1)
        if not sel.any():
            continue
        p = m.P[sel] - o
        lab = L[sel]
        i0 = np.floor(p).astype(int)
        known = np.ones(len(p), bool)
        for d in np.ndindex(2, 2, 2):
            q = i0 + d
            known &= np.isfinite(f[q[:, 0], q[:, 1], q[:, 2]])
        if not known.any():
            continue
        v = ndimage.map_coordinates(np.nan_to_num(f), p[known].T, order=1)
        errs.append(np.abs(v - lab[known]))
    e = np.concatenate(errs) if errs else np.zeros(0)
    return {"n": int(len(e)), "within": int((e <= tol).sum())}


def validate(crop_ids: list[str], compressed_abs: float | None = None) -> dict:
    """compressed_abs: v1.1 rule, crop median voxel pitch <= this many voxels (v1: None = 0.5 x pooled)."""
    meshes = load_meshes()
    per, pitches = {}, []
    for cid in crop_ids:
        rng = crop_rng(cid)
        c = load_crop(cid)
        f, k, ct = c["f"], c["k"], c["ct"]
        known = np.isfinite(f)
        inscroll = ct > 0
        frac = np.abs(f - k)
        sheet = known & (frac <= 0.1)
        mid = known & (frac >= 0.4)
        r = {"origin": c["origin"]}
        r["coverage"] = float((known & inscroll).sum() / max(1, inscroll.sum()))
        r["inscroll"] = float(inscroll.mean())
        r["known"] = float(known.mean())
        r["x_frac"] = float(c["order_violation"].sum() / max(1, known.sum()))
        labs = sorted(int(x) for x in np.unique(k[k >= 0]))
        run = best = 1
        for a, b in zip(labs, labs[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
        r["labels"] = labs
        r["max_consecutive"] = best if labs else 0
        pk = c["pitch"][known & np.isfinite(c["pitch"])]
        r["median_pitch"] = float(np.median(pk)) if len(pk) else float("nan")
        pitches.append(pk)
        r["round_trip"] = round_trip(meshes, c)
        ia, ib = _sample(sheet & inscroll, rng), _sample(mid & inscroll, rng)
        ct_scan = _shift_scan(ct, ia, ib)
        r["ct_contrast0"] = ct_scan[(0, 0, 0)]
        r["ct_sheet_mean"] = float(ct[sheet & inscroll].mean())
        r["ct_mid_mean"] = float(ct[mid & inscroll].mean())
        r["ct_best_shift"], r["ct_shift_ok"] = _best(ct_scan)
        cos = lasagna_l2(c, "cos")
        iv = _sample(known & inscroll, rng)
        target = 0.5 + 0.5 * np.cos(2 * np.pi * f[iv[:, 0], iv[:, 1], iv[:, 2]])
        las_scan = _shift_scan(cos, iv, vals=target)
        r["lasagna_r0"] = las_scan[(0, 0, 0)]
        r["lasagna_best_shift"], r["lasagna_shift_ok"] = _best(las_scan)
        r["lasagna_r_best"] = max(las_scan.values())
        meta = c["sites_meta"]
        r["sites"] = {t: sum(s["type"] == t for s in meta) for t in ("T", "N", "X")}
        r["n_sites"] = len(meta)
        r["done"] = json.loads((crop_dir(cid) / "done.json").read_text())
        per[cid] = r
        del c, cos
    pooled = float(np.median(np.concatenate(pitches))) if pitches else float("nan")
    for r in per.values():
        r["compressed_relative"] = bool(r["median_pitch"] <= 0.5 * pooled)
        r["compressed"] = bool(r["median_pitch"] <= compressed_abs) if compressed_abs else r["compressed_relative"]
    P = list(per.values())
    rt_n = sum(r["round_trip"]["n"] for r in P)
    rt_w = sum(r["round_trip"]["within"] for r in P)
    crit = {
        "C1": {"n_crops": len(P), "aligned": all(all(x % 128 == 0 for x in r["origin"]) for r in P),
               "min_consecutive": min(r["max_consecutive"] for r in P),
               "n_compressed": sum(r["compressed"] for r in P), "pooled_median_pitch": pooled,
               "compressed_threshold": compressed_abs or 0.5 * pooled,
               "n_compressed_relative": sum(r["compressed_relative"] for r in P)},
        "C2": {"min_coverage": min(r["coverage"] for r in P)},
        "C3": {"share_within_0.15": rt_w / max(1, rt_n), "n_vertices": rt_n},
        "C4": {"max_x_frac": max(r["x_frac"] for r in P)},
        "C5": {"all_sheet_brighter": all(r["ct_sheet_mean"] > r["ct_mid_mean"] for r in P),
               "share_shift_ok": float(np.mean([r["ct_shift_ok"] for r in P]))},
        "C6": {"median_r0": float(np.median([r["lasagna_r0"] for r in P])),
               "share_shift_ok": float(np.mean([r["lasagna_shift_ok"] for r in P]))},
        "C7": {"n_sites": sum(r["n_sites"] for r in P),
               "min_sites_compressed": min([r["n_sites"] for r in P if r["compressed"]], default=None)},
    }
    c = crit
    c["C1"]["pass"] = (6 <= c["C1"]["n_crops"] <= 10 and c["C1"]["aligned"] and c["C1"]["min_consecutive"] >= 3
                       and c["C1"]["n_compressed"] >= 2)
    c["C2"]["pass"] = c["C2"]["min_coverage"] >= 0.40
    c["C3"]["pass"] = c["C3"]["share_within_0.15"] >= 0.99
    c["C4"]["pass"] = c["C4"]["max_x_frac"] <= 0.01
    c["C5"]["pass"] = c["C5"]["all_sheet_brighter"] and c["C5"]["share_shift_ok"] >= 0.8
    c["C6"]["pass"] = c["C6"]["median_r0"] >= 0.3 and c["C6"]["share_shift_ok"] >= 0.8
    c["C7"]["pass"] = c["C7"]["n_sites"] >= 20 and (c["C7"]["min_sites_compressed"] or 0) >= 1
    return {"criteria": crit, "crops": per}
