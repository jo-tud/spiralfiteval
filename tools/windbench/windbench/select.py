"""Crop candidates from the GT meshes and the coarse scan mask only (no Lasagna, no metric results).

Per labelled surface vertex of wrap k with a GT neighbour k+1: local pitch = signed point-to-plane
distance to the nearest vertex of wrap k+1 (positive = k+1 lies outside, as it should).
Candidate 512^3 boxes on the 128-voxel chunk grid are summarised by the wraps they contain, their
median pitch, the share of wrong-side vertices (< -2 voxels), their distance in angle to the
labelling seam, and their in-mask share (CT > 0 at level 5).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from .rasterise import SEAM_DEG, axis_centre, quad_normals

THREADS = int(os.environ.get("WINDBENCH_THREADS", "6"))  # set WINDBENCH_THREADS to change
CHUNK = 128
SIZE = 512


def label_surfaces(meshes, step: int = 2):
    """{label: (points, outward normals)} from label-consistent quads of all meshes."""
    out = {}
    for m in meshes.values():
        L = m.label
        l00 = L[:-1, :-1]
        q = np.isfinite(l00) & (L[1:, :-1] == l00) & (L[:-1, 1:] == l00) & (L[1:, 1:] == l00)
        q[1::step] = False
        q[:, 1::step] = False
        p = m.P[:-1, :-1][q]
        n = (quad_normals(m.P) * m.sign)[q]
        lab = l00[q].astype(int)
        for k in np.unique(lab):
            out.setdefault(int(k), []).append((p[lab == k], n[lab == k]))
    return {k: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for k, v in out.items()}


def vertex_pitch(surfs):
    pts, lab, pit = [], [], []
    for k in sorted(surfs):
        if k + 1 not in surfs:
            continue
        pa, _ = surfs[k]
        pb, nb = surfs[k + 1]
        d, i = cKDTree(pb).query(pa, k=1, distance_upper_bound=100, workers=THREADS)
        ok = np.isfinite(d)
        sd = np.einsum("ij,ij->i", pb[i[ok]] - pa[ok], nb[i[ok]])
        pts.append(pa[ok]); lab.append(np.full(ok.sum(), k)); pit.append(sd)
    return np.concatenate(pts), np.concatenate(lab), np.concatenate(pit)


def box_angles(centre, origin, size=SIZE):
    """Min angular distance (deg) between the box's edges and the seam (0 if the box straddles it),
    and whether the box contains the scroll axis. `centre` = rasterise.axis_centre(meshes)."""
    z0, y0, x0 = origin
    best, contains = 180.0, False
    for z in (z0, z0 + size / 2, z0 + size):
        cy, cx = centre(np.array([z]))
        cy, cx = float(cy[0]), float(cx[0])
        if y0 <= cy <= y0 + size and x0 <= cx <= x0 + size:
            contains = True
        t = np.linspace(0, 1, 33)
        ys = np.concatenate([y0 + 0 * t, y0 + size + 0 * t, y0 + size * t, y0 + size * t])
        xs = np.concatenate([x0 + size * t, x0 + size * t, x0 + 0 * t, x0 + size + 0 * t])
        ang = np.degrees(np.arctan2(ys - cy, xs - cx))
        # the box spans an angle interval; check whether the seam lies inside it
        rel = np.angle(np.exp(1j * np.radians(ang - SEAM_DEG)), deg=True)
        if rel.min() < 0 < rel.max() and (rel.max() - rel.min()) < 180:
            return 0.0, contains
        best = min(best, float(np.abs(rel).min()))
    return best, contains


PITCH_BINS = np.arange(-50.0, 100.5, 0.5)


def candidates(pts, lab, pit, vol_shape_l2=(9269, 3808, 3808), min_vertices=300):
    """Per-chunk histograms (labels, pitch), summed over 4x4x4 chunks = one 512^3 box."""
    nz, ny, nx = (s // CHUNK for s in vol_shape_l2)
    c = np.floor(pts / CHUNK).astype(int)
    ok = np.all((c >= 0) & (c < [nz, ny, nx]), 1)
    c, lab, pit = c[ok], lab[ok], pit[ok]
    labs = np.arange(lab.min(), lab.max() + 1)
    flat = (c[:, 0] * ny + c[:, 1]) * nx + c[:, 2]
    nb = len(PITCH_BINS) - 1
    pb = np.clip(np.digitize(pit, PITCH_BINS) - 1, 0, nb - 1)
    H_lab = np.bincount(flat * len(labs) + (lab - labs[0]), minlength=nz * ny * nx * len(labs)).reshape(nz, ny, nx, len(labs))
    H_pit = np.bincount(flat * nb + pb, minlength=nz * ny * nx * nb).reshape(nz, ny, nx, nb).astype(np.int32)

    def box_sum(H):
        S = H.cumsum(0).cumsum(1).cumsum(2)
        S = np.pad(S, ((1, 0), (1, 0), (1, 0), (0, 0)))
        k = SIZE // CHUNK
        return (S[k:, k:, k:] - S[:-k, k:, k:] - S[k:, :-k, k:] - S[k:, k:, :-k]
                + S[:-k, :-k, k:] + S[:-k, k:, :-k] + S[k:, :-k, :-k] - S[:-k, :-k, :-k])

    B_lab = box_sum(H_lab)
    B_pit = box_sum(H_pit)
    # pitch-weighted histogram: a gap of pitch p holds ~p voxels per unit area (v1.1 proxy)
    H_vox = np.bincount(flat * nb + pb, weights=np.clip(pit, 0, None), minlength=nz * ny * nx * nb).reshape(nz, ny, nx, nb)
    B_vox = box_sum(H_vox)
    occ = (H_lab[..., labs <= 30].sum(-1) >= 20).astype(np.int32)[..., None]  # chunk holds wraps <= 30
    B_occ = box_sum(occ)[..., 0] / float((SIZE // CHUNK) ** 3)
    rows = []
    for idx in np.argwhere((B_lab >= min_vertices).sum(-1) >= 3):
        cnt = B_lab[tuple(idx)]
        hp = B_pit[tuple(idx)]
        cum = np.cumsum(hp)
        med = PITCH_BINS[np.searchsorted(cum, cum[-1] / 2)] + 0.25
        cv = np.cumsum(B_vox[tuple(idx)])
        vmed = PITCH_BINS[np.searchsorted(cv, cv[-1] / 2)] + 0.25 if cv[-1] > 0 else float("nan")
        pairs = [int(labs[i]) for i in np.nonzero(cnt >= min_vertices)[0]]
        rows.append({"origin": [int(x) * CHUNK for x in idx], "pairs": pairs, "median_pitch": float(med), "voxel_pitch_proxy": float(vmed),
                     "wrong_side": float(hp[PITCH_BINS[:-1] < -2].sum() / cum[-1]),
                     "max_label": int(max(pairs) + 1), "n_vertices": int(cum[-1]),
                     "chunk_occupancy": float(B_occ[tuple(idx)])})
    return rows


def overlaps(a, b, size=SIZE):
    return all(abs(x - y) < size for x, y in zip(a, b))


def eligible_v11(rows, centre, taken, min_occupancy=0.5, max_wrong=0.02, min_seam=5.0):
    """v1.1 rule (README, v1.1 item 2): GT-only filters, ranked by the voxel-weighted pitch proxy.
    Greedy non-overlap with `taken` origins and with each other happens in the caller (rasterise in
    rank order). Returns rows sorted by proxy, each with seam distance added."""
    out = []
    for r in rows:
        if r["max_label"] > 30 or r["wrong_side"] > max_wrong or r["chunk_occupancy"] < min_occupancy:
            continue
        if any(overlaps(r["origin"], t) for t in taken):
            continue
        seam, contains = box_angles(centre, r["origin"])
        if contains or seam < min_seam:
            continue
        out.append({**r, "seam_distance_deg": round(seam, 1)})
    return sorted(out, key=lambda r: r["voxel_pitch_proxy"])
