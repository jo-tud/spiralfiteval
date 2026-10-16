"""Rasterise per-wrap tifxyz meshes into winding-position (f) and wrap-index (k) volumes.

Level-2 voxel grid (9.596 um). Level-0 point c -> level-2 coordinate (c - 1.5) / 4.
See experiments/2026-10-13-w0b-windbench/README.md for the definitions.
"""
from __future__ import annotations

from dataclasses import dataclass

import os

import numpy as np
from scipy import ndimage

from . import io
from .metrics import UNKNOWN

THREADS = int(os.environ.get("WINDBENCH_THREADS", "6"))  # set WINDBENCH_THREADS to change
L0_TO_L2 = 4.0
DENSE = 12        # samples per quad edge; quads are ~5 voxels, so <= 0.45 voxel spacing
MARGIN = 32       # voxels of context around the crop for surfaces and distances


def to_l2(p_l0: np.ndarray) -> np.ndarray:
    return (p_l0 - 1.5) / L0_TO_L2


# Wrap labels come from turn counting, not from the mesh name alone (see README, "GT labels").
# Every mesh runs inward as its column index grows. A vertex's wrap index is
#   BASE[mesh] - (crossings of the seam half-plane SEAM_DEG between the mesh start and the vertex).
# Single-turn meshes: BASE = the wNNN name. The three multi-turn meshes (w013, w018, w023 span
# ~5 turns each and cover the gap wraps) start at their outer end; their BASE is set by radial
# order against the neighbouring GT meshes (raycheck in select.py).
SEAM_DEG = -90.0
BASE_OVERRIDE: dict[int, int] = {13: 18, 18: 23, 23: 27}


@dataclass
class Mesh:
    wrap: int
    P: np.ndarray      # (H, W, 3) level-2 coords (z, y, x)
    valid: np.ndarray  # (H, W) bool
    sign: float        # +1 if (row x col) normal points outward
    label: np.ndarray | None = None  # (H, W) float wrap index per vertex, NaN invalid


def unwrap_grid(theta: np.ndarray, valid: np.ndarray):
    """Unwrap vertex angles on the mesh grid: along each row, then rows chained by their median 2pi offset."""
    H, _ = theta.shape
    U = np.full(theta.shape, np.nan)
    for r in range(H):
        ok = np.nonzero(valid[r])[0]
        if len(ok) >= 2:
            U[r, ok] = np.unwrap(theta[r, ok])
    seed = int(np.argmax(valid.sum(1)))
    for rows in (range(seed + 1, H), range(seed - 1, -1, -1)):
        prev = seed
        for r in rows:
            both = np.isfinite(U[r]) & np.isfinite(U[prev])
            if both.sum() < 5:
                U[r] = np.nan
                continue
            U[r] -= 2 * np.pi * np.round(np.median(U[r, both] - U[prev, both]) / (2 * np.pi))
            prev = r
    return U, seed


def label_vertices(m: Mesh, centre) -> np.ndarray:
    cy, cx = centre(m.P[..., 0])
    th = np.arctan2(m.P[..., 1] - cy, m.P[..., 2] - cx)
    U, seed = unwrap_grid(th, m.valid)
    ok = np.nonzero(np.isfinite(U[seed]))[0]
    u0 = U[seed, ok[0]]  # mesh start: first valid column of the fullest row
    seam = np.radians(SEAM_DEG)
    cross = np.floor((U - seam) / (2 * np.pi)) - np.floor((u0 - seam) / (2 * np.pi))
    return BASE_OVERRIDE.get(m.wrap, m.wrap) - cross


def axis_centre(meshes: dict[int, Mesh], z_bin: float = 32.0):
    """Scroll axis (y, x) as a function of level-2 z, from the innermost GT wrap's centroid per z bin."""
    m = meshes[min(meshes)]
    q = m.P[m.valid]
    zb = np.floor(q[:, 0] / z_bin).astype(int)
    zs, inv = np.unique(zb, return_inverse=True)
    cy = np.bincount(inv, q[:, 1]) / np.bincount(inv)
    cx = np.bincount(inv, q[:, 2]) / np.bincount(inv)
    zc = (zs + 0.5) * z_bin

    def f(z):
        return np.interp(z, zc, cy), np.interp(z, zc, cx)
    return f


def quad_normals(P: np.ndarray) -> np.ndarray:
    """Per-quad normal (H-1, W-1, 3), from the row and column edge vectors."""
    du = P[1:, :-1] - P[:-1, :-1]
    dv = P[:-1, 1:] - P[:-1, :-1]
    n = np.cross(du, dv)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


def load_meshes(wraps: list[int] | None = None) -> dict[int, Mesh]:
    out = {}
    segs = [s["wrap"] for s in io.manifest()["segments"]]
    for w in segs:
        if wraps and w not in wraps:
            continue
        P, v = io.load_tifxyz(w)
        out[w] = Mesh(w, to_l2(P), v, 1.0)
    centre = axis_centre(out)
    for m in out.values():
        qv = m.valid[1:, 1:] & m.valid[:-1, 1:] & m.valid[1:, :-1] & m.valid[:-1, :-1]
        n = quad_normals(m.P)[qv][::50]
        c = m.P[:-1, :-1][qv][::50]
        cy, cx = centre(c[:, 0])
        radial = np.stack([np.zeros_like(cy), c[:, 1] - cy, c[:, 2] - cx], -1)
        s = np.sign(np.einsum("ij,ij->i", n, radial))
        m.sign = 1.0 if s.mean() >= 0 else -1.0
        m.sign_agreement = float((s == m.sign).mean())
        m.label = label_vertices(m, centre)
    return out


def dense_surface(m: Mesh, lo: np.ndarray, hi: np.ndarray):
    """Dense surface samples of mesh m inside the box [lo, hi) (level-2 coords).

    Returns points (N, 3), outward normals (N, 3), border flag (N,) - a quad is 'border' when any
    quad of its 3x3 neighbourhood is invalid, off-grid or on the seam - and the wrap label (N,).
    """
    L = m.label
    l00 = L[:-1, :-1]
    qv = (np.isfinite(l00) & (L[1:, :-1] == l00) & (L[:-1, 1:] == l00) & (L[1:, 1:] == l00))  # drops seam quads
    # border: any invalid quad in the 3x3 quad neighbourhood (pads count as invalid)
    border = ~ndimage.binary_erosion(qv, structure=np.ones((3, 3), bool), border_value=0)
    P = m.P
    qmin = np.minimum.reduce([P[:-1, :-1], P[1:, :-1], P[:-1, 1:], P[1:, 1:]])
    qmax = np.maximum.reduce([P[:-1, :-1], P[1:, :-1], P[:-1, 1:], P[1:, 1:]])
    sel = qv & np.all(qmax >= lo - 1, -1) & np.all(qmin <= hi, -1)
    r, c = np.nonzero(sel)
    if len(r) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, bool), np.zeros(0, int)
    p00, p10, p01, p11 = P[r, c], P[r + 1, c], P[r, c + 1], P[r + 1, c + 1]
    t = (np.arange(DENSE) + 0.5) / DENSE
    a, b = np.meshgrid(t, t, indexing="ij")
    a, b = a.ravel()[None, :, None], b.ravel()[None, :, None]
    pts = ((1 - a) * (1 - b) * p00[:, None] + a * (1 - b) * p10[:, None]
           + (1 - a) * b * p01[:, None] + a * b * p11[:, None])
    nrm = quad_normals(P)[r, c] * m.sign
    k = DENSE * DENSE
    return (pts.reshape(-1, 3), np.repeat(nrm, k, 0), np.repeat(border[r, c], k),
            np.repeat(l00[r, c].astype(int), k))


@dataclass
class CropGT:
    f: np.ndarray          # float32, NaN unknown
    k: np.ndarray          # int16, -1 unknown
    sheet: np.ndarray      # int16, wrap whose surface passes through the voxel; -1 none; -2 several
    pitch: np.ndarray      # float32, local pitch d_in + d_out (NaN unknown)
    order_violation: np.ndarray  # bool
    wraps: list[int]
    stats: dict


DMAX = 64.0  # voxels; a voxel with no bounding neighbour wrap within DMAX stays unknown


def _surfaces(meshes, wlo, whi, W):
    """All labelled surface voxels in the work box: sorted flat index, label (-2 = several), unit
    outward normal, border flag (border only if every sample in the voxel is border)."""
    flats, labs, nrms, brds = [], [], [], []
    for m in meshes.values():
        pts, nrm, brd, lab = dense_surface(m, wlo, whi)
        if len(pts) == 0:
            continue
        idx = np.rint(pts - wlo).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < W), 1)
        flats.append(np.ravel_multi_index(idx[ok].T, (W, W, W)))
        labs.append(lab[ok]); nrms.append(nrm[ok]); brds.append(brd[ok])
    if not flats:
        return (np.zeros(0, np.int64), np.zeros(0, np.int16), np.zeros((0, 3), np.float32), np.zeros(0, bool))
    flat = np.concatenate(flats); lab = np.concatenate(labs); nrm = np.concatenate(nrms); brd = np.concatenate(brds)
    uniq, first, inv = np.unique(flat, return_index=True, return_inverse=True)
    ulab = lab[first].astype(np.int16)
    lmin = np.full(len(uniq), 10**6); np.minimum.at(lmin, inv, lab)
    lmax = np.full(len(uniq), -10**6); np.maximum.at(lmax, inv, lab)
    ulab[lmin != lmax] = -2
    nsum = np.zeros((len(uniq), 3))
    for a in range(3):
        nsum[:, a] = np.bincount(inv, nrm[:, a], minlength=len(uniq))
    nsum /= np.linalg.norm(nsum, axis=1, keepdims=True) + 1e-12
    ubrd = np.bincount(inv, ~brd, minlength=len(uniq)) == 0
    return uniq, ulab, nsum.astype(np.float32), ubrd


def rasterise_crop(meshes: dict[int, Mesh], origin, size: int = 512, margin: int = MARGIN) -> CropGT:
    """f, k, sheet, pitch for one crop.

    Step 1: one EDT over all labelled surfaces gives each voxel its nearest wrap a, the distance
    d_a and the side (outside/inside a, by the outward normal at the nearest surface voxel).
    Step 2: the bounding neighbour b = a + 1 (outside) or a - 1 (inside) is found exactly with a
    KD-tree over b's surface voxels. f = a +/- d_a / (d_a + d_b) when b is on the opposite side
    and neither nearest point is on a mesh border; otherwise unknown. b on the same side as a is
    an order violation (type X).
    """
    from scipy.spatial import cKDTree
    origin = np.asarray(origin, float)
    wlo = origin - margin
    W = size + 2 * margin
    c0 = margin
    shp = (size,) * 3
    uniq, ulab, unrm, ubrd = _surfaces(meshes, wlo, wlo + W, W)
    stats = {"surface_voxels_work": int(len(uniq))}
    f = np.full(shp, np.nan, np.float32)
    pitch = np.full(shp, np.nan, np.float32)
    viol = np.zeros(shp, bool)
    sheet = np.full(shp, UNKNOWN, np.int16)
    if len(uniq) == 0:
        return CropGT(f, np.full(shp, UNKNOWN, np.int16), sheet, pitch, viol, [], stats)
    surf = np.zeros(W ** 3, bool)
    surf[uniq] = True
    dist, ind = ndimage.distance_transform_edt(~surf.reshape((W,) * 3), return_indices=True)
    del surf
    crop_sl = (slice(c0, c0 + size),) * 3
    d_a = dist[crop_sl].astype(np.float32)
    del dist
    ind = ind[(slice(None),) + crop_sl]
    pos = np.searchsorted(uniq, np.ravel_multi_index(ind, (W, W, W)))
    a = ulab[pos]
    on = d_a == 0
    sheet[on] = a[on]
    side = np.zeros(shp, np.float32)
    for ax in range(3):
        coord = np.arange(c0, c0 + size, dtype=np.float32).reshape([-1 if i == ax else 1 for i in range(3)])
        side += unrm[pos, ax] * (coord - ind[ax])
    del ind
    s_a = np.sign(side).astype(np.int8)
    del side
    ok_a = (~ubrd[pos]) & (a >= 0) & (~on) & (s_a != 0)
    del pos
    b = np.where(s_a > 0, a + 1, a - 1).astype(np.int16)
    b[~ok_a] = UNKNOWN
    present = sorted(int(x) for x in np.unique(ulab) if x >= 0)
    coords_all = np.stack(np.unravel_index(uniq, (W, W, W)), 1).astype(np.float32)
    d_b = np.full(shp, np.inf, np.float32)
    s_b = np.zeros(shp, np.int8)
    for w in present:
        sel = np.nonzero((b == w).ravel())[0]
        if len(sel) == 0:
            continue
        mine = ulab == w
        tree = cKDTree(coords_all[mine])
        nrm_w, brd_w, pts_w = unrm[mine], ubrd[mine], coords_all[mine]
        for chunk in np.array_split(sel, max(1, len(sel) // 4_000_000)):
            q = np.stack(np.unravel_index(chunk, shp), 1).astype(np.float32) + c0
            dd, jj = tree.query(q, k=1, distance_upper_bound=DMAX, workers=THREADS)
            hit = np.isfinite(dd)
            j = jj[hit]
            sd = np.einsum("ij,ij->i", nrm_w[j], q[hit] - pts_w[j])
            sb = np.sign(sd).astype(np.int8)
            sb[brd_w[j]] = 0  # nearest point on a border: unknown
            d_b.ravel()[chunk[hit]] = dd[hit]
            s_b.ravel()[chunk[hit]] = sb
    good = ok_a & (s_b == -s_a) & np.isfinite(d_b)
    viol = ok_a & (s_b == s_a)
    tot = d_a + d_b
    fa = a.astype(np.float32)
    f[good] = fa[good] + s_a[good] * d_a[good] / tot[good]
    pitch[good] = tot[good]
    f[sheet >= 0] = sheet[sheet >= 0]
    k = np.where(np.isfinite(f), np.rint(f), UNKNOWN).astype(np.int16)
    stats["present_wraps"] = present
    return CropGT(f, k, sheet, pitch, viol, present, stats)
