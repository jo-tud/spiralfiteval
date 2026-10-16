"""Read a spiral-fit output (one tifxyz mesh per winding) into one point set.

A fit is a folder of winding meshes ``wNNN/`` (and optionally ``wNNN_spliced/``), each a tifxyz
grid: ``x.tif``, ``y.tif``, ``z.tif`` and ``meta.json``. This is what villa's ``fit_spiral.py``
writes under ``meshes/mesh/``; anything else in that layout (another fitter, a tracer's per-wrap
output) is scored the same way. Only the meshes are read: no checkpoint, no GPU.

Every vertex gets a **spiral time** ``f = w + t``: ``w`` is the winding number from the folder
name and ``t`` in [0, 1) is the vertex's column position inside that winding, in spiral order
(columns by increasing spiral angle). One full turn of the sheet is Δf = 1, and f is continuous
across the seam between two windings, so a step along the sheet changes f by much less than 1
and a jump to the neighbouring wrap changes it by about ±1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

WINDING_RE = re.compile(r"^w(\d+)(_spliced)?$")


def read_tifxyz(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (P, valid, meta). P is (H, W, 3) float32 xyz in level-0 voxels (VC3D order).

    Invalid vertices follow the VC3D loader rule (z <= 0, or -1 fill)."""
    import tifffile
    d = Path(path)
    x, y, z = (tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "xyz")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    valid = (z > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x != -1)
    return np.stack([x, y, z], axis=-1), valid, meta


def spiral_order(P: np.ndarray, valid: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Undo the export layout (outermost-first columns, top-first rows) so that columns run by
    increasing spiral angle. A grid without ``grid_orientation`` is already in spiral order."""
    go = meta.get("grid_orientation")
    if go is None:
        return P, valid
    flip_rows = go.get("z_direction_is_top_to_bottom") is False
    P = P[:, ::-1]
    valid = valid[:, ::-1]
    if flip_rows:
        P, valid = P[::-1], valid[::-1]
    return np.ascontiguousarray(P), np.ascontiguousarray(valid)


@dataclass
class Fit:
    name: str
    xyz: np.ndarray        # (N, 3) float32, valid vertices of all windings
    f: np.ndarray          # (N,) float64 spiral time
    winding: np.ndarray    # (N,) int32 winding number from the folder name
    normal: np.ndarray     # (N, 3) float32 unit grid normal (sign arbitrary)
    windings: list[int]
    step: float            # median grid step in voxels (sampling density of the fit)
    _tree: cKDTree | None = None

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.xyz)
        return self._tree

    def project(self, pts: np.ndarray, tol_n, *, k: int = 16, chunk: int = 200_000,
                exclude_f: np.ndarray | None = None):
        """Find the fit sheet through each point.

        Among the ``k`` nearest vertices, keep those whose tangent plane passes the point within
        one grid step sideways, and take the one with the smallest distance along its normal.
        Returns (f, dn): spiral time of that vertex and the normal distance in voxels. Both are
        nan where no sheet lies within ``tol_n`` (scalar or per point) along the normal.

        ``exclude_f`` (per point) skips vertices within half a turn of that spiral time, so a
        fit vertex can look for the *other* windings next to it. The own sheet then fills the
        nearest vertices, so ``k`` is only the first guess: the search widens until no unseen
        vertex could be a better candidate, and the result does not depend on ``k`` (before commit
        3d32d3f a fixed k = 48 cut off gaps above ~16 voxels at a 4.5 voxel grid step).
        """
        pts = np.asarray(pts, np.float64)
        tol_n = np.broadcast_to(np.asarray(tol_n, np.float64), (len(pts),))
        f_out = np.full(len(pts), np.nan)
        dn_out = np.full(len(pts), np.nan)
        r = float(np.hypot(self.step, tol_n.max())) + 1e-6
        if exclude_f is None:
            for s in range(0, len(pts), chunk):
                f_out[s:s + chunk], dn_out[s:s + chunk], _ = self._best(
                    pts[s:s + chunk], tol_n[s:s + chunk], k, r, None)
            return f_out, dn_out
        exclude_f = np.asarray(exclude_f, np.float64)
        budget = chunk * k  # neighbour slots per query, as in the fixed-k search
        todo = np.arange(len(pts))
        while len(todo):
            k = min(k, len(self.xyz))
            step = max(1, budget // k)
            left = []
            for s in range(0, len(todo), step):
                j = todo[s:s + step]
                f, dn, d_last = self._best(pts[j], tol_n[j], k, r, exclude_f[j])
                f_out[j], dn_out[j] = f, dn
                # an unseen vertex is >= d_last away, so its normal distance (if it passes the
                # sideways test) is >= sqrt(d_last^2 - step^2): done when that cannot beat the
                # best hit, or (no hit) when the k-ball already holds every vertex within reach
                bound = np.where(np.isfinite(dn), dn, np.hypot(self.step, tol_n[j]))
                done = ~np.isfinite(d_last) | (d_last ** 2 - self.step ** 2 >= bound ** 2)
                left.append(j[~done])
            todo = np.concatenate(left) if left else todo[:0]
            if k >= len(self.xyz):
                break
            k *= 4
        return f_out, dn_out

    def _best(self, p, tol_n, k, r, exclude_f):
        """One fixed-k search: (f, dn, distance of the k-th neighbour, inf if fewer within r)."""
        d, i = self.tree.query(p, k=k, distance_upper_bound=r)
        if d.ndim == 1:
            d, i = d[:, None], i[:, None]
        ok = np.isfinite(d)
        i = np.where(ok, i, 0)
        v = p[:, None, :] - self.xyz[i]
        n = self.normal[i]
        dn = np.abs(np.einsum("pkc,pkc->pk", v, n))
        dt = np.sqrt(np.maximum(np.einsum("pkc,pkc->pk", v, v) - dn ** 2, 0))
        cand = ok & (dt <= self.step) & (dn <= tol_n[:, None])
        if exclude_f is not None:
            cand &= np.abs(self.f[i] - exclude_f[:, None]) >= 0.5
        dn = np.where(cand, dn, np.inf)
        best = np.argmin(dn, axis=1)
        rows = np.arange(len(p))
        hit = np.isfinite(dn[rows, best])
        f_out = np.where(hit, self.f[i[rows, best]], np.nan)
        dn_out = np.where(hit, dn[rows, best], np.nan)
        return f_out, dn_out, d[:, -1]


def grid_normals(P: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unit normals of a tifxyz grid from row and column differences (nan where undefined)."""
    Q = np.where(valid[..., None], P, np.nan).astype(np.float64)
    du = np.full_like(Q, np.nan)
    dv = np.full_like(Q, np.nan)
    du[:, 1:-1] = Q[:, 2:] - Q[:, :-2]
    du[:, 0], du[:, -1] = Q[:, 1] - Q[:, 0], Q[:, -1] - Q[:, -2]
    dv[1:-1] = Q[2:] - Q[:-2]
    dv[0], dv[-1] = Q[1] - Q[0], Q[-1] - Q[-2]
    # fall back to one-sided differences next to holes
    for D, a, b in ((du, Q[:, 1:] - Q[:, :-1], 1), (dv, Q[1:] - Q[:-1], 0)):
        bad = ~np.isfinite(D).all(-1)
        if b == 1:
            fwd = np.concatenate([a, np.full_like(a[:, :1], np.nan)], axis=1)
            bwd = np.concatenate([np.full_like(a[:, :1], np.nan), a], axis=1)
        else:
            fwd = np.concatenate([a, np.full_like(a[:1], np.nan)], axis=0)
            bwd = np.concatenate([np.full_like(a[:1], np.nan), a], axis=0)
        alt = np.where(np.isfinite(fwd).all(-1, keepdims=True), fwd, bwd)
        D[bad] = alt[bad]
    n = np.cross(du, dv)
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n.astype(np.float32)


def read_grids(source, *, spliced: bool = False, windings=None, xyz_order: str = "xyz"):
    """Winding grids in spiral order: {w: (P (H, W, 3) xyz, valid (H, W))}.

    ``source`` is a fit folder of ``wNNN[_spliced]`` tifxyz meshes, or an ``.npz`` with one
    ``wNNN`` key per winding holding a (3, H, W) array in ``xyz_order`` channel order (the
    layout of the grids some fitters release instead of tifxyz; -1 marks invalid)."""
    source = Path(source)
    out = {}
    if source.suffix == ".npz":
        z = np.load(source)
        perm = ["xyz".index(c) for c in xyz_order]
        inv = np.argsort(perm)
        for k in z.files:
            m = WINDING_RE.match(k)
            if not m or bool(m.group(2)) != spliced:
                continue
            w = int(m.group(1))
            if windings is not None and w not in windings:
                continue
            a = np.moveaxis(z[k].astype(np.float32), 0, -1)[..., inv]
            valid = (a[..., 2] > 0) & np.isfinite(a).all(-1) & (a != -1).all(-1)
            out[w] = (a, valid)
        return dict(sorted(out.items()))
    for d in sorted(source.iterdir()):
        m = WINDING_RE.match(d.name)
        if not m or bool(m.group(2)) != spliced or not d.is_dir():
            continue
        w = int(m.group(1))
        if windings is not None and w not in windings:
            continue
        P, valid, meta = read_tifxyz(d)
        out[w] = spiral_order(P, valid, meta)
    return out


def refine_grid(P: np.ndarray, valid: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear upsampling of a grid by an integer factor; a new vertex is valid only when all
    four corners of its source quad are valid."""
    if factor <= 1:
        return P, valid
    H, W = valid.shape
    r = np.arange((H - 1) * factor + 1) / factor
    c = np.arange((W - 1) * factor + 1) / factor
    r0 = np.minimum(np.floor(r).astype(int), H - 2) if H > 1 else np.zeros(len(r), int)
    c0 = np.minimum(np.floor(c).astype(int), W - 2) if W > 1 else np.zeros(len(c), int)
    fr = (r - r0)[:, None, None]
    fc = (c - c0)[None, :, None]
    R0, C0 = np.meshgrid(r0, c0, indexing="ij")
    R1, C1 = np.minimum(R0 + 1, H - 1), np.minimum(C0 + 1, W - 1)
    Q = (P[R0, C0] * (1 - fr) * (1 - fc) + P[R1, C0] * fr * (1 - fc)
         + P[R0, C1] * (1 - fr) * fc + P[R1, C1] * fr * fc)
    v = valid[R0, C0] & valid[R1, C0] & valid[R0, C1] & valid[R1, C1]
    return Q.astype(np.float32), v


def build_fit(grids: dict, *, name="fit", bbox_xyz=None, column_sense: int = 1,
              max_step: float = 5.0) -> Fit:
    """One point set from winding grids (``read_grids``). Grids coarser than ``max_step`` voxels
    are refined bilinearly so that the sheet search sees the surface between vertices.
    ``bbox_xyz`` = ((x0, y0, z0), (x1, y1, z1)) keeps only vertices inside the box.
    ``column_sense`` = -1 if a producer numbers windings against the column order (see
    ``check_seams``)."""
    if not grids:
        raise ValueError("no winding grids")
    xyz, f, wn, nrm, steps = [], [], [], [], []
    for w, (P, valid) in grids.items():
        step = np.nan
        both = valid[:, :-1] & valid[:, 1:]
        if both.any():
            step = float(np.median(np.linalg.norm(P[:, 1:][both] - P[:, :-1][both], axis=-1)))
        factor = int(np.ceil(step / max_step)) if np.isfinite(step) and step > max_step else 1
        H0, W0 = valid.shape
        P, valid = refine_grid(P, valid, factor)
        if np.isfinite(step):
            steps.append(step / factor)
        H, W = valid.shape
        # column position inside the winding, in units of the original grid's W0 columns
        t = np.broadcast_to(np.arange(W, dtype=np.float64) / (factor * W0), (H, W))
        if column_sense < 0:
            t = 1.0 - 1.0 / W0 - t
        N = grid_normals(P, valid)
        sel = valid & np.isfinite(N).all(-1)
        if bbox_xyz is not None:
            lo, hi = np.asarray(bbox_xyz[0]), np.asarray(bbox_xyz[1])
            sel = sel & np.all((P >= lo) & (P < hi), axis=-1)
        if not sel.any():
            continue
        xyz.append(P[sel])
        f.append(w + t[sel])
        wn.append(np.full(int(sel.sum()), w, np.int32))
        nrm.append(N[sel])
    if not xyz:
        raise ValueError(f"no fit vertices inside {bbox_xyz}")
    return Fit(name, np.concatenate(xyz), np.concatenate(f), np.concatenate(wn),
               np.concatenate(nrm), list(grids), float(np.median(steps)) if steps else float("nan"))


def unique_names(sources, names=None) -> list[str]:
    """One distinct label per fit source, for callers that key results by name.

    ``Path(src).stem`` is not safe: the usual layout ``<run>/meshes/mesh`` gives every run the
    name ``mesh``, and a caller that keys a dict by it silently compares a fit with itself (the
    bug once found in ``compare``: the fits collapsed to one and the difference came out
    0.0 % with a CI of [0, 0] - a false no-go). So: explicit ``names`` if given, else the
    shortest suffix of path parts that separates every source, and an error if even the full
    paths repeat.
    """
    srcs = [str(x) for x in sources]
    if names is not None:
        if len(names) != len(srcs):
            raise ValueError(f"{len(names)} names for {len(srcs)} fits")
        if len(set(names)) != len(names):
            raise ValueError(f"fit names must be distinct, got {names}")
        return list(names)
    parts = [[q for q in Path(x).parts if q not in ("/", ".")] for x in srcs]
    for depth in range(1, max(len(q) for q in parts) + 1):
        cand = ["/".join(q[-depth:]) for q in parts]
        if len(set(cand)) == len(cand):
            return cand
    raise ValueError(f"cannot tell these fits apart by path: {srcs}")


def load_fit(source, *, spliced: bool = False, windings=None, bbox_xyz=None, name=None,
             column_sense: int | None = None, xyz_order: str = "xyz", max_step: float = 5.0) -> Fit:
    """``read_grids`` + ``build_fit``. ``column_sense`` None = detect it with ``check_seams``."""
    grids = read_grids(source, spliced=spliced, windings=windings, xyz_order=xyz_order)
    if column_sense is None:
        column_sense = check_seams(grids)["sense"]
    return build_fit(grids, name=name or Path(source).stem, bbox_xyz=bbox_xyz,
                     column_sense=column_sense, max_step=max_step)


def check_seams(grids: dict, *, rows: int = 50) -> dict:
    """Which way do winding numbers run against columns?

    For consecutive windings w, w+1, compare the last column of w and the first column of w+1
    (and the reverse pairing). On a continuous spiral one pairing is about half a grid step or
    less. Returns the median distance for both pairings; ``sense`` = +1 if last(w) meets
    first(w+1)."""
    fwd, rev = [], []
    for w in grids:
        if w + 1 not in grids:
            continue
        (Pa, va), (Pb, vb) = grids[w], grids[w + 1]
        H = min(len(Pa), len(Pb))
        for i in np.linspace(0, H - 1, min(rows, H)).astype(int):
            for out, a, b in ((fwd, -1, 0), (rev, 0, -1)):
                if va[i, a] and vb[i, b]:
                    out.append(float(np.linalg.norm(Pa[i, a] - Pb[i, b])))
    mf = float(np.median(fwd)) if fwd else float("nan")
    mr = float(np.median(rev)) if rev else float("nan")
    return {"last_w_to_first_w+1": mf, "first_w_to_last_w+1": mr, "n": len(fwd),
            "sense": 1 if (np.isnan(mr) or mf <= mr) else -1}
