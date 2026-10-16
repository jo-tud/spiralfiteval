"""Build one benchmark crop: CT, Lasagna (native resolution), GT f/k/sheet/pitch, error sites, GT traces.

Idempotent: a crop whose `done.json` exists with the same spec is skipped.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import io
from .rasterise import rasterise_crop
from .sites import extract_sites, type_masks

VERSION = "v1"
SIZE = 512


def crop_dir(crop_id: str) -> Path:
    return io.data_root() / VERSION / "crops" / crop_id


def lasagna_box(origin, size: int, level: int):
    """Native-resolution box of a Lasagna array at `level` that covers the level-2 crop, +1 voxel each side.

    Level-2 voxel i maps to level-L coordinate (i + 1.5) * 4 / 2**L - 0.5 - ... (mean-pool centres):
    centre of L2 voxel i is level-0 4i + 1.5; centre of level-L voxel j is 2**L j + (2**L - 1) / 2.
    """
    s = 2 ** level
    lo = [int(np.floor((4 * o + 1.5 - (s - 1) / 2) / s)) - 1 for o in origin]
    hi = [int(np.ceil((4 * (o + size - 1) + 1.5 - (s - 1) / 2) / s)) + 2 for o in origin]
    return lo, hi


def upsample_to_l2(block: np.ndarray, block_lo, level: int, origin, size: int = SIZE) -> np.ndarray:
    """Separable linear interpolation of a native level-L block onto the level-2 crop grid (float32)."""
    s = 2 ** level
    out = block.astype(np.float32)
    for ax in range(3):
        i = np.arange(origin[ax], origin[ax] + size)
        src = (4 * i + 1.5 - (s - 1) / 2) / s - block_lo[ax]
        i0 = np.clip(np.floor(src).astype(int), 0, out.shape[ax] - 2)
        t = np.clip(src - i0, 0, 1).astype(np.float32)
        a = np.take(out, i0, axis=ax)
        b = np.take(out, i0 + 1, axis=ax)
        shp = [1, 1, 1]
        shp[ax] = -1
        out = a + t.reshape(shp) * (b - a)
    return out


def gt_traces(meshes, origin, size: int = SIZE, stride: int = 2) -> list[np.ndarray]:
    """GT mesh grid lines (rows and columns, every `stride`-th) inside the crop, split at invalid,
    out-of-crop or label-change vertices. Crop voxel coords."""
    out = []
    o = np.asarray(origin, float)
    for m in meshes.values():
        L = m.label
        inside = np.all((m.P >= o - 0.5) & (m.P < o + size - 0.5), -1) & np.isfinite(L)
        if not inside.any():
            continue
        rows = np.nonzero(inside.any(1))[0]
        cols = np.nonzero(inside.any(0))[0]
        lines = [(m.P[r], inside[r], L[r]) for r in rows[::stride]]
        lines += [(m.P[:, c], inside[:, c], L[:, c]) for c in cols[::stride]]
        for P, ok, lab in lines:
            cut = ~ok[1:] | ~ok[:-1] | (lab[1:] != lab[:-1])
            start = None
            for j in range(len(P)):
                if ok[j] and start is None:
                    start = j
                end_here = (j == len(P) - 1) or (not ok[j]) or (j < len(P) - 1 and cut[j])
                if start is not None and end_here:
                    stop = j + 1 if ok[j] else j
                    if stop - start >= 2:
                        out.append((P[start:stop] - o).astype(np.float32))
                    start = None
    return out


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_crop(meshes, spec: dict, force: bool = False) -> dict:
    import zarr
    d = crop_dir(spec["id"])
    done = d / "done.json"
    if done.exists() and not force:
        rec = json.loads(done.read_text())
        if rec["spec"] == spec:
            return rec
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b0 = io.ByteCounter.total
    man = io.manifest()
    o = spec["origin"]
    hi = [x + SIZE for x in o]
    ct = io.read_box(man["volume"]["path"] + str(man["volume"]["level"]), o, hi)
    las = {}
    for name, path in man["lasagna"]["arrays"].items():
        lvl = man["lasagna"]["levels"][name]
        lo_, hi_ = lasagna_box(o, SIZE, lvl)
        las[name] = (io.read_box(man["lasagna"]["base"] + path, lo_, hi_), lo_, lvl)
    t_read = time.time() - t0
    gt = rasterise_crop(meshes, o, SIZE)
    t_gt = time.time() - t0 - t_read
    cos_up = upsample_to_l2(*las["cos"], o)
    masks = type_masks(gt.f, gt.k, gt.sheet, gt.pitch, gt.order_violation, cos_up)
    del cos_up
    site_labels, sites = extract_sites(masks, gt.k, o)
    traces = gt_traces(meshes, o)

    g = zarr.open_group(str(d / "crop.zarr"), mode="w")
    arrays = {"ct": ct, "f": gt.f, "k": gt.k, "sheet": gt.sheet, "pitch": gt.pitch,
              "order_violation": gt.order_violation, "sites": site_labels}
    for name, (blk, lo_, lvl) in las.items():
        arrays[f"lasagna_{name}"] = blk
    hashes = {}
    for name, a in arrays.items():
        z = g.create_array(name, shape=a.shape, dtype=a.dtype, chunks=tuple(min(128, s) for s in a.shape))
        z[...] = a
        hashes[name] = _sha(a)
    g.attrs.update({"origin_l2": o, "size": SIZE,
                    "lasagna": {n: {"lo": lo_, "level": lvl} for n, (_, lo_, lvl) in las.items()}})
    np.savez_compressed(d / "traces.npz", points=np.concatenate(traces) if traces else np.zeros((0, 3), np.float32),
                        offsets=np.cumsum([0] + [len(t) for t in traces]))
    (d / "sites.json").write_text(json.dumps(sites, indent=1))
    rec = {"spec": spec, "sha256": hashes, "n_sites": len(sites),
           "sites_by_type": {t: sum(s["type"] == t for s in sites) for t in ("T", "N", "X")},
           "present_wraps": gt.wraps, "n_traces": len(traces),
           "bytes_pulled": io.ByteCounter.total - b0,
           "seconds": {"read": round(t_read, 1), "gt": round(t_gt, 1), "total": round(time.time() - t0, 1)},
           "stats": gt.stats}
    done.write_text(json.dumps(rec, indent=1, default=int))
    return rec


def load_crop(crop_id: str) -> dict:
    """Load a built crop as numpy arrays; Lasagna channels upsampled to the 512^3 grid on request."""
    import zarr
    d = crop_dir(crop_id)
    g = zarr.open_group(str(d / "crop.zarr"), mode="r")
    out = {k: g[k][...] for k in ("ct", "f", "k", "sheet", "pitch", "order_violation", "sites")}
    out["origin"] = list(g.attrs["origin_l2"])
    out["lasagna_native"] = {n: (g[f"lasagna_{n}"][...], v["lo"], v["level"]) for n, v in g.attrs["lasagna"].items()}
    out["sites_meta"] = json.loads((d / "sites.json").read_text())
    tr = np.load(d / "traces.npz")
    off = tr["offsets"]
    out["traces"] = [tr["points"][a:b] for a, b in zip(off[:-1], off[1:])]
    return out


def lasagna_l2(crop: dict, name: str) -> np.ndarray:
    """Upsampled Lasagna channel on the crop grid, decoded: cos in [0,1], grad_mag in wraps per
    level-2 voxel, nx/ny in [-1, 1]."""
    blk, lo, lvl = crop["lasagna_native"][name]
    x = upsample_to_l2(blk, lo, lvl, crop["origin"])
    if name == "cos":
        return x / 255.0
    if name == "grad_mag":
        return x / 1000.0 * 4.0
    return (x - 128.0) / 127.0
