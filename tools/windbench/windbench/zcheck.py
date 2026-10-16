"""Z1 (README, v1.1): does the GT offset come from the 7.91 um -> 2.4 um mesh transfer?

A: transferred meshes (2.4 um scan, level 2) on the 2.4 um CT. B: native meshes on the 7.91 um CT
(20231117161658, level 0, public Scroll 4 volpkg). For each, the shift that maximises the mean CT
at the mesh vertices; B is mapped into the 2.4 um scan frame with the affine's linear part.
CPU only. 7.91 um chunks are cached under <data>/native791/ct with a hard byte ceiling.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import io
from .rasterise import load_meshes, to_l2

THREADS = int(os.environ.get("WINDBENCH_THREADS", "6"))
NATIVE_URL = "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes_zarr/20231117161658.zarr/0"
NATIVE_SHAPE = (11174, 3340, 3440)  # level 0, zyx, uint16, 128^3 raw chunks ("/" separator)
CHUNK = 128
CACHE_CEILING = 4 * 1024 ** 3        # hard cap on cached 7.91 um chunks
# metadata.json, PHerc1667 volume_transforms, 20231117161658 -> 20251217075048 (xyz, level-0 voxels)
AFFINE_791_TO_2399 = np.array([
    [0.5215597940837622, -3.228160802871612, 0.007120727845433622, 11915.371996649377],
    [-3.2194856258880042, -0.5227458296108656, -0.22973008901314615, 14762.725661766222],
    [0.22610217963704587, 0.02629462240233314, -3.275482322608939, 37027.46281243874]])
UM_L2, UM_2399, UM_791 = 9.596, 2.399, 7.91
N_MAX = 50_000
REGION = (128, 384)  # central 256^3 of a 512^3 crop, crop-local level-2 voxels


def native_root() -> Path:
    return io.data_root() / "native791"


def fetch_native_meshes() -> int:
    pulled = 0
    for seg in io.manifest()["segments"]:
        remote = seg["tifxyz"].replace("20251217075048-2.399um", "20231117161658-7.91um")
        dst = native_root() / "meshes" / f"w{seg['wrap']:03d}"
        for info in io.s3().ls(f"{io.BUCKET}/{remote}", detail=True):
            name = info["name"].rsplit("/", 1)[-1]
            pulled += io._get(f"{remote}{name}", dst / name, int(info["size"]))
    return pulled


def load_native(wrap: int) -> np.ndarray:
    """(N, 3) valid native vertices, xyz, 7.91 um level-0 voxels."""
    import tifffile
    d = native_root() / "meshes" / f"w{wrap:03d}"
    x, y, z = (tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz")
    ok = (z > 0) & (x > 0) & (y > 0)
    return np.stack([x[ok], y[ok], z[ok]], -1)


def native_to_l2(p_xyz: np.ndarray) -> np.ndarray:
    q = np.c_[p_xyz, np.ones(len(p_xyz))] @ AFFINE_791_TO_2399.T  # 2.4 um level-0, xyz
    return to_l2(q[:, ::-1])


class Bytes:
    pulled = 0


def _chunk(cz: int, cy: int, cx: int) -> np.ndarray:
    p = native_root() / "ct" / f"{cz}" / f"{cy}" / f"{cx}"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{NATIVE_URL}/{cz}/{cy}/{cx}", timeout=60) as r:
                    buf = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    buf = b""
                    break
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            except Exception:  # noqa: BLE001 - capped retry, then fail loudly
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if buf and len(buf) != CHUNK ** 3 * 2:
            raise IOError(f"chunk {cz}/{cy}/{cx}: {len(buf)} bytes")
        Bytes.pulled += len(buf)
        tmp = p.with_suffix(".part")
        tmp.write_bytes(buf)
        tmp.rename(p)
    buf = p.read_bytes()
    if not buf:
        return np.zeros((CHUNK,) * 3, np.uint16)
    return np.frombuffer(buf, "<u2").reshape((CHUNK,) * 3)


def read_native(lo, hi) -> np.ndarray:
    """[lo, hi) zyx of the 7.91 um level-0 volume, chunk-aligned fetch with 8 parallel streams."""
    lo = [max(0, int(v)) for v in lo]
    hi = [min(s, int(v)) for s, v in zip(NATIVE_SHAPE, hi)]
    rng = [range(l // CHUNK, (h - 1) // CHUNK + 1) for l, h in zip(lo, hi)]
    keys = [(a, b, c) for a in rng[0] for b in rng[1] for c in rng[2]]
    root = native_root() / "ct"
    cache = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if root.exists() else 0
    missing = sum(not (root / f"{a}" / f"{b}" / f"{c}").exists() for a, b, c in keys)
    if cache + missing * CHUNK ** 3 * 2 > CACHE_CEILING:
        raise RuntimeError(f"native791 chunk cache {cache / 1e9:.2f} GB + {missing} chunks would pass "
                           f"the {CACHE_CEILING / 1e9:.1f} GB ceiling")
    with ThreadPoolExecutor(8) as ex:
        blocks = dict(zip(keys, ex.map(lambda k: _chunk(*k), keys)))
    out = np.zeros([h - l for l, h in zip(lo, hi)], np.float32)
    for (a, b, c), blk in blocks.items():
        o = np.array([a, b, c]) * CHUNK
        s_lo = np.maximum(lo, o)
        s_hi = np.minimum(hi, o + CHUNK)
        out[tuple(slice(x - l, y - l) for x, y, l in zip(s_lo, s_hi, lo))] = \
            blk[tuple(slice(x - b0, y - b0) for x, y, b0 in zip(s_lo, s_hi, o))]
    return out, lo


def best_shift(vol: np.ndarray, pts: np.ndarray, half: float, step: float = 0.5):
    """argmax over s in [-half, half]^3 (step) of mean trilinear vol at pts + s. pts: (N, 3) array coords."""
    g = np.arange(-half, half + 1e-9, step)
    best, arg, s0 = -np.inf, None, None
    for dz in g:
        for dy in g:
            for dx in g:
                v = ndimage.map_coordinates(vol, (pts + [dz, dy, dx]).T, order=1, mode="nearest").mean()
                if dz == 0 and dy == 0 and dx == 0:
                    s0 = float(v)
                if v > best:
                    best, arg = float(v), [float(dz), float(dy), float(dx)]
    return arg, best, s0


def zcheck(crop_specs: list[dict]) -> dict:
    import zarr
    from .build import crop_dir
    t0 = time.time()
    fetch_native_meshes()
    meshes = load_meshes()
    native = {w: native_to_l2(load_native(w)) for w in meshes}
    lin = AFFINE_791_TO_2399[:, :3]
    rng = np.random.default_rng(0)
    out = {}
    for c in crop_specs:
        o = np.array(c["origin"], float)
        lo, hi = o + REGION[0], o + REGION[1]
        # A: transferred vertices with labels <= 30 in the region
        pa, la = [], []
        for m in meshes.values():
            sel = m.valid & np.isfinite(m.label) & np.all((m.P >= lo) & (m.P < hi), -1)
            pa.append(m.P[sel]); la.append(m.label[sel])
        pa, la = np.concatenate(pa), np.concatenate(la)
        tree = cKDTree(pa)
        pa = pa[la <= 30]
        if len(pa) > N_MAX:
            pa = pa[rng.choice(len(pa), N_MAX, replace=False)]
        ct = zarr.open_group(str(crop_dir(c["id"]) / "crop.zarr"), mode="r")["ct"][...].astype(np.float32)
        sA, vA, vA0 = best_shift(ct, pa - o, half=4.0)
        # B: native vertices whose affine image is in the region and whose nearest 2.4 vertex is <= 30
        pb_l2, pb_nat = [], []
        for w in meshes:
            q = native[w]
            sel = np.all((q >= lo) & (q < hi), -1)
            pb_l2.append(q[sel]); pb_nat.append(load_native(w)[sel])
        pb_l2, pb_nat = np.concatenate(pb_l2), np.concatenate(pb_nat)
        d, i = tree.query(pb_l2, k=1, workers=THREADS)
        keep = la[i] <= 30
        pb = pb_nat[keep][:, ::-1]  # zyx native voxels
        if len(pb) > N_MAX:
            pb = pb[rng.choice(len(pb), N_MAX, replace=False)]
        b_lo = np.floor(pb.min(0)) - 8
        b_hi = np.ceil(pb.max(0)) + 9
        b0 = Bytes.pulled
        vol, vlo = read_native(b_lo, b_hi)
        sB, vB, vB0 = best_shift(vol, pb - vlo, half=5.0)
        sB_xyz = np.array(sB[::-1])
        sB_2399_um = (lin @ sB_xyz)[::-1] * UM_2399  # zyx, um, 2.4 um scan frame
        out[c["id"]] = {
            "n_A": int(len(pa)), "n_B": int(len(pb)), "nn_dist_B_to_A_l2_median": float(np.median(d[keep])),
            "A_shift_l2": sA, "A_shift_um_zyx": [round(x * UM_L2, 1) for x in sA],
            "A_ct_at_0": vA0, "A_ct_best": vA,
            "B_shift_native_zyx": sB, "B_shift_um_zyx_in_2399_frame": [round(float(x), 1) for x in sB_2399_um],
            "B_ct_at_0": vB0, "B_ct_best": vB,
            "native_box_zyx": [vlo, [int(x) for x in np.array(vlo) + vol.shape]],
            "native_bytes_pulled": Bytes.pulled - b0,
        }
        print(c["id"], json.dumps(out[c["id"]]), flush=True)
    zA = float(np.median([r["A_shift_um_zyx"][0] for r in out.values()]))
    zB = float(np.median([r["B_shift_um_zyx_in_2399_frame"][0] for r in out.values()]))
    if abs(zA) < UM_L2:
        verdict = "no z offset at this resolution"
    elif abs(zB) <= 0.5 * abs(zA):
        verdict = "the offset comes from the transfer"
    elif abs(zA - zB) <= 0.5 * abs(zA):
        verdict = "the offset is already in the native trace, not the transfer"
    else:
        verdict = "mixed, not resolved"
    return {"crops": out, "z_A_um_median": zA, "z_B_um_median": zB, "verdict": verdict,
            "native_bytes_pulled": Bytes.pulled, "seconds": round(time.time() - t0, 1)}
