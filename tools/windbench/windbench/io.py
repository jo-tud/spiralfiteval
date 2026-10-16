"""Data access: the manifest, GT meshes (tifxyz) and chunk-aligned zarr crops from S3.

All reads are anonymous range reads against the public bucket (AWS pays egress).
Nothing here downloads a whole volume: meshes are ~36 MB each, crops are 512^3 at level 2.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

MANIFEST = Path(__file__).resolve().parent.parent / "manifest" / "pherc1667.json"
BUCKET = "vesuvius-challenge-open-data"

# Fail fast: a stalled stream must error in seconds, not hang for an hour.
S3_CONFIG = {"connect_timeout": 10, "read_timeout": 60, "retries": {"max_attempts": 5, "mode": "standard"}}


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def data_root() -> Path:
    """Shared cache for derived benchmark data (gitignored).

    Default: <main checkout>/data/windbench, shared by all worktrees. Override with WINDBENCH_DATA.
    """
    env = os.environ.get("WINDBENCH_DATA")
    if env:
        return Path(env)
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=Path(__file__).parent,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return (Path(__file__).parent / common).resolve().parent / "data" / "windbench"


@lru_cache(maxsize=1)
def s3():
    import s3fs
    return s3fs.S3FileSystem(anon=True, config_kwargs=S3_CONFIG)


class ByteCounter:
    """Counts bytes pulled from S3 in this process (egress is money, even when AWS pays)."""
    total = 0


def _get(remote: str, local: Path, expect_size: int | None = None) -> int:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and (expect_size is None or local.stat().st_size == expect_size):
        return 0
    tmp = local.with_suffix(local.suffix + ".part")
    for attempt in range(4):
        try:
            s3().get(f"{BUCKET}/{remote}", str(tmp))
            break
        except Exception:  # noqa: BLE001 - retry with a cap, then fail loudly
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    n = tmp.stat().st_size
    if expect_size is not None and n != expect_size:
        raise IOError(f"{remote}: got {n} bytes, expected {expect_size}")
    tmp.rename(local)
    ByteCounter.total += n
    return n


def fetch_meshes(wraps: list[int] | None = None) -> int:
    """Download the per-wrap tifxyz meshes (on the 2.399 um volume). Idempotent. Returns bytes pulled."""
    pulled = 0
    for seg in manifest()["segments"]:
        if wraps and seg["wrap"] not in wraps:
            continue
        dst = data_root() / "meshes" / f"w{seg['wrap']:03d}"
        for info in s3().ls(f"{BUCKET}/{seg['tifxyz']}", detail=True):
            name = info["name"].rsplit("/", 1)[-1]
            pulled += _get(f"{seg['tifxyz']}{name}", dst / name, int(info["size"]))
    return pulled


def load_tifxyz(wrap: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (P, valid): P is (H, W, 3) float32 in level-0 voxel coords, order (z, y, x)."""
    import tifffile
    d = data_root() / "meshes" / f"w{wrap:03d}"
    x, y, z = (tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "xyz")
    valid = z > 0  # tifxyz loader rule: Z <= 0 is invalid
    if (d / "mask.tif").exists():
        m = tifffile.imread(d / "mask.tif")
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape == valid.shape:
            valid &= m >= 255
    return np.stack([z, y, x], axis=-1), valid


def open_zarr(path: str):
    import zarr
    return zarr.open_array(f"s3://{BUCKET}/{path}", mode="r", storage_options={"anon": True, "config_kwargs": S3_CONFIG})


def read_box(path: str, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> np.ndarray:
    """Read [lo, hi) from a remote zarr array, zero-filled outside the array. Counts bytes (uncompressed)."""
    a = open_zarr(path)
    out = np.zeros(tuple(h - l for l, h in zip(lo, hi)), dtype=a.dtype)
    slo = [max(0, l) for l in lo]
    shi = [min(s, h) for s, h in zip(a.shape, hi)]
    if all(h > l for l, h in zip(slo, shi)):
        for attempt in range(4):
            try:
                blk = a[tuple(slice(l, h) for l, h in zip(slo, shi))]
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        out[tuple(slice(l - o, h - o) for l, h, o in zip(slo, shi, lo))] = blk
        ByteCounter.total += blk.nbytes
    return out
