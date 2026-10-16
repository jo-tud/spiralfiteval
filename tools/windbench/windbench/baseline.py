"""Naive spiral-prior baseline: k = floor(r / p), r = distance to the scroll axis in the slice,
p = the crop's mean GT pitch (as first defined; the v1/v1.1 rows before commit 145ed00 used the
median by mistake). An Archimedean spiral with the right pitch and no image input.

It uses two GT-derived quantities (the axis from the innermost GT wrap, and the GT pitch), so it
is a generous reference row, not a fair method.
"""
from __future__ import annotations

import numpy as np

from .build import SIZE, load_crop
from .rasterise import axis_centre, load_meshes

_centre = None


def crop_pitch(pitch: np.ndarray) -> float:
    """The crop's pitch for the prior: mean local GT pitch over voxels where it is defined."""
    return float(np.nanmean(pitch))


def spiral_prior(crop_id: str) -> np.ndarray:
    global _centre
    if _centre is None:
        _centre = axis_centre(load_meshes())
    c = load_crop(crop_id)
    pitch = crop_pitch(c["pitch"])
    z0, y0, x0 = c["origin"]
    k = np.empty((SIZE,) * 3, np.int16)
    yy, xx = np.meshgrid(np.arange(y0, y0 + SIZE), np.arange(x0, x0 + SIZE), indexing="ij")
    for i in range(SIZE):
        cy, cx = _centre(np.array([z0 + i], float))
        r = np.hypot(yy - cy[0], xx - cx[0])
        k[i] = np.floor(r / pitch).astype(np.int16)
    return k
