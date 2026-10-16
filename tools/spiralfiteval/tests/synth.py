"""A synthetic spiral fit on disk: one tifxyz mesh per winding of an Archimedean spiral."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CENTER = (500.0, 500.0)
PITCH = 20.0        # voxels per turn
R0 = 40.0


def spiral_xyz(f, z):
    """Point at spiral time f (turns) and height z. Radius grows outward with f."""
    th = 2 * np.pi * f
    r = R0 + PITCH * f
    return np.stack([CENTER[0] + r * np.cos(th), CENTER[1] + r * np.sin(th), z], axis=-1)


def write_fit(root, windings=range(2, 8), *, cols=180, rows=20, z0=100.0, dz=5.0,
              export_layout=True, jump=None):
    """Write w{NNN}/ meshes. ``export_layout`` stores columns outermost-first as villa does.
    ``jump`` = (col0, col1, dw): in that angular sector every winding is moved dw turns, so a
    sheet traced through the sector meets the fit's neighbouring winding (planted switches)."""
    import tifffile
    root = Path(root)
    for w in windings:
        t = np.arange(cols) / cols
        f = w + t[None, :].repeat(rows, 0)
        pos = f.copy()
        if jump is not None:
            pos[:, jump[0]:jump[1]] += jump[2]
        z = z0 + dz * np.arange(rows)[:, None].repeat(cols, 1)
        P = spiral_xyz(pos, z)
        meta = {"type": "seg", "format": "tifxyz", "scale": [1.0, 1.0]}
        if export_layout:
            P = P[:, ::-1]
            meta["grid_orientation"] = {"z_direction_is_top_to_bottom": True}
        d = root / f"w{w:03d}"
        d.mkdir(parents=True, exist_ok=True)
        for c, i in zip("xyz", range(3)):
            tifffile.imwrite(d / f"{c}.tif", P[..., i].astype(np.float32))
        (d / "meta.json").write_text(json.dumps(meta))
    return root
