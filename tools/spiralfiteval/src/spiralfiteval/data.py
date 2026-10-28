"""Public Paris 4 references: human winding chains (via windingcheck) and the organisers'
evaluation fibers. Small files, fetched once into ``data/spiral_datasets/PHercParis4``."""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from windingcheck.chains import load_pcl
from windingcheck.data import BASE, _get, default_root, fetch as fetch_windings

# eval fibers are traced on the 2.4 um scan 20260411134726; the spiral-input frame is its level 2.
# A level-L voxel k is the block mean of L0 voxels 2^L k ... 2^L k + 2^L - 1, so its centre is L0
# 2^L k + (2^L - 1) / 2. VC3D writes a point traced on a grid s times coarser as base = s * t, with
# no offset, so the true L0 position is p + (s - 1) / 2 and the level-2 index is
# (p + (s - 1) / 2 - 1.5) / 4. The Paris 4 fibers are traced at level 1 (s = 2): (p - 1) / 4.
EVAL_FIBER_SCALE = 0.25
ANNOTATION_FILES = ("relative_windings.json", "same_windings.json", "abs_winding.json")


def fetch(scroll="PHercParis4", root=None, *, fibers=True, workers=8) -> Path:
    """Human winding files (windingcheck) plus ``eval_fibers/`` (645 JSON files, ~290 MB)."""
    d = fetch_windings(scroll, root)
    if fibers:
        url = f"{BASE}/{scroll}/eval_fibers/"
        html = urllib.request.urlopen(url).read().decode()
        names = sorted(set(re.findall(r'href="([^"/]+\.json)"', html)))
        with ThreadPoolExecutor(workers) as ex:  # at most 8 parallel connections to the data server
            list(ex.map(lambda n: _get(url + n, d / "eval_fibers" / n), names))
    return d


def scroll_dir(scroll="PHercParis4", root=None) -> Path:
    return (Path(root) if root else default_root()) / scroll


def trace_scale(fiber: dict) -> float:
    """The grid the fiber was traced on, in L0 voxels per trace voxel: the most common
    ``trace_to_base_scale`` over its segments (larger wins a tie), 1 if no segment has one."""
    c = Counter(float(s["trace_to_base_scale"]) for s in
                (cp.get("segment_to_next") or {} for cp in fiber.get("control_points") or [])
                if s.get("trace_to_base_scale"))
    return max(c, key=lambda v: (c[v], v)) if c else 1.0


def fiber_to_fit_frame(line_points, scale: float) -> np.ndarray:
    """L0 base points of a fiber traced at ``scale`` -> level-2 voxel centres (see above)."""
    p = np.asarray(line_points, np.float64)
    return (p + (scale - 1) / 2 - 1.5) * EVAL_FIBER_SCALE


def eval_fibers(scroll="PHercParis4", root=None, *, z_range=None, min_points=20) -> list[dict]:
    """Eval fibers in the spiral-input frame (xyz, level-2 voxels), cut to ``z_range``.
    A fiber leaving and re-entering the range becomes several traces."""
    out = []
    for f in sorted((scroll_dir(scroll, root) / "eval_fibers").glob("*.json")):
        d = json.loads(f.read_text())
        p = fiber_to_fit_frame(d["line_points"], trace_scale(d))
        tag = (d.get("hv_classification") or {}).get("automatic_tag")
        inside = np.ones(len(p), bool) if z_range is None else \
            (p[:, 2] >= z_range[0]) & (p[:, 2] < z_range[1])
        cuts = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
        for a, b in zip(cuts[::2], cuts[1::2]):
            if b - a >= min_points:
                out.append({"name": f.stem, "tag": tag, "xyz": p[a:b]})
    return out


def annotation_points(scroll="PHercParis4", root=None) -> np.ndarray:
    """All human winding-annotation clicks (relative, same, absolute files), xyz."""
    d = scroll_dir(scroll, root)
    pts = [load_pcl(d / n).xyz for n in ANNOTATION_FILES if (d / n).exists()]
    return np.concatenate(pts) if pts else np.zeros((0, 3))
