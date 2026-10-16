"""Public Paris 4 references: human winding chains (via windingcheck) and the organisers'
evaluation fibers. Small files, fetched once into ``data/spiral_datasets/PHercParis4``."""

from __future__ import annotations

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from windingcheck.chains import load_pcl
from windingcheck.data import BASE, _get, default_root, fetch as fetch_windings

# eval fibers are traced on the 2.4 um scan 20260411134726; the spiral-input frame is its level 2
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


def eval_fibers(scroll="PHercParis4", root=None, *, z_range=None, min_points=20) -> list[dict]:
    """Eval fibers in the spiral-input frame (xyz, level-2 voxels), cut to ``z_range``.
    A fiber leaving and re-entering the range becomes several traces."""
    out = []
    for f in sorted((scroll_dir(scroll, root) / "eval_fibers").glob("*.json")):
        d = json.loads(f.read_text())
        p = np.asarray(d["line_points"], np.float64) * EVAL_FIBER_SCALE
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
