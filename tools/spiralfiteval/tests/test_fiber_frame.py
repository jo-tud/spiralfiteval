"""Eval-fiber L0 -> level-2 map: fibers written the way VC3D writes
them land on the sheet they were traced on; the corner map ``p / 4`` misses it by a sub-voxel."""

import json

import numpy as np
import pytest

from synth import spiral_xyz, write_fit
from spiralfiteval.data import EVAL_FIBER_SCALE, eval_fibers, fiber_to_fit_frame, trace_scale
from spiralfiteval.fit import load_fit


def write_fiber(path, l2_xyz, scale):
    """A vc3d_fiber JSON for points at level-2 voxel centres ``l2_xyz``, traced on a grid ``scale``
    times coarser than L0: true L0 = 4 q + 1.5, trace index t = (L0 - (s - 1) / 2) / s, and VC3D
    writes base = s * t."""
    t = (4 * np.asarray(l2_xyz) + 1.5 - (scale - 1) / 2) / scale
    base = scale * t
    seg = {"interp_mode": "trace", "trace_to_base_scale": scale}
    cps = [{"position": base[0].tolist(), "segment_to_next": seg},
           {"position": base[-1].tolist(), "segment_to_next": None}]
    path.write_text(json.dumps({"type": "vc3d_fiber", "line_points": base.tolist(),
                                "control_points": cps,
                                "hv_classification": {"automatic_tag": "H"}}))


def on_winding(w, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return spiral_xyz(w + rng.uniform(0.02, 0.98, n), np.linspace(105, 190, n))


@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0])
def test_map_inverts_vc3d_write(scale):
    q = on_winding(4)
    l0 = 4 * q + 1.5
    base = l0 - (scale - 1) / 2       # s * t
    assert np.allclose(fiber_to_fit_frame(base, scale), q, atol=1e-9)


def test_trace_scale_reads_the_segments():
    seg = lambda s, m="trace": {"segment_to_next": {"interp_mode": m, "trace_to_base_scale": s}}
    assert trace_scale({"control_points": [seg(2.0), seg(2.0), seg(1.0, "cspline"),
                                           {"segment_to_next": None}]}) == 2.0
    assert trace_scale({"control_points": [seg(1.0), seg(4.0)]}) == 4.0
    assert trace_scale({"control_points": []}) == 1.0


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_pc2_fibers_land_on_their_sheet(tmp_path, scale):
    fit = load_fit(write_fit(tmp_path / "fit"))
    fibers = tmp_path / "refs" / "PHercParis4" / "eval_fibers"
    fibers.mkdir(parents=True)
    q = on_winding(4)
    write_fiber(fibers / "f0.json", q, scale)
    (tr,) = eval_fibers("PHercParis4", tmp_path / "refs")
    _, dn = fit.project(tr["xyz"], 10.0)
    assert np.nanmedian(np.abs(dn)) <= 0.05
    # the corner map p / 4 that shipped in 63cf0243: off by 0.25 (s = 2) or 0.375 (s = 1) per axis
    raw = np.asarray(json.loads((fibers / "f0.json").read_text())["line_points"])
    _, dn_corner = fit.project(raw * EVAL_FIBER_SCALE, 10.0)
    lo, hi = {2.0: (0.15, 0.35), 1.0: (0.25, 0.55)}[scale]
    assert lo <= np.nanmedian(np.abs(dn_corner)) <= hi
