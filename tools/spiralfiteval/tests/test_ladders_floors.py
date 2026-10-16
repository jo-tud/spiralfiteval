"""Ladder H1 (published protocol), per-winding H5 and the floors, on synthetic fits."""
import json

import numpy as np
import pytest

from synth import PITCH, spiral_xyz, write_fit
from spiralfiteval.evaluate import floors
from spiralfiteval.fit import load_fit
from spiralfiteval.ladders import ladder_slips, load_ladders, windings_per_wrap
from spiralfiteval.score import sheet_spacing_by_winding


def ladder_file(path, ladders):
    doc = {"vc_pointcollections_json_version": 1, "collections": {
        str(i): {"name": f"l{i}", "points": {str(k): {"p": list(map(float, p)), "wind_a": float(w)}
                                           for k, (p, w) in enumerate(zip(xyz, ws))}}
        for i, (xyz, ws) in enumerate(ladders)}}
    path.write_text(json.dumps(doc))
    return path


def radial_ladder(f0, n, z):
    f = f0 + np.arange(n)
    return spiral_xyz(f, np.full(n, z)), np.arange(n, dtype=float)


@pytest.fixture
def fit(tmp_path):
    return load_fit(write_fit(tmp_path / "fit"))


def test_ladder_selection_and_clean_fit(tmp_path, fit):
    a = radial_ladder(2.3, 5, 120.0)
    b = radial_ladder(2.6, 4, 140.0)
    tilted = (a[0] + np.array([0, 0, 1.0]) * np.arange(5)[:, None], a[1])  # z spread 4: dropped
    lad = load_ladders(ladder_file(tmp_path / "l.json", [a, b, tilted]), (90, 200))
    assert len(lad) == 2
    r = ladder_slips(fit, lad)
    assert r["pairs"] == 7 and r["pairs_assigned"] == 7
    assert r["published"]["slips"] == 0 and r["tool"]["slips"] == 0
    assert windings_per_wrap(fit, lad)["ratio"] == pytest.approx(1.0, abs=0.01)


def test_ladder_rules_on_a_mislabelled_and_a_reversed_ladder(tmp_path, fit):
    xyz, w = radial_ladder(2.3, 5, 120.0)
    w_bad = w.copy()
    w_bad[-1] += 1                      # last step labelled two wraps: one slip under both rules
    rev = (xyz[::-1], w)                # clicked inward: the published rule flips the sign
    lad = load_ladders(ladder_file(tmp_path / "l.json", [(xyz, w_bad), rev]), (90, 200))
    r = ladder_slips(fit, lad)
    assert r["published"]["S"] == [1, 0]
    assert r["tool"]["S"] == [1, 4]


def test_windings_per_wrap_sees_surplus_windings(tmp_path):
    # a fit with twice the windings: spiral time doubles between the same sheets
    fit = load_fit(write_fit(tmp_path / "fit", windings=range(4, 16)))
    xyz, w = radial_ladder(2.3, 5, 120.0)
    lad = [{"xyz": xyz, "w": w / 2.0}]   # each real sheet is labelled as half a wrap
    assert windings_per_wrap(fit, lad)["ratio"] == pytest.approx(2.0, abs=0.02)


def test_per_winding_spacing_locates_the_duplicate(tmp_path):
    root = write_fit(tmp_path / "dup", jump=(60, 90, 1), windings=[4])
    write_fit(root, windings=[2, 3, 5, 6, 7])
    r = sheet_spacing_by_winding(load_fit(root), per_winding=1000, max_gap=30, blocks=((2, 3), (4, 5), (6, 7)),
                                 fit_median_gap=PITCH)
    pw = r["per_winding"]
    assert pw[2]["duplicate"] == 0.0 and pw[7]["duplicate"] == 0.0
    assert 0.1 < pw[4]["duplicate"] < 0.25 and 0.1 < pw[5]["duplicate"] < 0.25
    assert r["share_of_duplicates"]["w004-005"] == pytest.approx(1.0)
    assert r["stacked_outer"]["n"] == 0


def test_floors_on_a_clean_fit(fit):
    f = np.linspace(3.0, 5.9, 400)
    traces = [spiral_xyz(f, np.full_like(f, z)) for z in (115.0, 130.0, 150.0)]
    out = floors(fit, fibers=traces, gap=PITCH, seeds=3)
    h3 = out["H3"]
    assert h3["best_shift"] == 0.0 and h3["rise_at_4"] > 3.0
    assert h3["jitter"]["value"] < 0.05 and h3["jitter"]["margin"] > 2.0
    assert h3["random"]["margin"] > 2.0
    assert out["H2"]["shift"]["0.0"] == 0.0


def test_near_click_shift_finds_a_frame_offset(fit):
    from spiralfiteval.floors import near_reference_shift, radial
    f = np.linspace(3.0, 5.9, 4000)
    fib = spiral_xyz(f, np.full_like(f, 130.0))
    clicks = spiral_xyz(np.array([3.5, 4.5, 5.5]), np.full(3, 130.0))
    on = near_reference_shift(fit, fib, clicks)
    assert on["n"] > 20 and on["best_shift"] == 0.0 and on["click_median"] < 0.05
    off = fib + 4.0 * radial(fit, fib)                 # fibers 4 vx outside their sheet
    r = near_reference_shift(fit, off, clicks)
    assert r["best_shift"] == -4.0 and r["at_0"] > 3.0 and r["at_best"] < 0.5
