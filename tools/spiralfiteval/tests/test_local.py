"""Local H5 -> H1 instrument on a synthetic spiral with one duplicated sector."""

import numpy as np
import pytest

from synth import spiral_xyz, write_fit
from spiralfiteval.fit import load_fit
from spiralfiteval.floors import axis_xy
from spiralfiteval.local import (GapCache, auroc, exposure, rotate, rotation_p, score_exposure,
                                 tube, vertex_key)

PHIS = [0.0] + list(np.deg2rad(np.arange(30, 331, 5)))
SECTOR = (60 / 180, 90 / 180)      # w4 is moved onto w5 in these columns (t in [1/3, 1/2))


@pytest.fixture(scope="module")
def dup_fit(tmp_path_factory):
    root = tmp_path_factory.mktemp("dup")
    write_fit(root, jump=(60, 90, 1), windings=[4], rows=40)
    write_fit(root, windings=[2, 3, 5, 6, 7], rows=40)
    return load_fit(root)


def pairs(n=72, seed=0):
    """One-wrap click pairs from sheet 4 to sheet 5 at evenly spread angles, z 150-250."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) + 0.5) / n
    z = rng.uniform(150, 250, n)
    return t, {"A": spiral_xyz(4 + t, z), "B": spiral_xyz(5 + t, z)}


def test_auroc_ties_and_order():
    assert auroc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert auroc([0, 0, 0, 0], [0, 1, 0, 1]) == 0.5
    assert auroc([1, 0], [0, 1]) == 0.0
    assert np.isnan(auroc([1, 2], [1, 1]))
    assert rotation_p(0.9, [0.5, 0.6, 0.95]) == pytest.approx(2 / 4)


def test_rotation_keeps_radius_and_height(dup_fit):
    _, p = pairs()
    q = rotate(dup_fit, p["A"], 1.0)
    ra = np.linalg.norm(p["A"][:, :2] - axis_xy(dup_fit, p["A"][:, 2]), axis=1)
    rq = np.linalg.norm(q[:, :2] - axis_xy(dup_fit, q[:, 2]), axis=1)
    assert np.allclose(ra, rq) and np.allclose(p["A"][:, 2], q[:, 2])
    assert np.allclose(rotate(dup_fit, q, -1.0), p["A"])


def test_tube_is_the_segment_neighbourhood(dup_fit):
    A, B = spiral_xyz(np.array([4.9]), np.array([200.0]))[0], spiral_xyz(np.array([5.9]), np.array([200.0]))[0]
    idx = tube(dup_fit, A, B, 6.0)
    assert len(idx) > 0
    P = dup_fit.xyz[idx].astype(np.float64)
    d = B - A
    s = np.clip((P - A) @ d / (d @ d), 0, 1)
    assert np.linalg.norm(P - (A + s[:, None] * d), axis=1).max() <= 6.0
    # brute force agrees
    allP = dup_fit.xyz.astype(np.float64)
    s = np.clip((allP - A) @ d / (d @ d), 0, 1)
    brute = np.flatnonzero(np.linalg.norm(allP - (A + s[:, None] * d), axis=1) <= 6.0)
    assert set(idx.tolist()) == set(brute.tolist())


def run(fit, t, p, slip, radius=10.0):
    gaps = GapCache(fit, max_gap=30.0)
    ex = exposure(fit, p, PHIS, radius=radius, gaps=gaps, key=vertex_key(len(fit.f)), cap=64)
    sc = score_exposure(ex["D"], ex["n"], slip)
    return ex, sc


def test_slips_inside_the_duplicate_sector_are_found(dup_fit):
    t, p = pairs()
    inside = (t >= SECTOR[0]) & (t < SECTOR[1])
    ex, sc = run(dup_fit, t, p, inside)
    D0 = ex["D"][0]
    assert (D0[inside] > 0).all() and (D0[~inside & ((t < 0.25) | (t > 0.6))] == 0).all()
    assert sc["auroc"][0] > 0.95
    assert rotation_p(sc["auroc"][0], sc["auroc"][1:]) == pytest.approx(1 / len(PHIS))
    assert np.nanmean(sc["auroc"][1:]) < 0.75


def test_slips_away_from_duplicates_are_not_found(dup_fit):
    t, p = pairs()
    away = (t >= 0.8) & (t < 0.95)
    ex, sc = run(dup_fit, t, p, away)
    assert sc["auroc"][0] <= 0.5
    assert rotation_p(sc["auroc"][0], sc["auroc"][1:]) > 0.2
