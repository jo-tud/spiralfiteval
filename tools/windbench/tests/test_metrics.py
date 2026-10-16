import numpy as np
import pytest

from windbench.metrics import UNKNOWN, defect_pr, densify, switch_rate, wrap_accuracy


def layered(shape=(16, 32, 32), pitch=4, k0=11):
    """Flat sheets along y: k = k0 + y // pitch."""
    k = np.zeros(shape, np.int16)
    k[:] = (k0 + np.arange(shape[1]) // pitch)[None, :, None]
    return k


@pytest.mark.parametrize("c", [-7, 0, 3])
def test_accuracy_offset_invariant(c):
    k = layered()
    r = wrap_accuracy(k + c, k)
    assert r["acc"] == 1.0 and r["offset"] == c and r["coverage"] == 1.0


def test_accuracy_planted_wrong_region():
    k = layered()
    p = k.copy() + 5
    p[:4] += 1  # 4 of 16 z-slices off by one wrap -> 25 % wrong
    assert wrap_accuracy(p, k)["acc"] == pytest.approx(0.75)


def test_accuracy_unknown_ignored_and_missing_counted():
    k = layered()
    k[:, :, :8] = UNKNOWN  # GT unknown: ignored
    p = layered()
    p[:, :, 8:16] = UNKNOWN  # no prediction on a third of the known voxels: counted wrong
    r = wrap_accuracy(p, k)
    assert r["n"] == 16 * 32 * 24
    assert r["acc"] == pytest.approx(2 / 3) and r["coverage"] == pytest.approx(2 / 3)


def test_densify_step():
    d = densify(np.array([[0, 0, 0], [0, 0, 10.5]]), 1.0)
    assert np.all(np.linalg.norm(np.diff(d, axis=0), axis=1) <= 1.0 + 1e-9)


def test_switches_planted_splice():
    k = layered()
    # Trace along x on sheet y=6 (k=12): 31 steps. Prediction swaps wrap for x in [10, 20).
    tr = [np.array([[8, 6, 0], [8, 6, 31]], float)]
    assert switch_rate(tr, k)["switches"] == 0
    p = k.copy()
    p[:, 6, 10:20] = 13
    r = switch_rate(tr, p)
    assert r["switches"] == 2 and r["steps"] == 31


def test_switches_trace_crossing_sheets():
    # A predicted surface tilted across GT sheets: going y 0 -> 31 crosses 7 boundaries.
    k = layered()
    tr = [np.array([[8, 0, 5], [8, 31, 5]], float)]
    assert switch_rate(tr, k)["switches"] == 7


def test_switches_unknown_steps_skipped():
    k = layered()
    k[:, :, 10:] = UNKNOWN
    r = switch_rate([np.array([[8, 6, 0], [8, 6, 31]], float)], k)
    assert r["steps"] == 9 and r["switches"] == 0


def test_defect_pr_constructed():
    sites = np.zeros((64, 64, 64), np.int32)
    sites[10:13, 10:13, 10:13] = 1
    sites[50:53, 50:53, 50:53] = 2
    types = {1: "T", 2: "N"}
    pred = np.array([[11, 11, 30], [5, 60, 5]])  # 1st within 20 of site 1; 2nd far from both
    r = defect_pr(pred, sites, types, r=20)
    assert r["recall"] == 0.5 and r["precision"] == 0.5
    assert r["per_type"]["T"]["recall"] == 1.0 and r["per_type"]["N"]["recall"] == 0.0


def test_defect_pr_volume_components():
    sites = np.zeros((64, 64, 64), np.int32)
    sites[30:33, 30:33, 30:33] = 1
    pred = np.zeros_like(sites, bool)
    pred[30:40, 31, 31] = True  # one component, overlaps the site
    pred[0:2, 0:2, 0:2] = True  # second component, far
    r = defect_pr(pred, sites, r=20)
    assert r["n_pred"] == 2 and r["precision"] == 0.5 and r["recall"] == 1.0
