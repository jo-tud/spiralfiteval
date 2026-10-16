"""Commit 145ed00 (bug review findings #5, #6): prior pitch is the mean; validate samples per crop."""
import numpy as np

from windbench.baseline import crop_pitch
from windbench.validate import _sample, crop_rng


def test_crop_pitch_is_mean_not_median():
    p = np.array([10.0, 10.0, 10.0, 40.0, np.nan], np.float32)
    assert crop_pitch(p) == 17.5


def test_crop_rng_independent_of_order():
    mask = np.zeros((20, 20, 20), bool)
    mask[2:18, 3:17, 4:16] = True
    a1 = _sample(mask, crop_rng("c05"), n=100)
    r = crop_rng("c01")
    _sample(mask, r, n=100)  # another crop drawn first must not change c05's draw
    a2 = _sample(mask, crop_rng("c05"), n=100)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, _sample(mask, crop_rng("c04"), n=100))


def test_sample_matches_argwhere():
    rng = np.random.default_rng(3)
    mask = rng.random((16, 17, 18)) > 0.6
    idx = np.argwhere(mask)
    want = idx[np.random.default_rng(7).choice(len(idx), 50, replace=False)]
    assert np.array_equal(_sample(mask, np.random.default_rng(7), n=50), want)
    assert np.array_equal(_sample(mask, np.random.default_rng(7), n=10**6), idx)
