import numpy as np
import pytest

from windbench.build import lasagna_box, upsample_to_l2


@pytest.mark.parametrize("level", [3, 4])
def test_upsample_linear_ramp_is_exact(level):
    """A field linear in level-0 coordinates must come back exactly on the level-2 grid."""
    origin, size = (1024, 640, 2048), 32
    lo, hi = lasagna_box(origin, size, level)
    s = 2 ** level
    # value = level-0 coordinate of each level-L voxel centre along axis 1
    j = np.arange(lo[1], hi[1])
    centre_l0 = s * j + (s - 1) / 2
    block = np.broadcast_to(centre_l0[None, :, None], tuple(h - l for l, h in zip(lo, hi))).copy()
    up = upsample_to_l2(block, lo, level, origin, size)
    expect = 4 * np.arange(origin[1], origin[1] + size) + 1.5
    assert np.allclose(up[0, :, 0], expect, atol=1e-3)


@pytest.mark.parametrize("level", [3, 4])
def test_box_covers_crop(level):
    origin, size = (0, 128, 256), 512
    lo, hi = lasagna_box(origin, size, level)
    s = 2 ** level
    for ax in range(3):
        first = (4 * origin[ax] + 1.5 - (s - 1) / 2) / s
        last = (4 * (origin[ax] + size - 1) + 1.5 - (s - 1) / 2) / s
        assert lo[ax] <= np.floor(first) and np.ceil(last) < hi[ax]
