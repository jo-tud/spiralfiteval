import numpy as np
from scipy import ndimage

from windbench.zcheck import AFFINE_791_TO_2399, best_shift, native_to_l2


def test_best_shift_recovers_planted_offset():
    # bright sheet (plane x = 20.5 + 0.3 z) in a 40^3 volume; vertices traced 1.5 voxels short in x
    z, y, x = np.mgrid[:40, :40, :40].astype(np.float32)
    vol = ndimage.gaussian_filter(np.exp(-0.5 * ((x - 20.5 - 0.3 * z) / 1.2) ** 2), 0.5).astype(np.float32)
    zz, yy = np.meshgrid(np.arange(10, 30, 1.0), np.arange(10, 30, 1.0), indexing="ij")
    pts = np.stack([zz.ravel(), yy.ravel(), 20.5 + 0.3 * zz.ravel() - 1.5], -1)
    s, best, s0 = best_shift(vol, pts, half=3.0)
    assert best > s0
    # only the normal component is constrained: the shift must land the points back on the sheet
    assert abs((pts[:, 2] + s[2]) - (20.5 + 0.3 * (pts[:, 0] + s[0]))).max() < 0.5


def test_affine_scale_matches_pixel_sizes():
    lin = AFFINE_791_TO_2399[:, :3]
    assert abs(abs(np.linalg.det(lin)) ** (1 / 3) - 7.91 / 2.399) < 0.03
    p = native_to_l2(np.array([[1000.0, 1000.0, 1000.0]]))
    assert p.shape == (1, 3)
