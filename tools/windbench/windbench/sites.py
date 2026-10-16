"""GT error sites (README, "Definitions"): T = touch, N = naive-signal failure, X = order violation."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

MIN_VOXELS = 27
MERGE_DIST = 10        # regions closer than this (voxels) are one site
TOUCH_PITCH = 3.0      # voxels
COS_LOW, COS_HIGH = 64, 191   # Lasagna cos_u8 thresholds (cos(2 pi w) < -0.5 / > +0.5)
SHEET_TOL, MIDGAP_TOL = 0.1, 0.4
PRIORITY = ("X", "T", "N")    # a voxel in several masks keeps the first type


def type_masks(f, k, sheet, pitch, order_violation, cos_u8) -> dict[str, np.ndarray]:
    known = np.isfinite(f)
    frac = np.abs(f - k)
    touch = (known & (pitch <= TOUCH_PITCH)) | (sheet == -2)
    sheet_vox = known & (frac <= SHEET_TOL)
    mid_vox = known & (frac >= MIDGAP_TOL)
    naive = (sheet_vox & (cos_u8 < COS_LOW)) | (mid_vox & (cos_u8 > COS_HIGH))
    return {"X": order_violation, "T": touch, "N": naive}


def extract_sites(masks: dict[str, np.ndarray], k: np.ndarray, origin_l2) -> tuple[np.ndarray, list[dict]]:
    """Label volume (0 none, i > 0 site i) and a list of site records."""
    labels = np.zeros(next(iter(masks.values())).shape, np.int32)
    taken = np.zeros(labels.shape, bool)
    sites = []
    ball = ndimage.generate_binary_structure(3, 1)
    for t in PRIORITY:
        m = masks[t] & ~taken
        if not m.any():
            continue
        # merge regions within MERGE_DIST: label the dilated mask, keep original voxels
        grown = ndimage.binary_dilation(m, ball, iterations=MERGE_DIST // 2)
        lab, n = ndimage.label(grown, structure=np.ones((3, 3, 3), bool))
        lab = np.where(m, lab, 0)
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        objs = ndimage.find_objects(lab)
        for i in range(1, n + 1):
            if counts[i] < MIN_VOXELS or objs[i - 1] is None:
                continue
            sl = objs[i - 1]
            sub = lab[sl] == i
            sid = len(sites) + 1
            labels[sl][sub] = sid
            idx = np.argwhere(sub) + [s.start for s in sl]
            kk = k[tuple(idx.T)]
            kk = kk[kk >= 0]
            cen = idx.mean(0)
            sites.append({
                "id": sid, "type": t, "voxels": int(counts[i]),
                "centroid_crop": [round(float(x), 1) for x in cen],
                "centroid_l2": [round(float(x + o), 1) for x, o in zip(cen, origin_l2)],
                "bbox_crop": [[s.start for s in sl], [s.stop for s in sl]],
                "wraps": sorted(set(int(x) for x in np.unique(kk))) if len(kk) else [],
            })
        taken |= labels > 0
    return labels, sites
