# windbench

Winding benchmark on PHerc. 1667: ground-truth wrap index (k) and winding position (f) on 512³
crops at 9.6 µm, the Lasagna inputs for the same crops, GT error sites, and metrics. CPU only.
In this repository windbench ships because spiralfiteval uses its trace metrics; the benchmark
crops themselves are built with the commands below.

    PY=external/villa/vesuvius/.venv/bin/python      # numpy, scipy, zarr 3, s3fs, tifffile
    cd tools/windbench
    $PY -m windbench.cli fetch                        # 19 GT meshes, ~880 MB, idempotent
    $PY -m windbench.cli build                        # crops in manifest/crops.json, idempotent
    $PY -m windbench.cli validate --out validation.json
    $PY -m windbench.cli baseline /tmp/prior && $PY -m windbench.cli score /tmp/prior
    $PY -m pytest -q tests

Derived data goes to `<main checkout>/data/windbench/` (gitignored; override with `WINDBENCH_DATA`).

## Using it from a method

```python
from windbench.build import load_crop, lasagna_l2
c = load_crop("c01")              # ct, f, k, sheet, pitch, sites, traces, origin, sites_meta
cos = lasagna_l2(c, "cos")        # Lasagna cos on the 512³ grid, in [0, 1]
# ... your method -> k_pred (int16, -1 = no prediction), defects (bool volume or Mx3 points)
np.savez_compressed("pred/c01.npz", k=k_pred, defects=defects)
```

Then `windbench score pred/`. The metric functions (`windbench.metrics`) take plain arrays
and are shared with spiralfiteval (`tools/spiralfiteval`):

- `wrap_accuracy(k_pred, k_gt, mask=None)`: accuracy up to one global offset; missing = wrong.
- `switch_rate(traces, k_volume)`: sheet switches per 1000 steps of ≤ 1 voxel along traces.
- `defect_pr(pred, site_labels, site_types, r=20)`: defect precision / recall vs GT error sites.

## What the GT is (and is not)

- Wrap index per mesh vertex comes from **turn counting**, not from the segment name: every
  mesh runs inward as its column index grows; the index drops by one each time the mesh
  crosses the seam half-plane at −90° around the scroll axis. w013, w018 and w023 are not
  single wraps: each spans about five turns and covers the "gap" wraps 14–17, 19–22 and 24–27.
  Labels are therefore wNNN-consistent along the spiral but shifted relative to the organisers'
  per-mesh seams; any k is only defined up to one global offset anyway.
- k jumps by one across the seam. Crops do not straddle it.
- Wraps ≥ 31 (outer turns) are excluded from v1 crops: their meshes lie on the wrong side of the
  next wrap for 5–26 % of vertices, and parts lie outside the 2.4 µm scan mask.

Licence: code MIT. Scan data and meshes: CC BY-NC 4.0 (scrollprize.org).
