import numpy as np
import pytest

from synth import spiral_xyz, write_fit
from spiralfiteval.fit import check_seams, load_fit, read_grids
from spiralfiteval.score import surface_distance, trace_switches, winding_identity
from windingcheck.chains import from_arrays
from windingcheck.score import build_truth


@pytest.fixture
def fit(tmp_path):
    return load_fit(write_fit(tmp_path / "fit"))


def test_seam_sense_export_layout(tmp_path):
    assert check_seams(read_grids(write_fit(tmp_path / "a")))["sense"] == 1
    assert check_seams(read_grids(write_fit(tmp_path / "b", export_layout=False)))["sense"] == 1


def test_projection_recovers_spiral_time(fit):
    f_true = np.array([3.25, 4.5, 5.9, 6.0])
    f, dn = fit.project(spiral_xyz(f_true, np.array([120.0, 130, 140, 150])), 3.0)
    assert np.allclose(f, f_true, atol=1 / 180 + 1e-9)
    assert np.all(dn < 1e-3)


def radial_chain(f0, n, z, jitter=0.0):
    f = f0 + np.arange(n)
    xyz = spiral_xyz(f, np.full(n, z))
    return xyz, np.arange(n, dtype=float)


def test_winding_identity_perfect_and_shifted(tmp_path):
    fit = load_fit(write_fit(tmp_path / "fit"))
    xyz, lev = radial_chain(2.3, 5, 120.0)
    xyz2, lev2 = radial_chain(2.6, 4, 140.0)
    truth = build_truth(from_arrays(np.r_[xyz, xyz2], np.r_[lev, lev2], np.r_[[0] * 5, [1] * 4]))
    r = winding_identity(fit, truth)
    assert r["exact"] == 1.0 and r["human_pairs_covered"] == 1.0
    # human says the outermost click is two wraps out, not one: every pair with it is off by 1
    lev_bad = lev.copy()
    lev_bad[-1] += 1
    truth = build_truth(from_arrays(xyz, lev_bad, [0] * 5))
    r = winding_identity(fit, truth)
    assert r["tests"] == 10 and r["exact"] == pytest.approx(6 / 10)


def test_switches_zero_on_clean_fit_and_two_on_planted(tmp_path):
    clean = load_fit(write_fit(tmp_path / "clean"))
    bad = load_fit(write_fit(tmp_path / "bad", jump=(60, 90, 1)))
    f = np.linspace(4.2, 4.8, 400)  # one same-sheet trace through the shifted sector
    tr = spiral_xyz(f, np.full_like(f, 130.0))
    r0 = trace_switches(clean, [tr])
    r1 = trace_switches(bad, [tr])
    assert r0["switches"] == 0 and r0["coverage"] == 1.0
    assert r1["switches"] == 2 and len(r1["sites"]) == 2 and r1["traces_with_switch"] == 1.0
    assert r1["post_hoc_clean_switches"] == 2
    # crossing the seam between windings is not a switch
    f = np.linspace(4.8, 5.2, 200)
    assert trace_switches(clean, [spiral_xyz(f, np.full_like(f, 130.0))])["switches"] == 0


def test_surface_distance(fit):
    f = np.linspace(3.0, 3.9, 50)
    on = spiral_xyz(f, np.full_like(f, 120.0))
    r = surface_distance(fit, on)
    assert r["median"] < 0.05 and r["within_2"] == 1.0
    c = np.array([500.0, 500.0, 0])
    radial = (on - c) * np.array([1, 1, 0])
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    r = surface_distance(fit, on + 3 * radial)
    assert 2.5 < r["median"] < 3.5 and r["within_2"] == 0.0 and r["within_4"] == 1.0


def test_sheet_spacing_flags_duplicate(tmp_path):
    from spiralfiteval.score import sheet_spacing
    clean = load_fit(write_fit(tmp_path / "clean"))
    r = sheet_spacing(clean, max_gap=30)
    assert r["duplicate"] == 0.0 and 17 < r["median"] < 23  # pitch 20
    dup = load_fit(write_fit(tmp_path / "dup", jump=(60, 90, 1), windings=[4]) and
                   write_fit(tmp_path / "dup", windings=[2, 3, 5, 6, 7]))
    r = sheet_spacing(dup, max_gap=30)
    # winding 4 moved onto winding 5 in 30 of 180 columns: ~1/6 of w4 and w5 vertices duplicated
    assert 0.03 < r["duplicate"] < 0.08


def test_npz_and_refine_match_tifxyz(tmp_path):
    import numpy as np
    from spiralfiteval.fit import read_grids
    root = write_fit(tmp_path / "fit", cols=36)  # coarse: ~24 voxel steps
    grids = read_grids(root)
    np.savez(tmp_path / "g.npz", **{f"w{w:03d}": np.moveaxis(P, -1, 0) for w, (P, v) in grids.items()})
    a, b = load_fit(root), load_fit(tmp_path / "g.npz")
    assert a.step <= 5.0 and np.allclose(a.f, b.f) and np.allclose(a.xyz, b.xyz)
    f_true = np.array([3.31, 4.52, 5.93])
    f, dn = a.project(spiral_xyz(f_true, np.array([120.0, 130, 140])), 3.0)
    assert np.allclose(f, f_true, atol=0.03) and np.all(dn < 1.0)


def test_flicker_is_not_a_switch():
    import numpy as np
    from spiralfiteval.score import _switch_steps
    p = np.c_[np.arange(40.0), np.zeros(40), np.zeros(40)]
    f = np.linspace(3.0, 3.2, 40)
    g = f.copy(); g[20:23] += 1          # 3-voxel excursion to the next wrap and back
    assert len(_switch_steps(p, g, jump=0.5, flicker=10, bridge=10)[0]) == 0
    g = f.copy(); g[20:] += 1            # a lasting switch
    assert len(_switch_steps(p, g, jump=0.5, flicker=10, bridge=10)[0]) == 1
    g[15:20] = np.nan                    # off the fit for 5 voxels, back on the next wrap
    assert len(_switch_steps(p, g, jump=0.5, flicker=10, bridge=10)[0]) == 1


def test_sheet_spacing_wide_gap_does_not_depend_on_k():
    """Regression test for 3d32d3f: two flat sheets 25 voxels apart on a 4.45 voxel grid. The 48 nearest vertices of
    an interior vertex all lie on its own sheet (radius ~17 vx), so the old fixed-k search
    reported no neighbour; the widening search must find the true gap whatever k it starts at."""
    from spiralfiteval.fit import Fit
    from spiralfiteval.score import sheet_spacing
    step, gap, n = 4.45, 25.0, 60
    y, z = np.meshgrid(np.arange(n) * step, np.arange(n) * step, indexing="ij")
    t = np.broadcast_to(np.arange(n) / n, (n, n))
    xyz = np.concatenate([np.stack([np.full(n * n, x), y.ravel(), z.ravel()], 1) for x in (0.0, gap)])
    f = np.r_[t.ravel(), 1 + t.ravel()]
    nrm = np.tile([1.0, 0, 0], (2 * n * n, 1))
    fit = Fit("planes", xyz.astype(np.float32), f, (f // 1).astype(np.int32),
              nrm.astype(np.float32), [0, 1], step)
    r = sheet_spacing(fit, max_gap=60)
    assert r["no_neighbour"] == 0.0 and r["duplicate"] == 0.0
    assert r["median"] == pytest.approx(gap, abs=1e-3) and r["p10"] == pytest.approx(gap, abs=1e-3)
    p = fit.xyz[:50]
    for k in (4, 48, 300):
        _, dn = fit.project(p, 60.0, exclude_f=fit.f[:50], k=k, chunk=7)
        assert np.allclose(dn, gap, atol=1e-3)
    # beyond max_gap there is no neighbour, and the search still ends
    r = sheet_spacing(fit, max_gap=20)
    assert r["no_neighbour"] == 1.0
