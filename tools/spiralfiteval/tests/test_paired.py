from pathlib import Path

import numpy as np
import pytest
from synth import spiral_xyz, write_fit

from spiralfiteval.fit import load_fit
from spiralfiteval.paired import chain_parity_ids, fit_tests, paired_from_tests, paired_slips
from windingcheck.chains import from_arrays
from windingcheck.score import build_truth


def chains(n_chains=10, n=4, f0=2.0, t0=0.30, dt=0.05):
    """n_chains chains of n clicks one wrap apart, each click a little further round the
    turn (``dt``), so a chain crosses the angular sector a planted jump shifts. The fit's
    Δf is 1 + dt per step, which still rounds to the human Δw of 1."""
    xyz, lev, ch = [], [], []
    for c in range(n_chains):
        k = np.arange(n)
        f = f0 + k + t0 + dt * k
        xyz.append(spiral_xyz(f, np.full(n, 105.0 + 2.0 * c)))
        lev.append(k.astype(float))
        ch.append([c] * n)
    return build_truth(from_arrays(np.concatenate(xyz), np.concatenate(lev),
                                  np.concatenate(ch)))


def test_parity_follows_the_chain_index():
    truth = chains(6)
    assert chain_parity_ids(truth, "odd") == [1, 3, 5]
    assert chain_parity_ids(truth, "even") == [0, 2, 4]
    assert chain_parity_ids(truth, "all") is None
    # a band restriction comes first, parity second
    assert chain_parity_ids(truth, "odd", ids=[2, 3, 4, 5]) == [3, 5]
    assert chain_parity_ids(truth, "all", ids=[2, 3]) == [2, 3]
    with pytest.raises(ValueError):
        chain_parity_ids(truth, "sometimes")


def test_paired_difference_separates_a_worse_fit(tmp_path):
    """A fit with a shifted sector slips where a clean one does not, and the paired CI
    excludes 0 even though a single fit's own CI on 10 chains is wide."""
    # a winding range wider than the clicks need, so the shifted sector still has a sheet
    # under every click: the bad fit is wrong about the wrap, not missing
    wide = range(1, 9)
    clean = load_fit(write_fit(tmp_path / "clean", windings=wide))
    bad = load_fit(write_fit(tmp_path / "bad", windings=wide, jump=(60, 90, 1)))
    truth = chains(10)
    res = paired_slips([("clean", clean), ("bad", bad)], truth, tol_n=3.0, n_boot=2000)
    assert res["unit"] == "human chain" and res["chains"] == 10
    assert res["common"]["clean"]["slips_per_wrap"] == 0.0
    assert res["common"]["bad"]["slips_per_wrap"] > 0.0
    d = res["differences"]["bad-clean"]
    assert d["point"] > 0 and d["excludes_zero"] and d["share_below_zero"] == 0.0


def test_paired_difference_is_zero_for_a_fit_against_itself(tmp_path):
    fit = load_fit(write_fit(tmp_path / "fit"))
    truth = chains(8)
    res = paired_slips([("a", fit), ("b", fit)], truth, tol_n=3.0, n_boot=500)
    d = res["differences"]["b-a"]
    assert d["point"] == 0.0 and d["ci95"] == [0.0, 0.0] and not d["excludes_zero"]


def test_common_tests_are_the_intersection(tmp_path):
    """A fit that covers fewer clicks shrinks the paired test set, and the per-fit
    standalone numbers keep their own (larger) test counts."""
    fit = load_fit(write_fit(tmp_path / "fit"))
    narrow = load_fit(write_fit(tmp_path / "narrow", windings=range(2, 5)))
    truth = chains(6)
    res = paired_slips([("fit", fit), ("narrow", narrow)], truth, tol_n=3.0, n_boot=200)
    assert res["common_adjacent_tests"] <= res["own"]["fit"]["adjacent_tests"]
    assert res["own"]["narrow"]["clicks_on_fit"] < res["own"]["fit"]["clicks_on_fit"]


def test_unique_names_separates_fits_with_the_same_stem():
    """Regression: ``<run>/meshes/mesh`` is the usual layout, so ``Path(src).stem`` gave
    every run the name ``mesh``. The names key the paired result, so that compared one fit
    with itself and returned +0.0 % with a CI of [0, 0] - a false no-go."""
    from spiralfiteval.fit import unique_names
    srcs = ["fits/A/meshes/mesh", "fits/B/meshes/mesh", "fits/C/meshes/mesh"]
    assert [Path(s).stem for s in srcs] == ["mesh"] * 3        # the bug
    assert unique_names(srcs) == ["A/meshes/mesh", "B/meshes/mesh", "C/meshes/mesh"]
    assert unique_names(["x/A.npz", "y/B.npz"]) == ["A.npz", "B.npz"]  # shortest that separates
    assert unique_names(srcs, ["A", "B", "C"]) == ["A", "B", "C"]      # explicit wins
    with pytest.raises(ValueError):
        unique_names(srcs, ["A", "A", "C"])                            # duplicate labels
    with pytest.raises(ValueError):
        unique_names(srcs, ["A", "B"])                                 # count mismatch
    with pytest.raises(ValueError):
        unique_names(["same/mesh", "same/mesh"])                       # same path twice


def test_paired_slips_refuses_duplicate_names(tmp_path):
    """The guard at the other end: a caller that still passes two fits the same name gets an
    error, not a silent self-comparison."""
    fit = load_fit(write_fit(tmp_path / "fit"))
    truth = chains(6)
    with pytest.raises(ValueError):
        paired_slips([("mesh", fit), ("mesh", fit)], truth, tol_n=3.0, n_boot=10)


def test_by_winding_is_reported_next_to_the_chain_unit(tmp_path):
    """The README promises the by-winding CI as a secondary number. It comes from the same
    common test set, so the point estimates agree exactly and only the CI differs."""
    wide = range(1, 9)
    clean = load_fit(write_fit(tmp_path / "clean", windings=wide))
    bad = load_fit(write_fit(tmp_path / "bad", windings=wide, jump=(60, 90, 1)))
    truth = chains(10)
    res = paired_slips([("clean", clean), ("bad", bad)], truth, tol_n=3.0, n_boot=1000)
    assert res["unit"] == "human chain" and res["unit_key"] == "chain"
    assert set(res["by_unit"]) == {"chain", "winding"}
    assert res["by_unit"]["chain"]["differences"] == res["differences"]
    w = res["by_unit"]["winding"]
    assert w["groups"] == 3            # 4 clicks per chain -> 3 adjacent pairs, levels 0..2
    assert w["differences"]["bad-clean"]["point"] == \
        pytest.approx(res["differences"]["bad-clean"]["point"])
    assert w["differences"]["bad-clean"]["ci95"] != res["differences"]["bad-clean"]["ci95"]
    only = paired_slips([("clean", clean), ("bad", bad)], truth, tol_n=3.0, n_boot=200,
                        units=("winding",))
    assert only["unit"] == "human winding" and set(only["by_unit"]) == {"winding"}


def test_one_fit_at_a_time_gives_the_same_numbers(tmp_path):
    """run.py scores fits one at a time (four fits at once ran out of 14 GB). The split path
    must equal the FROZEN pre-split paired_slips (ec7c32a), draw for draw, on chains that
    give a spread-out bootstrap (only some chains cross the planted jump)."""
    import frozen_paired_ec7c32a as frozen
    wide = range(1, 9)
    clean = load_fit(write_fit(tmp_path / "clean", windings=wide))
    bad = load_fit(write_fit(tmp_path / "bad", windings=wide, jump=(60, 90, 1)))
    narrow = load_fit(write_fit(tmp_path / "narrow", windings=range(2, 5)))
    truth = mixed_chains(16)
    fits = [("clean", clean), ("bad", bad), ("narrow", narrow)]
    for parity, n_boot, seed in (("odd", 10000, 0), ("all", 500, 3)):
        ids = chain_parity_ids(truth, parity)
        old = frozen.paired_slips(fits, truth, chain_ids=ids, tol_n=3.0, n_boot=n_boot,
                                  seed=seed)
        tests = [fit_tests(n, f, truth, chain_ids=ids, tol_n=3.0, n_boot=n_boot, seed=seed)
                 for n, f in fits]
        new = paired_from_tests(tests, truth, chain_ids=ids, tol_n=3.0, n_boot=n_boot,
                                seed=seed)
        assert new == old
        # the data must not be degenerate, or equality proves only the plumbing
        lo, hi = new["differences"]["bad-clean"]["ci95"]
        assert hi - lo > 0


def mixed_chains(n_chains):
    """Chains that start at different angles: some cross the jump sector, some do not, so
    the chain bootstrap of bad - clean has a CI of non-zero width."""
    xyz, lev, ch = [], [], []
    for c in range(n_chains):
        n = 4
        k = np.arange(n)
        t0 = (0.05 + 0.137 * c) % 0.9
        f = 2.0 + k + t0 + 0.05 * k
        xyz.append(spiral_xyz(f, np.full(n, 105.0 + 2.0 * c)))
        lev.append(k.astype(float))
        ch.append([c] * n)
    return build_truth(from_arrays(np.concatenate(xyz), np.concatenate(lev),
                                  np.concatenate(ch)))
