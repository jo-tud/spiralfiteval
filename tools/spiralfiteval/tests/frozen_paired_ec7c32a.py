# FROZEN copy of spiralfiteval/paired.py at ec7c32a (the reviewed version).
# Test fixture only: test_paired.py checks the one-fit-at-a-time path against it, so the
# equivalence test can fail. Do not edit; the only changes are the absolute import and one
# docstring line reworded (no code change).
"""Paired comparison of two or more fits on the same human chains (H1 slips per wrap).

A single fit's H1 CI (``score.winding_identity``) is too wide to settle a difference between
two fits: the two scores are strongly correlated, because they are the same clicks on the same
chains. This module resamples chains once and recomputes **both** fits on the resampled chains,
so the CI is on the difference.

Two rules, fixed here rather than per caller:

* **Common tests.** Only click pairs that every compared fit covers (both clicks within
  ``tol_n`` of a fit sheet) enter the paired difference. Coverage differences are a separate
  effect and are reported per fit, not folded into the difference.
* **Chains are the resampling unit.** Clicks inside a chain are not independent. This is the
  unit ``score.winding_identity`` and Jashann's published CIs use.
"""
from __future__ import annotations

import numpy as np

from spiralfiteval.score import winding_identity


def chain_parity_ids(truth, parity: str, ids=None) -> list | None:
    """Chain ids of one parity, in windingcheck's held-out convention.

    ``parity`` is ``"all"``, ``"odd"`` or ``"even"``; odd means chain index % 2 == 1, the
    chains windingcheck's shipped models never saw (``windingcheck --heldout``). The index is
    the chain's position in ``relative_windings.json``, which ``load_pcl`` keeps as the chain
    label and ``build_truth`` does not renumber. ``ids`` restricts the pool first (the band).
    """
    if parity not in ("all", "odd", "even"):
        raise ValueError(f"parity must be all, odd or even, not {parity!r}")
    pool = np.unique(truth.chains.chain) if ids is None else np.asarray(list(ids))
    if parity == "all":
        return None if ids is None else [int(c) for c in pool]
    want = 1 if parity == "odd" else 0
    return [int(c) for c in pool if int(c) % 2 == want]


def _per_chain(truth, tests, chain_ids):
    """Adjacent-pair (a, b) keys, their error flags and their chain, for one fit."""
    a, b = tests["a"], tests["b"]
    adj = (truth.chains.level[b] - truth.chains.level[a]) == 1
    a, b = a[adj], b[adj]
    err = np.rint(tests["f"][b] - tests["f"][a]) != 1
    return {(int(x), int(y)): bool(e) for x, y, e in zip(a, b, err)}


def paired_slips(fits, truth, *, chain_ids=None, tol_n: float = 10.0,
                 n_boot: int = 10000, seed: int = 0, units=("chain", "winding")) -> dict:
    """H1 slips per wrap for every fit, and the paired CI of every pairwise difference.

    ``fits``: list of (name, Fit); the names key the result, so they must be distinct. Returns
    per-fit standalone numbers (own coverage), the common-test numbers, and for each ordered
    pair (i, j) the difference j - i with a bootstrap 95 % CI and the share of draws on each
    side of 0. ``units[0]`` is the resampling unit of ``differences`` (the bar's unit, the human
    **chain**); every unit in ``units`` is also reported under ``by_unit`` - ``"winding"``
    resamples human winding levels, the unit the original bar text named, as a secondary CI.
    """
    names = [n for n, _ in fits]
    if len(set(names)) != len(names):
        raise ValueError(f"fit names must be distinct, got {names}; see fit.unique_names")
    if not units:
        raise ValueError("units must name at least one resampling unit")
    own, cover = {}, {}
    for name, fit in fits:
        h1, tests = winding_identity(fit, truth, tol_n=tol_n, chain_ids=chain_ids,
                                     n_boot=n_boot, seed=seed, return_tests=True)
        own[name] = {"slips_per_wrap": h1["slips_per_wrap"],
                     "slips_ci95": h1.get("slips_ci95"),
                     "adjacent_tests": h1["adjacent_tests"],
                     "clicks_on_fit": h1["clicks_on_fit"]}
        cover[name] = _per_chain(truth, tests, chain_ids)

    common = sorted(set.intersection(*(set(c) for c in cover.values())))
    group_of = {"chain": lambda a: int(truth.chains.chain[a]),
                "winding": lambda a: int(truth.chains.level[a])}
    by_unit, n_groups = {}, {}
    for unit in units:
        if unit not in group_of:
            raise ValueError(f"unit must be one of {sorted(group_of)}, not {unit!r}")
        g = np.array([group_of[unit](a) for a, _ in common])
        uniq, inv = np.unique(g, return_inverse=True)
        n_c = np.bincount(inv, minlength=len(uniq)).astype(float)
        err_c = {name: np.bincount(inv, weights=[float(cover[name][k]) for k in common],
                                   minlength=len(uniq))
                 for name in names}
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, len(uniq), size=(n_boot, len(uniq))) if len(uniq) else None
        rates = {name: (err_c[name][draw].sum(1) / np.maximum(n_c[draw].sum(1), 1)
                        if draw is not None else np.zeros(0))
                 for name in names}
        rate = {name: float(err_c[name].sum() / max(n_c.sum(), 1)) for name in names}
        diffs = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = rates[b] - rates[a]
                lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                diffs[f"{b}-{a}"] = {
                    "point": float(rate[b] - rate[a]), "ci95": [lo, hi],
                    "share_below_zero": float((d < 0).mean()),
                    "excludes_zero": bool(lo > 0 or hi < 0)}
        by_unit[unit] = {"unit": unit, "groups": int(len(uniq)),
                         "common": {name: {"slips_per_wrap": rate[name]} for name in names},
                         "differences": diffs}
        n_groups[unit] = int(len(uniq))

    primary = by_unit[units[0]]
    out = {
        "unit": {"chain": "human chain", "winding": "human winding"}[units[0]],
        "unit_key": units[0],
        "n_boot": n_boot, "seed": seed, "tol_n": tol_n,
        "chains": n_groups.get("chain", int(len(np.unique(
            [int(truth.chains.chain[a]) for a, _ in common])) if common else 0)),
        "common_adjacent_tests": int(len(common)),
        "own": own,
        "common": primary["common"],
        "differences": primary["differences"],
        "by_unit": by_unit,
    }
    return out


def resolvable_difference(fits, truth, *, chain_ids=None, tol_n: float = 10.0,
                          n_boot: int = 10000, seed: int = 0) -> dict:
    """Half-width of the paired CI over the pairs of ``fits``: what this chain set can resolve.

    Run on fits whose difference is already known (published runs) this is a power calibration
    for a chain set, computable before the fits under test exist.
    """
    res = paired_slips(fits, truth, chain_ids=chain_ids, tol_n=tol_n, n_boot=n_boot, seed=seed)
    half = {k: (v["ci95"][1] - v["ci95"][0]) / 2 for k, v in res["differences"].items()}
    return {"chains": res["chains"], "common_adjacent_tests": res["common_adjacent_tests"],
            "half_width": half, "differences": res["differences"], "own": res["own"]}
