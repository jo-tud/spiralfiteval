"""Figures for the spiral-fit scores, from results.json and arrays_A2_C.npz (run.py output)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools/windingcheck/src"))
from windingcheck.chains import load_pcl  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
BLUE, ORANGE, RED = "#2f6fb0", "#d9822b", "#c93c37"


def drift_maps(arr, res, path):
    """Top view of band 1: eval fibers coloured by distance to the fit, and H1 slips.

    H2 switch sites are not drawn: there are ~5,000 per fit, they cover the map, and H2 failed
    as a winding measure (README, Result)."""
    from run import scroll_dir  # noqa: F401  (same data root as run.py)
    clicks = load_pcl(scroll_dir() / "relative_windings.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharex=True, sharey=True)
    for ax, name in zip(axes, ("A2", "C")):
        p, dn = arr[f"{name}_fiber_pts"], arr[f"{name}_fiber_dn"]
        off = ~np.isfinite(dn)
        ax.scatter(p[off, 0], p[off, 1], s=0.2, c=GRID, rasterized=True)
        sc = ax.scatter(p[~off, 0], p[~off, 1], s=0.3, c=dn[~off], cmap="viridis", vmin=0, vmax=10,
                        rasterized=True)
        a, b = arr[f"{name}_h1_a"], arr[f"{name}_h1_b"]
        t, pr = arr[f"{name}_h1_true_dw"], arr[f"{name}_h1_pred_dw"]
        bad = (t == 1) & (pr != 1)
        good = (t == 1) & (pr == 1)
        xy = clicks.sorted().xyz
        for m, col, lab in ((good, MUTED, "adjacent human pair, right"), (bad, RED, "adjacent human pair, slip")):
            segs = np.stack([xy[a[m], :2], xy[b[m], :2]], axis=1)
            for k, s in enumerate(segs):
                ax.plot(s[:, 0], s[:, 1], c=col, lw=1.6 if col == RED else 0.8,
                        label=f"{lab} ({m.sum()})" if k == 0 else None)
        h1, h3 = res[name]["H1"], res[name]["H3"]
        ax.set_title(f"{name} ({res[name]['inputs']}): slips {h1['slips_per_wrap']:.1%}, "
                     f"{h3['within_4']:.0%} of fiber within 4 vx",
                     fontsize=10, color=INK, loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("x (9.6 µm voxels)")
        ax.legend(loc="lower left", fontsize=7, frameon=False, markerscale=1.5)
    axes[0].set_ylabel("y (9.6 µm voxels)")
    cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("eval fiber to fit sheet, voxels (grey: > 10, not on the fit)")
    fig.suptitle("PHerc. Paris 4, z 8400-9400: where the fit leaves the organisers' eval fibers",
                 x=0.01, ha="left", color=INK)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def reproduction(res, path):
    """This tool's slips per wrap against Jashann's published values."""
    names = [n for n, r in res.items() if r.get("jashann_ci")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(names))
    for i, n in enumerate(names):
        r = res[n]
        lo, hi = r["jashann_ci"]
        ax.plot([lo, hi], [i + 0.12] * 2, c=MUTED, lw=2)
        ax.plot(r["jashann_slips"], i + 0.12, "o", c=MUTED, ms=5, label="Jashann (published)" if i == 0 else None)
        ci = r["H1"].get("slips_ci95") or [np.nan, np.nan]
        ax.plot(ci, [i - 0.12] * 2, c=BLUE, lw=2)
        ax.plot(r["H1"]["slips_per_wrap"], i - 0.12, "o", c=BLUE, ms=5, label="spiralfiteval H1" if i == 0 else None)
    ax.set_yticks(y, [f"{n} ({res[n]['inputs']})" for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("slips per wrap (adjacent human pairs on the wrong wrap), 95% CI")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Same fits, independent re-implementation of the winding check", fontsize=10, loc="left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def conformity(res, path):
    """H4: share of eval-fiber points within 4 voxels of the fit, by distance to annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, col in (("A2", ORANGE), ("C", BLUE)):
        h4 = res[name]["H4"]
        lab = [f"{b['lo']:.0f}-{b['hi']:.0f}" if np.isfinite(b["hi"]) else f"≥{b['lo']:.0f}" for b in h4["within_4"]]
        x = np.arange(len(lab))
        axes[0].plot(x, [b["mean"] for b in h4["within_4"]], "o-", c=col, label=f"{name} ({res[name]['inputs']})")
        axes[1].plot(x, [b["per_1000_vox"] for b in h4["switches_per_1000_vox"]], "o-", c=col, label=name)
    for ax, t in zip(axes, ("eval-fiber points within 4 vx of the fit", "sheet switches per 1000 vx of fiber")):
        ax.set_xticks(x, lab, fontsize=8)
        ax.set_xlabel("distance to the nearest human winding annotation (voxels)")
        ax.set_title(t, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def duplicates(allres, path):
    """H5: share of fit vertices on a duplicated winding (gap < 2 vx), every fit on three scrolls."""
    rows = [(scroll, n, r["H5"]["duplicate"]) for scroll, fits in allres.items() if scroll != "criteria"
            for n, r in fits.items() if "H5" in r]
    fig, ax = plt.subplots(figsize=(7.5, 0.22 * len(rows) + 0.9))
    cols = {"paris4": BLUE, "pherc0826": ORANGE, "pherc0211": RED}
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows], color=[cols[r[0]] for r in rows], height=0.7)
    ax.set_yticks(y, [f"{s} {n}" for s, n, _ in rows], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.01, c=MUTED, lw=0.8, ls="--")
    ax.text(0.0105, -0.55, "predicted bound 1%", fontsize=7, color=MUTED)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of fit vertices < 2 vx from another winding (two windings on one sheet)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title("Duplicated windings, no annotations needed", fontsize=10, loc="left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    allres = json.loads((HERE / sys.argv[1] if len(sys.argv) > 1 else HERE / "results.json").read_text())
    res = allres["paris4"]
    duplicates(allres, HERE / "fig_duplicates.png")
    arr = np.load(HERE / "arrays_A2_C.npz")
    reproduction(res, HERE / "fig_reproduction.png")
    if "A2" in res and "C" in res:
        drift_maps(arr, res, HERE / "fig_drift_A2_C.png")
        conformity(res, HERE / "fig_conformity.png")
