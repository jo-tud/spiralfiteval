# Scoring 20 published spiral fits

**Question.** Given only a spiral fit's output meshes, how often does it put the sheet on the
wrong wrap, does it stack two windings on one sheet, and does it stay close to the organisers'
eval fibers? The fitter is tuned on rendered ink pixels (`spiral-fitting/autoresearch.md`),
which cannot see a duplicated winding and says nothing about winding identity.

The metrics (H1–H5) and the prior work they come from are described in
[`tools/spiralfiteval/README.md`](../../tools/spiralfiteval/README.md). I fixed the metrics and
the thresholds before scoring any fit, and wrote down predictions so I could check them
afterwards.

## Data (all public)

- **Fits.** Jashann's release `vesuvius-scrolling` v1.1 (2026-09-29): 13 PHerc. Paris 4 runs of
  `fit_spiral.py` (band z 8400–9400 for 11 runs, z 11000–12000 for 2) and 6 PHerc. 0826 runs
  (z 8500–9500). Inputs per run, from his report: A2 = no human annotations; B-runs add his
  generated relative windings; A and B also verified patches; **C = verified patches plus the
  organisers' human winding annotations**, so C's H1 is not a held-out score. Armando Gaona's
  PHerc. 0211 fit (w010–w129, plain and spliced, z 8000–9000).
- **Human winding chains.** `relative_windings.json` from the organisers' spiral-input dataset:
  40 chains, 500 clicks inside band 1.
- **Eval fibers.** 645 fibers traced in VC3D and published by the organisers as an evaluation
  set (2.4 µm coordinates, divided by 4 here). None of the fits lists them as an input.

## Run

With the quickstart's venv active:

    python experiments/2026-10-13-spiralfit-eval/run.py                  # all fits, fetches what it lacks
    python experiments/2026-10-13-spiralfit-eval/run_ladders_floors.py   # Jashann's ladders, floors, H5 per winding
    python experiments/2026-10-13-spiralfit-eval/plots.py                # figures from results.json

`run_ladders_floors.py` also downloads Jashann's own scorer at a pinned commit and runs it on his
grids; I ran that step in villa's `vesuvius` environment. Numbers: `results.json` and `results_ladders_floors.json`. Figures: `fig_*.png`. One CPU job,
about 7 GB peak; the whole first run took 15 minutes for 21 fit folders.

## Result

PHerc. Paris 4, band 1 (z 8400–9400) unless marked:

| run | inputs | H1 slips/wrap (95% CI) | Jashann | H3 median vx | H2 switches /1000 vx | H5 duplicates |
|---|---|---|---|---|---|---|
| C | patches + human windings | **2.5%** (0.6–5.1) | 3.0% | 2.93 | 10.5 | 1.1% |
| B3w10_seed2 | auto + PCU | 9.8% (6.8–13.5) | 8.8% | 3.09 | 10.1 | 2.2% |
| B3w10 | auto + PCU | 10.9% (8.0–14.8) | 9.3% | 3.11 | 10.4 | 2.3% |
| B4w20 | auto + PCU | 12.7% (8.4–17.2) | – | 3.07 | 10.3 | 2.2% |
| B | patches + PCU | 14.8% (7.9–21.7) | 15.6% | 3.02 | 10.0 | 1.1% |
| B2 | auto + PCU | 16.0% (13.0–19.6) | – | 3.27 | 10.4 | 2.8% |
| A | patches | 17.3% (12.5–22.7) | 15.9% | 3.14 | 10.3 | 1.4% |
| B3w40 | auto + PCU | 19.6% (13.5–26.6) | – | 3.05 | 10.0 | 2.2% |
| B4w4 | auto + PCU | 20.1% (16.4–24.7) | – | 3.18 | 10.2 | 2.7% |
| A2 | auto (no annotations) | 23.5% (19.5–29.3) | 20.3% | 3.19 | 9.6 | 3.2% |
| A2_seed2 | auto | 24.6% (20.3–31.3) | 22.1% | 3.16 | 9.6 | 3.2% |
| A2_z11000 (band 2) | auto | 31.3% (25.4–37.7) | 30.6% | 3.18 | 10.1 | 1.1% |
| B3w10_z11000 (band 2) | auto + PCU | 28.0% (20.5–36.2) | 28.6% | 3.19 | 10.9 | 0.4% |

Other scrolls (H5 only; no human windings or eval fibers are published for them):

| fit | median gap between windings | H5 duplicates |
|---|---|---|
| PHerc. 0826, 6 runs (Jashann) | 8.1–9.0 vx | 6.9–8.6% |
| PHerc. 0211, Gaona (plain and spliced) | 7.1 vx (66 µm) | 6.5% |

Gaona's published `wNNN_spliced` meshes are byte-identical to the plain `wNNN` ones, so his two
rows agree exactly.

What this means:

- **H1 works, and it is Jashann's metric.** On the human chains this tool lands inside his
  published 95 % CI on all 9 runs he reports (−0.8 to +3.2 points from his value). On his own
  held-out ladders, with his counting rule, it lands within 1.5 points on all 13 Paris 4 runs and
  within +0.2 to +0.8 on all 6 PHerc. 0826 runs; his scorer re-run on the release grids gives his
  published numbers on 19 of 19. H1 separates the human-annotated fit (2.5 %) from every fit
  without human windings (≥ 9.8 %). It needs human winding clicks, which exist only for Paris 4.
- **H5 finds duplicated windings on every scroll**, as claudepro1515, Nicodol, rodriguescarson
  and TAUIL found before. PHerc. 0826 and 0211 have 2–8× the Paris 4 share and smaller gaps
  between windings. On Paris 4 91–99 % of the duplicates sit in w110–129, the surplus outer
  windings of `fit_spiral.py`'s default of 130 (villa PR #2048). On PHerc. 0826 they sit
  throughout, worst in the core (w010–012 about 80 % duplicated), and the fits have 1.30–1.53
  fit windings per labelled wrap: consistent with that default, but I did not run the causal
  test (a fit with `num_windings: 90`).
- **The eval-fiber metrics (H2, H3, H4) do not measure these fits.** H2 and H3 beat their floor
  on none of the 13 Paris 4 fits: moving the fibers ±4 vx changes the H3 median by −0.10 to
  +0.04 vx, and jittered fibers score 9.8–11.1 switches against 9.6–10.9 for the real ones. The
  fits drift a few voxels off the sheet between their inputs, as Jashann reports. Whether the
  eval fibers work on a refined fit is open.

## What I got wrong

- **H2's own check failed.** Counted along the human chains on fit C, H2 finds 322 switches for
  443 human wraps crossed (0.73; I wanted ±20 %). Its 10-voxel bridge limit loses about a quarter
  of the real crossings. I first reported this check as passed because the first run used a
  different counter setting; that was wrong.
- **Seven of my ten predictions were wrong**: that H2 would rank fit C best (it ranked it worst);
  that H2 and H1 would agree (Spearman −0.68); that the human windings' benefit would fade with
  distance from the clicks; that every fit would have under 1 % duplicates (1 of 21 did); that H1
  would beat its floor on all 13 Paris 4 fits (10 of 13 did); that Paris 4 fits would have
  0.9–1.1 fit windings per labelled wrap (A2, A2_seed2 and B2 had 1.10–1.15); and that the eval
  fibers sit 4–6 vx off the clicked sheet near the clicks (the best shift was +1 vx, on only 54
  points). Right: H2 and H3 at floor on all 13 fits, where the duplicates sit, and the per-winding
  mean against the area-weighted share.
- **A bug in H5's gap search**, fixed on 2026-10-14: the search dropped too many neighbours, so
  median gaps were 0.9–1.9 vx too small and many vertices showed "no neighbour". Duplicate shares
  did not change. All 21 fits were re-run (`rerun_h5.py`); the old values are in
  `results_before_gap_fix.json`.

## Limits

- H1–H4 run on one scroll: Paris 4 is the only scroll with public human winding chains, eval
  fibers and published fits. H5 runs on three.
- Fit C was given the human windings H1 scores it on; all other runs are held out.
- The organisers' own whole-scroll fit is published only as flattened chunks and is not scored.
- Scoring needs no new human time: H5 needs no annotations, H1–H4 reuse the organisers' published
  clicks and fibers. I have no measured baseline for checking a fit by eye, so I claim no time
  saved.
