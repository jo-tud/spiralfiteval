# spiralfiteval

Score a spiral fit of a Herculaneum scroll from its output meshes alone: how often it puts the
sheet on the wrong wrap, whether two windings sit on one sheet, and how close it stays to the
organisers' eval fibers. CPU only, no checkpoint, no GPU, works on fits from any fitter.

I have scored 20 distinct published fits on three scrolls with it (21 fit folders: Gaona's
spliced PHerc. 0211 meshes are byte-identical to his plain ones), and it reproduces Jashann's
published slips-per-wrap numbers. The metrics are not new; credit is under
[Prior work](#prior-work). What the tool adds is one checked scorer that reads both common fit
formats and puts the fits side by side, plus a [leaderboard](leaderboard/README.md) you can add
your own fit to with one command.

A fit is a folder of per-winding tifxyz meshes (`wNNN/`, `wNNN_spliced/`, as villa's
`fit_spiral.py` writes them under `meshes/mesh/`) or an `.npz` with one `wNNN` key per winding.

## Quickstart

Needs Python 3.12 or newer with `venv` (tested on 3.12, Debian 13, x86_64), `git` and `curl`.
CPU only, no CT read, about 310 MB of downloads. The tested versions are pinned in
`tools/spiralfiteval/constraints.txt`.

    git clone https://github.com/jo-tud/spiralfiteval && cd spiralfiteval
    python3 -m venv .venv && . .venv/bin/activate
    pip install -e tools/windingcheck -e tools/windbench -e "tools/spiralfiteval[plot]" \
      -c tools/spiralfiteval/constraints.txt

    spiralfiteval fetch                                  # Paris 4 winding files + eval fibers, ~290 MB
    curl -sfLO https://github.com/Jashann/vesuvius-scrolling/releases/download/v1.1/grids_Paris4_A2.npz
    spiralfiteval score grids_Paris4_A2.npz --references PHercParis4 --z 8400 9400 --out scores.json

`spiralfiteval score FIT` without `--references` scores H5 alone: it needs no annotations and
works on any scroll. Downloads land in `data/spiral_datasets/` inside the clone
(`WINDINGCHECK_DATA` moves them). Tests: `pip install -e "tools/spiralfiteval[tests]" -c
tools/spiralfiteval/constraints.txt && pytest tools/spiralfiteval` (28 tests, all on synthetic
fits, about 15 s).

## What you should see

The last command scores Jashann's Paris 4 run A2 (his fit with no human annotations, release
`vesuvius-scrolling` v1.1, 20 MB) over the band z 8400–9400: windings w010–w129, grid step
4.7 voxels, refined to 66 M vertices.

| Output | Value |
|---|---|
| H1 slips per wrap | **23.5 %** (95 % CI 19.5–29.3) on 353 adjacent human pairs in 39 chains. Jashann's own figure for this run is 20.3 % (17.0–24.9) |
| H1 exact, all pairs | 18.3 % (the slips compound along a chain) |
| H3 distance to the eval fibers | median 3.19 vx, p90 7.09, within 4 vx 57.5 %, not on the fit 6.3 % |
| H5 sheet spacing | median gap 11.2 vx, **duplicate share 3.2 %**, no neighbour 0.0 % |

Cost, measured on 2026-10-14 in a fresh `python:3.12-slim` container with nothing from the host
(cold pip cache, no cached downloads, 4 CPUs, 14 GB limit):

| Step | Wall clock | Peak RSS |
|---|---|---|
| `pip install` from the constraints | 37 s | 0.1 GB |
| `spiralfiteval fetch` (645 eval fibers, ~290 MB) | 226 s (290 MB, 1.3 MB/s) | 0.04 GB |
| `spiralfiteval score` of one Paris 4 fit over a 1,000-slice band | 133 s | 5.8 GB (6.2 GB now, see below) |
| all 21 published fit folders (`experiments/2026-10-13-spiralfit-eval/run.py`) | 2,682 s | 6.3 GB |

Downloads dominate; the scoring itself runs on one core. Read the seconds as an upper bound: the
machine was shared with other jobs, and the same steps took 24 / 86 / 75 / 1,725 s the day before
on a quieter machine with a warm cache. Since this table was measured, `score` also reports H5
per winding, and a second fresh run of the quickstart measured **6.2 GB** for `score` (111 s).
Plan on about 6.5 GB.

## What it measures

Every fit vertex gets a spiral time f = w + t: w from the winding name, t its column position
inside the winding, so f is continuous across the seams and one wrap is Δf = 1. A point's
*fit sheet* is the nearby fit vertex whose tangent plane passes closest along the normal
(within 10 voxels). Grids coarser than 5 voxels are refined bilinearly first.

| | Metric | Needs |
|---|---|---|
| H1 | **Winding identity**: slips per wrap = share of adjacent human click pairs (one wrap apart) that the fit puts on the wrong wrap difference; exact rate on all pairs; chain-bootstrap 95% CI | human relative-winding chains (Paris 4) |
| H2 | **Sheet switches** per 1,000 voxels along same-sheet traces: jumps of more than half a wrap that are not undone within 10 voxels | eval fibers (Paris 4) or any same-sheet traces |
| H3 | **Distance to reference**: normal distance from reference sheet points to the fit (median, p90, share within 2 and 4 voxels) | eval fibers or patches |
| H4 | **Conformity away from annotations**: H2 and H3 split by distance to the nearest human annotation | annotations |
| H5 | **Sheet spacing / duplicates**: normal gap to the fit's next winding; share under 2 voxels = two windings on one sheet | nothing; any scroll |

The labels H1–H5 are the keys in the JSON output. H1 is a held-out score only if the fit did not
take `relative_windings.json` as input. H5 catches duplicated windings, which the organisers'
ink-pixel objective rewards rather than penalises (their own `spiral-fitting/autoresearch.md`,
villa PR #1842).

**Do not rank unrefined fits by H2.** On 11 Paris 4 runs of `fit_spiral.py` it spans only
9.6–10.5 switches per 1,000 vx and does not follow H1 (Spearman −0.68, −0.58 without the fit
trained on the chains). Those fits sit a few voxels off the sheet between their inputs, about
half a sheet spacing from many fiber points, so the nearest-sheet assignment flips whatever the
winding. Use H1 for winding identity and H5 where there are no annotations, and read every
reference metric against its floor (`--floors`, below). Scores and figures for the 20 distinct
fits: [`experiments/2026-10-13-spiralfit-eval/`](../../experiments/2026-10-13-spiralfit-eval/README.md).

## Floors, ladder sets and paired comparisons

    spiralfiteval score FIT --references PHercParis4 --z 8400 9400 --floors 5 \
        --ladders abs_winding.json relative_windings.json

- `--floors N` scores each reference metric (H1, H2, H3, H4 within-4) a second way, with the
  references moved off their sheet:
  - a **radial shift curve**: all references moved by −8 … +8 vx;
  - a **phase-jitter floor**: each click, trace or point moved by a random radial offset within ±
    half the fit's median sheet gap, N seeds;
  - for H3, also **uniform random points** in the references' bounding box.

  A fit that sits on the reference sheets scores best at shift 0 and clearly better than the
  floor. If the curve is flat and the value sits on the floor, the metric says nothing about that
  fit.
- `--ladders FILE …` scores H1 on a ladder set with the selection of Jashann's
  `exp/e31b_bootstrap.py`: one slice per ladder, mean z at least 20 slices inside the band. On
  Paris 4 band 8400–9400 this gives his 32 human ladders. It reports:
  - slips per wrap under his counting rule and under this tool's;
  - **fit windings per labelled wrap** (Σ|Δf| / Σ|Δw|): 1.0 means one fit winding per sheet,
    above 1 means surplus windings squeezed between the sheets.

  `--ladder-nmax 3000` matches his subsample of a PCU gold file.
- H5 is also reported **per winding**: duplicate share and median gap per winding, the mean over
  windings next to the area-weighted share (outer windings are large and dominate the latter),
  and the share of duplicates per winding block.
- `--chains odd|even` scores H1 on half of the human chains only, so a fit or a model tuned on one
  half can be scored on the other.
- `spiralfiteval compare FIT_A FIT_B …` gives the paired H1 difference between fits (second minus
  first) on the pairs both cover, with a bootstrap CI over chains.

From Python:

```python
from spiralfiteval.fit import load_fit
from spiralfiteval.score import winding_identity, trace_switches, sheet_spacing
fit = load_fit("run/meshes/mesh")            # or an .npz of grids
fit.project(points_xyz, tol_n=10.0)          # -> (spiral time, normal distance) per point
```

The switch and distance conventions follow `tools/windbench` (≤ 1 voxel trace steps), and H1
reuses `tools/windingcheck`'s human chain reader and pair metrics.

## Results

- **Leaderboard.** 21 rows (20 fits, two authors) scored with one command each:
  [`leaderboard/TABLE.md`](leaderboard/TABLE.md). The best held-out Paris 4 fit is Jashann's
  B3w10_seed2 at 9.8 % slips per wrap; fit C, which was given the human windings, scores 2.5 %.
  H1 beats its floor on all 13 Paris 4 rows under the leaderboard rule (a metric is at floor
  when it is less than two floor SDs better), but on only 10 of 13 under the stricter rule I had
  set before the runs, and the H1 shift curve is lowest at shift 0 for only 2 of 13 fits (A2,
  B3w10_seed2). H2 is at floor on 10 of 13 and H3 on 9 of 13 (leaderboard rule).
- **H1 reproduces Jashann's numbers.** On his own held-out ladders, with his counting rule, this
  tool lands within 1.5 points of his published slips per wrap on all 13 Paris 4 runs, and within
  +0.2 to +0.8 points on all 6 PHerc. 0826 runs. On the human chains it lands inside his 95 % CI
  on 9 of 9 runs.
- **Every scroll has duplicated windings.** All six PHerc. 0826 runs and Gaona's PHerc. 0211 fit
  have 6.5–8.6 % of their surface on a duplicated winding, 2–8× the Paris 4 runs. On Paris 4
  (z 8400–9400) 91–99 % of the duplicates sit in the outermost windings (w110–129); on PHerc. 0826
  they sit throughout, worst in the core.
- **One fitter setting, scored on the odd human chains** ([`leaderboard/fit-spacing/`](leaderboard/fit-spacing/README.md)).
  With patch inputs, turning `fit_spiral.py`'s dense-spacing loss on (`dense_spacing_mode =
  "grad_mag"`) cut slips per wrap from 20.6 % to 7.7 % on the same 155 pairs (+12.9 points for
  spacing off, 95 % CI +7.1 to +19.9; the CI is over chains only). Read it with four limits: one
  band (z 8400–9400); 21 odd chains that I had seen scores on when I chose this setting, so they
  are not fully held out; one fit (one seed) per setting; not re-tested on a fresh band. With
  Jashann's track inputs the same change showed no effect (−2.7 points, CI −11.4 to +3.7), so
  that part of the test failed and the finding holds for patch inputs only.

## Prior work

I did not invent these metrics. Credit, with dates of first public release:

- **H1 winding identity** is **Jashann's "slips per wrap"** (`vesuvius-scrolling`, release v1.1,
  2026-09-29, <https://github.com/Jashann/vesuvius-scrolling>): same definition (closest fit
  surface, 10 voxels), his fits, his bootstrap CIs. My re-implementation reproduces his 9
  published values within −0.8 to +3.2 points (6 of 9 higher, 3 lower). That checks this tool;
  the metric and the fits are his. Scoring human relative windings from meshes was also done
  before in **Nicodol/spiralcheck** (`annotations.py`, 2026-08-04,
  <https://github.com/Nicodol/spiralcheck>) and talanham19-png/scrypt-vesuvius-winding-audit
  (2026-10-05). Added here: the exact rate on all pairs (long-range accuracy), e.g. Paris 4 A2
  18 % against C 97 %.
- **H5 duplicated windings** was measured before me by:
  - **claudepro1515/first-letters-fit-audit** `tools/sheet_hits.py` (2026-09-27,
    <https://github.com/claudepro1515/first-letters-fit-audit>): gaps between fit crossings along
    radial rays; 7.6% of gaps under 3 voxels on Gaona's PHerc. 0211 fit (w062–077, as fitted). I
    have not re-measured it under their rule; H5 measures a different gap (normal gap to the next
    winding's tangent plane, under 2 voxels);
  - **Nicodol/spiralcheck** (2026-07-26): annotation-free check for collapsed gaps / fused sheets;
  - **rodriguescarson** (2026-09-16, <https://github.com/rodriguescarson/eligible-scroll-atlas>):
    per-pair spacing profile of 0826 fits;
  - **TAUIL/spiral-winding-audit** (2026-10-13, <https://github.com/TAUIL-Abd-Elilah/spiral-winding-audit>):
    gap to the next winding's tangent plane, the closest method to mine, public before my first
    commit.

  villa PR #1842 (2026-09-23) documents that the fitting objective does not catch duplicated
  coverage; villa PR #2048 (open) that the default of 130 windings exports surplus windings
  outside the papyrus, which is where almost all Paris 4 duplicates sit (w110–129, z 8400–9400). The same
  default makes the scored PHerc. 0826 exports carry 120 windings (w010–w129) where #2048 puts
  the scroll at about 90, so part of the 6.9–8.6 % H5 share on Jashann's 0826 runs may come
  from it. Added here: H5 on Jashann's 19 published runs.
- **H2–H4 eval-fiber metrics** use the organisers' eval fibers as published. Jashann's report
  already notes that unrefined fits "sit 2 to 8 voxels off the sheet". That explains why H2 and H3
  sit close to their random-point floor on these fits (fit C: eval fibers 2.93 voxels median,
  uniformly random points 3.38; `results_ladders_floors.json`, `paris4.C.floors.H3.random`). Added here: the finding that H2 on the eval fibers does not rank
  unrefined fits (above).
- **Formats**: villa `fit_spiral.py` tifxyz output (`wNNN/`), Jashann's `.npz` grids.

## What does not work yet

- **H2 undercounts real crossings.** Scored on fit C along the human chains, with H2's own
  settings, it counts 322 switches for 443 human wraps crossed (0.73; I wanted it within ±20 %).
  The 10-voxel bridge limit loses about a quarter of the real crossings. Do not use H2 to count
  crossings.
- **The eval-fiber metrics say little about unrefined fits.** With the experiment's rule (H2 at
  least 20 % below its floor, H3 at least 0.5 vx below), H2 and H3 beat their floor on none of the
  13 Paris 4 fits, and the H4 difference between fits C and A2 on none of its 6 distance bins. The
  leaderboard's two-SD rule is looser and still flags most of them. Whether the eval fibers work
  as a metric on a refined fit that sits on the sheet is open.
- **Most of my predictions were wrong.** I wrote down 10 predictions before the runs and kept
  them as written; 7 were wrong (listed in the experiment README). They were written in my
  private working repository; this public repository starts after the runs, so the date cannot
  be checked here.
- **Whether duplicated windings cause local winding slips could not be measured** on Paris 4:
  only 7 human click pairs touch a duplicate (from a run of `src/spiralfiteval/local.py` whose
  output is not in this repository), and I had set 50 as the minimum. The human clicks
  cover the inner windings, the duplicates sit in the outer ones (w110–129).
- **H1–H4 run on one scroll.** Paris 4 is the only scroll with public human winding chains, eval
  fibers and published fits. H5 runs on any scroll.

## Licence, data and credits

Code: MIT (`LICENSE`). Data: CC BY-NC 4.0 and EduceLab-Scrolls terms, fetched at run time, never
shipped; read `DATA_NOTICE.md` for the required citation (EduceLab-Scrolls, Parsons et al. 2023,
and the Vesuvius Challenge CT scans) and the PHerc. Paris 4 publication reservation. Outputs
contain coordinates and scores only, no CT and no text. Fits: as published by their authors
(Jashann; Armando Gaona). Human winding annotations and eval fibers: the Vesuvius Challenge team.
