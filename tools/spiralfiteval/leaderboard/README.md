# spiralfiteval leaderboard

Every published spiral fit I could download, scored by the same tool with the same command, plus
one command to score your own fit and add a row.

- **[TABLE.md](TABLE.md)**: the table, with the exact command and the input hash of every row.
- **[leaderboard.csv](leaderboard.csv)**: the same numbers, one row per scoring configuration, for plotting.
- **[fits.json](fits.json)**: the registry: author, release, download URL and score arguments per fit.
- **[rows/](rows/)**: one JSON per row: the full score, the command, the hashes and the cost.

This is not a ranking of people or of fitters. The fits were made for different purposes with
different inputs, and most of them are unrefined runs of the organisers' `fit_spiral.py`.
H2 is at floor on 10 of 13 reference rows and H3 on 9 of 13; H1 beats its floor on all 13.
Read the floor column before reading a number. The metrics and annotations come from others;
credit and dates are in [../README.md § Prior work](../README.md#prior-work).

I re-scored all 21 rows from a fresh clone in a stock `python:3.12-slim` container: every
headline number came out identical, and the largest peak memory was 6.62 GB
(`jashann-pherc0826-A0`).

## Add your fit (one command)

From a fresh clone of this repository, with the quickstart's venv active
([../README.md](../README.md#quickstart)):

    tools/spiralfiteval/leaderboard/add_row.sh --fit path/to/your/fit \
        --id yourname-scroll-run --author "Your Name" \
        -- --references PHercParis4 --z 8400 9400 --floors 5

`--fit` is a folder of per-winding tifxyz meshes (`wNNN/`, as villa's `fit_spiral.py` writes them
under `meshes/mesh/`) or an `.npz` with one `wNNN` key per winding. Everything after `--` is
passed to `spiralfiteval score` unchanged: drop `--references`/`--z` and you get H5 alone, which
needs no annotations and works on any scroll. The command downloads the references if they are
missing (~290 MB), scores the fit, writes `rows/<id>.json` and rebuilds `TABLE.md` and
`leaderboard.csv`. To re-score a row that is already in [fits.json](fits.json), including
downloading the fit itself:

    tools/spiralfiteval/leaderboard/add_row.sh --id jashann-paris4-A2

To rebuild every published row from scratch: `tools/spiralfiteval/leaderboard/run_published.sh`.
To check the add-a-row path on a synthetic fit (about 10 s, no download):
`tools/spiralfiteval/leaderboard/selftest.sh`.

Send a pull request with your `rows/<id>.json`, your entry in `fits.json` and the regenerated
table. I only merge rows whose registry entry downloads the fit from a public URL. A row created
with `--fit` records a local path, so its command needs that file on the reader's machine;
`--fit-url` records the URL as metadata only. For a row anyone can reproduce, add the download
source and score arguments to `fits.json` and generate the row with `add_row.sh --id <id>`:
registry rows download their inputs when missing.

## What the columns mean

Full definitions: [../README.md § What it measures](../README.md#what-it-measures). In short:

| | Metric | Direction | Needs |
|---|---|---|---|
| H1 | slips per wrap: share of adjacent human click pairs put on the wrong wrap | lower is better | human winding chains (Paris 4) |
| H1 exact | share of *all* click pairs with the right wrap difference (long range) | higher is better | the same |
| H2 | sheet switches per 1,000 voxels along the organisers' eval fibers | lower is better | eval fibers (Paris 4) |
| H3 | median normal distance from eval-fiber points to the fit | lower is better | eval fibers (Paris 4) |
| H5 | duplicate share: fit vertices whose next winding is under 2 voxels away | lower is better | nothing; any scroll |

`held_out_H1 = false` marks a fit that was given the human winding annotations as an input: its H1
is a fit-quality number, not a held-out score, and should not be compared with the others.

## The floor column

H1, H1-on-ladders, H2 and H3 are scored a second way with the references moved off their sheet
(`--floors N`: a phase-jitter floor over N seeds, plus a radial shift curve; H3 also gets a
uniform-random-point floor). A metric that does not beat its own floor says nothing about that
fit; the table flags it but keeps the value and the row order. H1 exact and H4 have no floor.
I fixed this rule before scoring any row.

A metric is reported **at floor** when

    margin < 2 x floor_sd

where `margin` is the improvement over the phase-jitter floor mean (positive = better than the
floor) and `floor_sd` is the spread of the floor over the seeds. H3 has two floors (phase jitter
and uniform random points) and is reported at floor when either test fails.

A row scored without `--floors` reports `not measured`, never a blank.
H5 needs no reference, so it has no floor and is never flagged; its per-winding breakdown
(`H5_by_winding`) is in the row JSON.

## Ordering

Rows with H1 come first, ordered by H1 slips per wrap ascending; rows without references follow,
ordered by H5 duplicate share ascending. The order is for presentation only: a fit whose H1 is
not held out, or whose metrics are at floor, is flagged in its row.

## Provenance of a row

Each row records the commit of the scoring code, a `tool.dirty` flag and the SHA-256 of its fit
input (the file, or a sorted manifest of a mesh folder). `dirty` can be true because of
uncommitted row files even when the code is unchanged, so it is a hint, not proof. The
fresh-container re-run above is the independent check: it clones one commit and re-scores the
committed inputs.

The commit ids in the rows (`bfb8665` and others) are from my private working history, before
this public repository was created from it as one commit; they do not resolve here. The scoring
code changed after `bfb8665` (coverage fields, chain parity, names), but not in a way that moves
a score as far as I can tell: the quickstart re-scores Jashann's A2 in this repository to the
row's numbers exactly. I have not re-scored every row with the public code.

## Licence and data

Code MIT (`../LICENSE`). The fits belong to their authors and are linked, never copied into this
repository; the references are fetched at run time. Required citations and the PHerc. Paris 4
publication reservation: [../DATA_NOTICE.md](../DATA_NOTICE.md). Rows hold scores derived from
coordinates only, no CT and no text.
