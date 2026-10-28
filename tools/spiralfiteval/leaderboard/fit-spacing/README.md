# Fit-spacing rows: one `fit_spiral.py` setting, scored on held-out chains

Eleven PHerc. Paris 4 spiral fits (z 8400-9400, villa `fit_spiral.py` @ `47111d1`, 120
windings) that differ in one setting from a fit next to them, and the paired comparisons that
test that setting. Three ran 12k steps (v0.2; villa's DT losses start at step 25k, so they were
off). Eight ran villa's default 30k steps (v0.2.2), three seeds per setting on patch inputs.
These rows are **not** in the main table: they score H1 on the **21 odd human chains only**
(`--chains odd`), while every main-table row uses all 39 band chains. Do not rank them against
the main table. The odd half is held out because windingcheck's shipped model was fitted on the
even chains; none of these fits takes a human annotation as input. The odd half is not
untouched, though: I had looked at odd-chain scores when I chose this setting to test and set
the thresholds below. Each row is one fit, the bootstrap CI of each `compare` file is over
chains only, and the result has not been re-tested on a fresh band. Only the three-seed 30k
patch result below has seed noise in its CI.

The H2, H3 and H4 fields in these rows' JSON were scored before the 2026-10-23 eval-fiber frame fix
([../../README.md](../../README.md#results)) and are a quarter of an L2 voxel off. Nothing on this
page uses them; read H1 only.

| row | config | one change from |
|---|---|---|
| `fitspacing-Aon` | patches, no tracks, `dense_spacing_mode = "grad_mag"` | - |
| `fitspacing-Aoff` | as Aon, `dense_spacing_mode = "winding_model"` | Aon (spacing source off, as in the published runs) |
| `fitspacing-A2on` | Jashann's A2 config (tracks, no patches), `dense_spacing_mode = "grad_mag"` | Jashann's published A2 |
| `fitspacing-Aon30` | as Aon, 30k steps (villa's default `optimizer_random_seed = 1`) | Aon (length) |
| `fitspacing-Aoff30` | as Aoff, 30k steps, seed 1 | Aon30 |
| `fitspacing-Aon30-s2`, `-s3` | as Aon30, `optimizer_random_seed` 2, 3 | Aon30 (seed) |
| `fitspacing-Aoff30-s2`, `-s3` | as Aoff30, `optimizer_random_seed` 2, 3 | Aon30-s2, Aon30-s3 |
| `fitspacing-A2on30` | as A2on, 30k steps | A2on (length) |
| `fitspacing-A2off30` | as A2on30, `dense_spacing_mode = "winding_model"` | A2on30 |

Why the setting matters: in villa the `winding_model` spacing source exists only when winding
inference is on. The published runs switch inference off and keep that default, so their
dense-spacing loss has no source; `grad_mag` pulls neighbouring windings to the sheet spacing
read from the CT gradient magnitude.

## Result at 30k steps (v0.2.2)

| | H1 slips per wrap, own pairs (seeds 1, 2, 3) | pairs |
|---|---|---|
| Aon30 (spacing on) | 6.9 %, 6.9 %, 6.9 % | 231, 233, 233 |
| Aoff30 (spacing off) | 9.0 %, 10.0 %, 7.8 % | 233, 229, 231 |
| A2on30 (A2 + spacing on) | 25.5 % | 220 |
| A2off30 (A2, spacing off) | 26.2 % | 225 |

- **Patch inputs: no gain from spacing shown.** On the 228 pairs all six patch fits cover (20
  chains), spacing off has 8.8 % slips per wrap and spacing on 6.9 % (each the mean of three
  seeds), a difference of +1.9 points. Per seed, on the pairs both fits of that seed cover:
  +1.7 points (CI +0.4 to +4.0, `compare-Aon30-Aoff30.json`), +3.1 (CI +1.5 to +5.2,
  `compare-Aon30-s2-Aoff30-s2.json`) and +0.9 (CI −0.6 to +3.5,
  `compare-Aon30-s3-Aoff30-s3.json`). The 95 % CI of the three-seed mean, seed noise included,
  is −1.3 to +5.1 points: a chain bootstrap of the mean (half-width 1.7, 10,000 draws, seed 0,
  one draw for all six fits) and a seed term t(0.975, 2) · sd / √3 = 4.303 · 1.1 / √3 = 2.7,
  added in quadrature. I had asked for a lower end above 0 before these fits ran, so I could not
  show a gain. All three seeds put spacing off higher, and spacing off varies more from seed to
  seed (sd 1.2 points against 0.3 on the common pairs).
- **Track inputs: no effect shown.** A2off30 minus A2on30 is +2.8 points (CI −0.5 to +6.4) on
  211 common pairs, one fit per setting (`compare-A2on30-A2off30.json`).
- The 12k gap below closes from the spacing-off side: Aoff goes from 18.6 % at 12k to 9.0 % at
  30k on its own pairs, while Aon stays at 6–7 %.

## Result at 12k steps (v0.2)

| | H1 slips per wrap, own pairs | pairs |
|---|---|---|
| Aon (spacing on) | 6.1 % | 198 |
| Aoff (spacing off) | 18.6 % | 183 |
| A2on (A2 + spacing on) | 24.1 % | 166 |
| Jashann A2 (published) | 25.7 % | 171 |

- **Patch inputs: spacing off costs +12.9 points** (95 % CI +7.1 to +19.9) on the 155 pairs
  both fits cover (20.6 % vs 7.7 %), 20 chains, 10,000 bootstrap draws over chains, seed 0.
- Aoff reproduces Jashann's published A: on the 178 pairs both cover, 21 chains, the same slip
  count in every chain (difference 0.0, CI 0.0 to 0.0; `compare-pubA-Aoff.json`).
- **Track inputs: no effect shown.** A2on minus published A2 is −2.7 points (CI −11.4 to +3.7)
  on 147 common pairs. I had asked for at least −9.8 points with the CI below 0, so this test
  **failed**. The finding holds for patch inputs only, not for track inputs.

## Reproduce

Download the three 12k fits (release assets, ~20 MB each) and Jashann's A2 grids:

    mkdir -p data/spiralfits/fit-spacing data/spiralfits/jashann_v1.1
    for f in Aon Aoff A2on; do curl -sfL -o data/spiralfits/fit-spacing/fitspacing-$f.npz \
      https://github.com/jo-tud/spiralfiteval/releases/download/v0.2/fitspacing-$f.npz; done
    curl -sfL -o data/spiralfits/jashann_v1.1/grids_Paris4_A2.npz \
      https://github.com/Jashann/vesuvius-scrolling/releases/download/v1.1/grids_Paris4_A2.npz

Each row (writes `rows/fitspacing-<fit>.json` here, leaves the main table alone; ~7 GB, a few
minutes on CPU):

    LEADERBOARD_ROWS=tools/spiralfiteval/leaderboard/fit-spacing/rows \
      tools/spiralfiteval/leaderboard/add_row.sh \
      --fit data/spiralfits/fit-spacing/fitspacing-Aoff.npz --id fitspacing-Aoff --author jo-tud \
      --scroll-name "PHerc. Paris 4" --no-table \
      -- --references PHercParis4 --z 8400 9400 --floors 5 --chains odd

The paired differences (~12 GB for two fits; the Aoff-vs-published-A line also needs
Jashann's `grids_Paris4_A.npz` from the same release as A2):

    FS=tools/spiralfiteval/leaderboard/fit-spacing
    spiralfiteval compare data/spiralfits/fit-spacing/fitspacing-Aon.npz \
      data/spiralfits/fit-spacing/fitspacing-Aoff.npz --name Aon --name Aoff \
      --references PHercParis4 --z 8400 9400 --chains odd --out $FS/compare-Aon-Aoff.json
    spiralfiteval compare data/spiralfits/jashann_v1.1/grids_Paris4_A2.npz \
      data/spiralfits/fit-spacing/fitspacing-A2on.npz --name pubA2 --name A2on \
      --references PHercParis4 --z 8400 9400 --chains odd --out $FS/compare-pubA2-A2on.json
    spiralfiteval compare data/spiralfits/jashann_v1.1/grids_Paris4_A.npz \
      data/spiralfits/fit-spacing/fitspacing-Aoff.npz --name pubA --name Aoff \
      --references PHercParis4 --z 8400 9400 --chains odd --out $FS/compare-pubA-Aoff.json

`compare` reports the second fit minus the first. The three JSON files in this folder are the
outputs of exactly these commands (run as `python -m spiralfiteval.cli compare ...`, which is
the same entry point): `compare-Aon-Aoff.json` (`Aoff-Aon`), `compare-pubA2-A2on.json`
(`A2on-pubA2`) and `compare-pubA-Aoff.json` (`Aoff-pubA`). A clean re-run of the first two
reproduces the committed files with no difference.
SHA-256 of the three `.npz` files: Aon `833fa424...e293293`, Aoff `92e79b65...c53786`,
A2on `0d84936a...0567d` (full hashes in each row's `input_hash`).

The eight 30k fits are v0.2.2 release assets:

    for f in Aon30 Aoff30 Aon30-s2 Aoff30-s2 Aon30-s3 Aoff30-s3 A2on30 A2off30; do
      curl -sfL -o data/spiralfits/fit-spacing/fitspacing-$f.npz \
        https://github.com/jo-tud/spiralfiteval/releases/download/v0.2.2/fitspacing-$f.npz; done

Their rows and compares come from the same two commands as above, with `--id fitspacing-<fit>`
and `--name <fit>`; spacing on is always the first fit, so each file reports off minus on:
`compare-Aon30-Aoff30.json`, `compare-Aon30-s2-Aoff30-s2.json`,
`compare-Aon30-s3-Aoff30-s3.json` and `compare-A2on30-A2off30.json`. For example:

    FS=tools/spiralfiteval/leaderboard/fit-spacing
    LEADERBOARD_ROWS=$FS/rows \
      tools/spiralfiteval/leaderboard/add_row.sh \
      --fit data/spiralfits/fit-spacing/fitspacing-Aoff30.npz --id fitspacing-Aoff30 --author jo-tud \
      --scroll-name "PHerc. Paris 4" --no-table \
      -- --references PHercParis4 --z 8400 9400 --floors 5 --chains odd
    spiralfiteval compare data/spiralfits/fit-spacing/fitspacing-Aon30.npz \
      data/spiralfits/fit-spacing/fitspacing-Aoff30.npz --name Aon30 --name Aoff30 \
      --references PHercParis4 --z 8400 9400 --chains odd --out $FS/compare-Aon30-Aoff30.json

The three-seed mean and its CI are not a `spiralfiteval` command; the per-seed numbers they
combine are in the three patch compare files.
SHA-256 of the eight 30k `.npz` files: Aon30 `0b166d24...c45a16`, Aoff30 `de07c753...a1d5`,
Aon30-s2 `434053a9...ffe2`, Aoff30-s2 `e7949c5f...9b44`, Aon30-s3 `0b216bb8...4327`,
Aoff30-s3 `1da266db...14a7`, A2on30 `68fb4387...8283`, A2off30 `2224b90d...cf3d`.
