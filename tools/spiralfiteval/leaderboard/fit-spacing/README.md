# Fit-spacing rows: one `fit_spiral.py` setting, scored on held-out chains

Three PHerc. Paris 4 spiral fits (z 8400-9400, villa `fit_spiral.py` @ `47111d1`, 120
windings, 12k steps) that differ in one setting from a fit next to them, and the two paired
comparisons that test that setting. These rows are **not** in the main table: they score H1
on the **21 odd human chains only** (`--chains odd`), while every main-table row uses all 39
band chains. Do not rank them against the main table. The odd half is held out because
windingcheck's shipped model was fitted on the even chains; none of these four fits takes a
human annotation as input. The odd half is not untouched, though: I had looked at odd-chain
scores when I chose this setting to test and set the thresholds below.

| row | config | one change from |
|---|---|---|
| `fitspacing-Aon` | patches, no tracks, `dense_spacing_mode = "grad_mag"` | - |
| `fitspacing-Aoff` | as Aon, `dense_spacing_mode = "winding_model"` | Aon (spacing source off, as in the published runs) |
| `fitspacing-A2on` | Jashann's A2 config (tracks, no patches), `dense_spacing_mode = "grad_mag"` | Jashann's published A2 |

Why the setting matters: in villa the `winding_model` spacing source exists only when winding
inference is on. The published runs switch inference off and keep that default, so their
dense-spacing loss has no source; `grad_mag` pulls neighbouring windings to the sheet spacing
read from the CT gradient magnitude.

## Result

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

Download the three fits (release assets, ~20 MB each) and Jashann's A2 grids:

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
