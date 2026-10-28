# Data notice

The MIT licence covers the code in this directory only. It does not cover any scroll data,
annotations or fits.

`spiralfiteval` ships **no scroll data**, apart from one documentation figure with crops of one
CT slice (see [What the outputs contain](#what-the-outputs-contain)). It reads no CT intensities
at all: it scores fit meshes against point annotations. `spiralfiteval fetch` downloads public files at run time from the
Vesuvius Challenge data server (`https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/`):

- `relative_windings.json`, `same_windings.json`, `abs_winding.json` (the `spiral-input`
  dataset, 2026-07): human winding annotations of PHerc. Paris 4;
- `eval_fibers/` (645 fibers traced in VC3D, about 290 MB): the organisers' evaluation set.

The fits you score are your own or third-party files and keep their authors' terms. The example
fits (Jashann's `vesuvius-scrolling` v1.1 release, Armando Gaona's PHerc. 0211 repository) are
fetched from their GitHub pages; see `README.md` § Prior work and credits.

Scrolls touched, all as coordinates only: **PHerc. Paris 4** (Scroll 1; annotations, eval
fibers and fits are in the frame of scan `20260411134726`, 2.4 µm, used at level 2 = 9.6 µm),
**PHerc. 0826** and **PHerc. 0211** (fits only).

## Licence and terms

- Vesuvius Challenge data is licensed under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) unless a file says otherwise,
  and is subject to the data server terms at <https://dl.ash2txt.org/LICENSE.txt>. By downloading
  it you accept those terms yourself. They include: no redistribution without written approval
  from Vesuvius Challenge, and no publication of revealed hidden text without written consent.
- Scrolls 1–4 and Fragments 1–6 scanned at Diamond Light Source before 2025, including
  PHerc. Paris 4, belong to the **EduceLab-Scrolls Dataset, © EduceLab / The University of
  Kentucky**. Vesuvius Challenge has permission to use them, with the extra citation terms below
  (<https://scrollprize.org/data#how-to-cite>).
- **PHerc. Paris 4 text is reserved.** Its title prize is open until 2027-06-25, and the Vesuvius
  Challenge rules ask that no reading of new text from it be made public before the prize is
  announced. Nothing this tool writes can carry text (see below). Do not combine its outputs with
  renders of Paris 4 in anything you publish without the organisers' approval.

## Required citation

If you publish or present results that use the data fetched by this tool, cite the dataset of
the scans the annotations sit on:

- **EduceLab-Scrolls** (PHerc. Paris 4 and the other legacy DLS scans). In any published
  abstract, cite EduceLab-Scrolls as the source of the data. In any published manuscript,
  reference: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023).
  *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT*.
  ArXiv [Cs.CV]. https://doi.org/10.48550/arXiv.2304.02084. In the methods section, include
  language like: "Data used in the preparation of this article were obtained from the
  EduceLab-Scrolls dataset [above citation]."
- **Vesuvius Challenge CT scans** (scans released directly by Vesuvius Challenge, which includes
  the 2026 Paris 4 scan `20260411134726` and the PHerc. 0826 / 0211 scans): Giorgio Angelotti,
  Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau,
  Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey,
  Youssef Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge – CT Scans of Herculaneum
  Papyri*. Vesuvius Challenge. 2026.

The human winding annotations and eval fibers were made and published by the Vesuvius Challenge
team. Credit the fit authors (Jashann; Armando Gaona) when you report
scores of their fits.

## What the outputs contain

Every output of `spiralfiteval` is geometry only: spiral times, normal distances, counts, shares
and confidence intervals in a JSON file. It renders no images of the scan and contains no ink,
letters or text. The experiment figures (`experiments/2026-10-13-spiralfit-eval/fig_*.png`)
are score plots, or annotation and fit positions drawn as coloured points on a blank background.

One documentation figure is different: `docs/dup_winding_0211.png` (shown in `README.md`) is drawn
on the raw CT. It shows one horizontal slice (z = 8500) of the public PHerc. 0211 scan
`20250821151803` (9.362 µm), as a downsampled overview and two crops of about 1.5 mm, with Armando
Gaona's published fit drawn on top. It is derived from Vesuvius Challenge data and falls under the
same terms (CC BY-NC 4.0: attribution, non-commercial use only). It is a cross-section of the
rolled scroll, not a flattened surface, and shows no ink, letters or text.
