# MIGA Controller LaTeX manual

This directory contains the English operation, optimization and analysis manual for the
`marker-optimization` branch.

## Build

From this directory, run either:

```bash
tectonic manual.tex --outdir ../../output/pdf
```

or, with a conventional TeX Live installation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../output/pdf manual.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../output/pdf manual.tex
```

The author is `Yiming MENG`. The title page and revision note use `\today`, so the displayed
date is the actual rendering date.

## Inputs

- `manual.tex`: complete source
- `assets/ui-*.jpg`: screenshots captured from the documented checkout

## Archive Server companion

`archive_server_manual.tex` is the standalone English Archive Server companion,
reviewed against `marker-optimization` revision `39067d3` on 9 October 2026. It
shares the Controller manual's A4 book layout, palette, running headers, procedure
blocks and code listings. It covers NAS/SSH preparation, the five-step Guide,
device management, dashboard, backup scheduling and overrides, Pull all sources,
integrity and revisions, SYNC/Collections, analysis, updates/reload and troubleshooting.
No additional images or LaTeX input files are required.

Open the source in Codex's LaTeX editor for the live PDF preview. Alternatively,
when an existing TeX environment is available, compile from this directory:

```bash
tectonic archive_server_manual.tex --outdir ../../output/pdf
```

The intended published copy is `miga_archive_server_manual.pdf` in this directory,
after rendering and visual verification of `output/pdf/archive_server_manual.pdf`.
At authoring time the built-in compiler could not fetch its uncached Tectonic v33
bundle (`relay.fullyjustified.net`); compilation and PDF visual QA are therefore
pending, and no verified Archive Server PDF has been published. This is an
environment/resource-download failure, not a confirmed LaTeX source error.

## Verification

After compilation, render every page for visual inspection:

```bash
mkdir -p ../../tmp/pdfs/rendered
pdftoppm -png -r 120 ../../output/pdf/manual.pdf ../../tmp/pdfs/rendered/page
```
