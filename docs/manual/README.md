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

## Verification

After compilation, render every page for visual inspection:

```bash
mkdir -p ../../tmp/pdfs/rendered
pdftoppm -png -r 120 ../../output/pdf/manual.pdf ../../tmp/pdfs/rendered/page
```
