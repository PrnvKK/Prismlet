# Rebuilding the report

The report is `main.tex`, a single-column `article` class document. The figures are
PDF files in `figures/`.

Build with a LaTeX installation that includes `pdflatex`, `bibtex`, `booktabs`,
`natbib`, `siunitx`, `microtype` and `hyperref`:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The output is `main.pdf`. The committed copy is `prismlet_report.pdf`.

The plots were generated from the public repository's `results/` files. The
hand-drawn diagrams are copied from `docs/diagrams/` and are not edited.
