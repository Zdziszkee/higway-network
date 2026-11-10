# Notebook explanation

Typst source for the line-by-line notebook explanation.

- `main.typ`: cover, executive summary, and styling.
- `cells_0_9.typ`, `cells_10_19.typ`, `cells_20_33.typ`: cell walkthroughs.
- `appendix.typ`: shape flow, parameter breakdown, hyperparameter rationale, Q&A, caveats.
- `../highway_notebook_explained.pdf`: compiled 22-page PDF.

Build from this directory:

```bash
typst compile main.typ ../highway_notebook_explained.pdf
```
