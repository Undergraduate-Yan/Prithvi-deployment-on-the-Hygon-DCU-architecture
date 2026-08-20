# Final Build Report

## Build commands

The repository-required LaTeX skill compiler was used; no PDF was edited directly.

```powershell
C:\Users\YCGNotFound\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe \
  C:\Users\YCGNotFound\.codex\skills\latex-paper-en\scripts\compile.py \
  main.tex --recipe pdflatex-bibtex

C:\Users\YCGNotFound\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe \
  C:\Users\YCGNotFound\.codex\skills\latex-paper-en\scripts\compile.py \
  main_CN.tex --recipe xelatex-bibtex
```

Each recipe ran LaTeX, BibTeX, LaTeX, and LaTeX.

## Final status

| Check | English | Chinese |
|---|---:|---:|
| Output | `main_final.pdf` | `main_final_CN.pdf` |
| Pages | 11 | 12 |
| Page size | 612 x 792 pt (US Letter) | 612 x 792 pt (US Letter) |
| Undefined references | 0 | 0 |
| Undefined citations | 0 | 0 |
| Duplicate labels | 0 | 0 |
| Missing figures | 0 | 0 |
| Overfull boxes | 0 | 0 |
| Underfull boxes | 24 | 21 |
| Embedded fonts | all | all |

The underfull boxes are non-clipping line-stretch warnings, concentrated in narrow bibliography/URL lines and compact table text. Visual rendering shows no margin crossing or unreadable row. The Chinese log also records font-shape substitution warnings from IEEEtran/XeLaTeX; the substituted fonts are embedded and the rendered pages are intact.

## Reference-column balancing experiment

For each native trigger value, the table records page count and the last-page left/right content bottoms measured from the rendered PDF. A zero right value denotes an empty final right column.

| Trigger | English pages | English L/R bottom (pt) | Chinese pages | Chinese L/R bottom (pt) |
|---:|---:|---:|---:|---:|
| 7 | 11 | 691.3 / 266.9 | 12 | 694.6 / 616.6 |
| 8 | 11 | 690.3 / 240.0 | 12 | 694.6 / 581.5 |
| 9 | 11 | 691.1 / 195.1 | 13 | 532.7 / 0.0 |
| 10 | 11 | 691.3 / 123.4 | 13 | 437.1 / 0.0 |
| 11 | 11 | 691.3 / 87.5 | 13 | 388.3 / 0.0 |
| 12 | 11 | 96.5 / 697.2 | 13 | 339.5 / 0.0 |

No native value in 7--12 balanced both versions. The final `balance` fallback produces:

- English: 401.4 / 399.6 pt (difference approximately 1.8 pt).
- Chinese: 649.7 / 651.8 pt (difference approximately 2.1 pt).

No negative spacing or reference deletion was used.

## Figure and layout decisions

- Figure 3 requested annotation fix appears as final **Figure 4** because the five-layer overview was restored as Figure 1.
- M5 parity text is dark, boxed, above the bar, and connected by a short arrow. The green bar retains its original value and scope.
- Full FP16 remains a separate same-CLI callout and does not overlap the plot.
- The qualitative figure is a single-column centered float placed after a complete storage paragraph.
- The environment table is a readable two-column `table*` at `\footnotesize`; no `\resizebox` or `scriptsize` compression is used.
- The five-layer overview is included as a vector Figure 1 in both languages.

## Visual QA

Both PDFs were rasterized at 120 dpi and reviewed page by page. Focused checks covered:

- overview figure and Method flow;
- hardware/software table;
- strict diagnostic and sensitivity tables;
- qualitative figure centering and prose continuity;
- M5/Full FP16 trade-off annotation;
- final bibliography numbering and column balance.

## Reproducibility artifacts

- Console captures: `build_en_final_console.txt`, `build_cn_final_console.txt`.
- Rendered QA pages: `tmp/final_render_v2/`.
- Native trigger trials: `tmp/trigger_tests/`.
- Source patch: `FINAL_NINE_ITEMS.patch`.

