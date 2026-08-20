# IEEE Format Audit

## Scope

Audited sources:

- `main.tex` and `IEEE_paper_K100_Prithvi_Deployment_V2.tex`
- `main_CN.tex` and `IEEE_paper_K100_Prithvi_Deployment_V2_CN.tex`
- `references.bib`
- `generate_k100_paper_figures.py`
- all final figure PDFs under `figures/`

Both PDFs were rendered page by page at 120 dpi and inspected. The English paper has 11 letter-size pages; the Chinese paper has 12 letter-size pages.

## Terminology scan

| Search term | Final result | Interpretation |
|---|---:|---|
| `FP16 full` | 0 | Removed. |
| lowercase `full FP16` | 0 | Named artifact is consistently `Full FP16`. |
| `task-track tolerance` | 0 | Replaced by strict cross-provider diagnostic wording. |
| `task tolerance` | 0 | Strict and application tracks remain separate. |
| `end-to-end logits` | 0 | Timing scope is `end-to-end-logits`. |
| `K100 Edge AI Accelerator` | 0 | Device is `K100 AI accelerator`. |
| `sensitivity-guided` | 0 | No over-strong method naming restored. |
| `universal sensitive blocks` | 0 | No universal sensitivity claim. |

Two broad substring scans produce legitimate prose rather than terminology defects: “under the same CLI” is a non-compound noun phrase, and “all 25 segments” is a plural noun phrase. Compound modifiers use `same-CLI` and `25-segment`.

## Tables

- Hardware/software environment table: two-column `table*`, `tabularx`, `\footnotesize`, no `\resizebox`, no overflow.
- Hardware table fields visually checked: container digest, DTK/MIGraphX/ORT versions, timing scope, `K100BundleRunner.run`, incremental VRAM definition.
- Strict diagnostic Table V preserves Full FP16 and M0--M5 values, including M4 MAE `0.108455` and maximum absolute error `0.393831` after table rounding.
- Capacity and task tables include Full FP16 and M0--M5 without column collision.

## Figures and floats

- Figure 1: vector five-layer audit overview restored at Method start; arrows and labels are readable.
- Figure 2: backend-discrepancy map remains vector and uses the diagnostic—not universal sensitivity—interpretation.
- Figure 3: qualitative flood figure is a centered single-column float. It no longer interrupts the logical-capacity sentence.
- Figure 4: M5 parity annotation is dark text in a white bordered box with a short arrow; Full FP16 callout is separate.
- All four figures have synchronized English and Chinese versions where applicable; all `\ref` targets resolve.

## References and final-page balance

The native trigger was compiled for every value 7--12. It did not adequately balance both final pages. The final build uses the IEEE-compatible `balance` package:

- English last-page column content bottoms: approximately 401.4 pt and 399.6 pt; difference 1.8 pt.
- Chinese last-page column content bottoms: approximately 649.7 pt and 651.8 pt; difference 2.1 pt.

Reference numbering is continuous 1--26. Visual inspection found no broken entry, clipped URL, or margin crossing.

## Static checks

- Duplicate labels: 0 in both manuscripts.
- Missing citation keys: 0.
- Unused BibTeX keys: 0.
- TODO/TBD/FIXME/PLACEHOLDER: 0.
- Missing figures: 0.
- Overfull boxes: 0 in both final logs.
- Undefined citations/references: 0 in both final logs.
- All listed PDF fonts are embedded and subset.
- Paper size: US Letter, 612 x 792 pt, IEEE two-column layout.
- Anonymous author information remains present.

## Visual QA pages

- English: pages 3, 6, 7, 9, and 11 received focused inspection; all 11 pages were also reviewed in a contact sheet.
- Chinese: pages 3, 6, 10, 11, and 12 received focused inspection; all 12 pages were also reviewed in a contact sheet.

