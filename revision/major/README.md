# Major revision reproduction

This directory accompanies *Onboard Deployment of Remote Sensing Foundation Models with a Server-Class DCU*. The experiments were run on ground-based Hygon K100 hardware. In the proposed flight design the server-class CPU handles camera input and scheduling, and the DCU performs onboard inference. The 50 kg allocation and 1800 W peak electrical limit are design budgets for the proposed onboard payload computer.

## Manuscript and supplementary material

| Material | File |
| --- | --- |
| Highlighted manuscript, editable | [Word](manuscript/Manuscript_Revised_Highlighted.docx) |
| Highlighted manuscript | [English PDF](manuscript/Manuscript_Revised_Highlighted.pdf) · [Chinese PDF](manuscript/Manuscript_Revised_ZH_Highlighted.pdf) |
| Revised supplementary sections S12–S19 and Tables S28–S40 | [Supplementary revision addendum](manuscript/Supplementary_Revision_Addendum.docx) |
| Original supplementary sections S1–S11 and Tables S1–S27 | [Original supplementary archive](manuscript/Supplementary_Materials_EN.zip) |

The original supplementary archive retains the manuscript identity of the original submission.

## Offline checks

Run from the repository root with Python 3 and NumPy. The analysis environment in [requirements-analysis.txt](../../environment/requirements-analysis.txt) supplies NumPy.

```bash
python revision/major/python/verify_offline.py
```

The checks use the compact records under [evidence](evidence). They recalculate the external flood event-weighted estimates and intervals and check the paired system measurements against retained tables. The original flood interval script can also be run directly.

```bash
python revision/major/evidence/flood/recompute_event_intervals.py
```

## Evidence and code map

| Revised analysis | Files |
| --- | --- |
| External flood evaluation and event-weighted statistics, Table 10 and S34–S36 | [Flood evidence](evidence/flood/) and [frozen inference worker source](hardware/flood/tools/) |
| Precision candidate and stability analyses, S28–S30 | [Precision evidence](evidence/precision/) |
| Numerical checks, compilation and design budgets, S31–S33 | [Numerical and platform evidence](evidence/platform/) |
| Matched flood topology, same-node cloud resources and scene processing, Tables 9, 11, 12 and S37–S40 | [System evidence](evidence/system/) and [measurement source](hardware/system/) |

The external flood assessment includes 195 windows from 33 scenes in 21 conservative event groups. FP16 and C5 met the prespecified joint noninferiority criteria on that cohort. The earlier evaluation using 15 image tiles from one flood in Bolivia retains its original result. The 90-image development-exposed pool remains descriptive. The event-weighted bootstrap and pooled metrics use different weighting.

Equation (3) in the revision defines the propagated discrepancy with the CPU operator held fixed. This matches the retained diagnostic implementation in [the blockwise diagnostic source](../../src/diagnostics/diagnose_blocks.py). The precision ranking, C5 assignment and saved task results retain their existing values.

System records use five matched process pairs. Flood control holds Node 3, Runner-A, the C5 backbone assignment and the FP32-compatible head fixed while comparing 25 and 13 segments. The cloud scene timer starts with reading stored, scaled six-band reflectance products and ends at mask and metadata output. Energy records describe the DCU package. Path columns that identified local or remote storage were omitted from the public CSV copies. Numerical columns and source digests were retained.

## Hardware inputs

The [hardware sources](hardware/) document the execution path used during the revision. See [K100 runtime setup](../../environment/k100_runtime.md) and [external inputs](../../docs/external_requirements.md). The flood worker expects the frozen protocol, runtime configs, sealed image archive and exact model and runtime artifacts described by those records. The cloud worker uses the Linux runner interface, telemetry paths, runtime plan, scene list and preprocessed scene inputs. The saved measurement scripts retain their original working-directory and sysfs conventions. A full hardware run requires the matching artifacts and environment.

The external flood dataset is WorldFloods v2. Cite the [dataset paper](https://doi.org/10.1038/s41598-023-47595-7). [License and third-party terms](../../docs/licensing.md) apply to the software and other materials according to their scope.
