# Reference Audit

## Summary

- Final bibliography size: **26** entries.
- Cited keys: **26**.
- Missing keys: **0**.
- Unused keys: **0**.
- Newly added entries: **11**.
- Materially updated entry: **1** (`prithvi2024`).
- Access date used for the two runtime documentation entries: **2026-08-19**.
- No title, author, DOI, page range, or venue was generated without a checked publisher, proceedings, DOI, or official project record.

## Updated record

| Key | Final metadata | Verification source | Status |
|---|---|---|---|
| `prithvi2024` | D. Szwarcman, S. Roy, P. Fraccaro *et al.*, “Prithvi-EO-2.0: A Versatile Multitemporal Foundation Model for Earth Observation Applications,” *IEEE TGRS*, vol. 64, pp. 1--20, 2026, DOI `10.1109/TGRS.2025.3642610` | [IEEE Xplore](https://ieeexplore.ieee.org/document/11296896), [DOI](https://doi.org/10.1109/TGRS.2025.3642610) | Verified; formal journal record replaces the preprint-style citation. |

## Newly added records

| Key | Coverage | Checked metadata | Verification source | Status |
|---|---|---|---|---|
| `scalemae2023` | Multiscale GeoFM representation learning | Reed *et al.*, ICCV 2023, pp. 4088--4099 | [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.html) | Verified |
| `croma2023` | Radar--optical foundation representations | Fuller, Millard, and Green, NeurIPS 2023, vol. 36, pp. 5506--5538 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1189e7f5cd8976e9e3c2de4a234ea29e-Abstract-Conference.html), DOI `10.52202/075280-0241` | Verified |
| `ringmo2023` | Remote-sensing foundation model | Sun *et al.*, IEEE TGRS 61, pp. 1--22, 2023, DOI `10.1109/TGRS.2022.3194732` | [DOI](https://doi.org/10.1109/TGRS.2022.3194732) | Verified |
| `gfm2023` | Continual pretraining for GeoFMs | Mendieta *et al.*, ICCV 2023, pp. 16806--16816 | [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.html) | Verified |
| `skysense2024` | Multimodal remote-sensing foundation model | Guo *et al.*, CVPR 2024, pp. 27672--27683 | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html) | Verified |
| `dofa2024` | Multimodal Earth-observation foundation model | Xiong *et al.*, arXiv:2403.15356, DOI `10.48550/arXiv.2403.15356` | [arXiv](https://arxiv.org/abs/2403.15356) | Verified |
| `vitptq2021` | Vision Transformer post-training quantization | Liu *et al.*, NeurIPS 2021, vol. 34, pp. 28092--28103 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2021/hash/ec8956637a99787bd197eacd77acce5e-Abstract.html) | Verified |
| `haq2019` | Hardware-aware mixed-precision allocation | Wang *et al.*, CVPR 2019, pp. 8612--8620 | [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html) | Verified |
| `mixqvit2026` | Layer-importance/quantization-sensitivity mixed precision | Ranjan and Savakis, CVPR Workshops 2026, pp. 3599--3609 | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/ECV/html/Ranjan_Mix-QViT_Mixed-Precision_Vision_Transformer_Quantization_Driven_by_Layer_Importance_and_CVPRW_2026_paper.html) | Verified |
| `mlperf2020` | Reproducible inference benchmarking | Reddi *et al.*, ISCA 2020, pp. 446--459, DOI `10.1109/ISCA45697.2020.00045` | [DOI](https://doi.org/10.1109/ISCA45697.2020.00045) | Verified |
| `sang2026` | Onboard deployment review for remote-sensing foundation models | Sang *et al.*, *Remote Sensing*, 18(2), article 298, 2026, DOI `10.3390/rs18020298` | [Publisher](https://www.mdpi.com/2072-4292/18/2/298), [DOI](https://doi.org/10.3390/rs18020298) | Verified |

## Runtime documentation retained and scoped

| Key | Purpose | Source | Scope guard |
|---|---|---|---|
| `onnxruntime` | ONNX Runtime MIGraphX Execution Provider | [Microsoft ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html) | Used only for provider/runtime context relevant to the evaluated stack. |
| `migraphx` | MIGraphX compilation and ORT EP documentation | [AMD MIGraphX documentation](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/) | Does not substitute for measured K100 placement or kernel evidence. |

## Related Work distribution

- GeoFM and Earth-observation models: Prithvi, SatMAE, Scale-MAE, CROMA, RingMo, GFM, SkySense, DOFA.
- Transformer PTQ and mixed precision: PTQ4ViT, FQ-ViT, RepQ-ViT, HAWQ-V3, ViT PTQ, HAQ, Mix-QViT.
- Deployment/benchmarking/reliability context: MLPerf, WorldCereal, PhiSat-1, onboard deployment review, ONNX Runtime, and MIGraphX documentation.

The Related Work wording remains a scoped statement about the studies reviewed, not a systematic-review or universal novelty claim.

