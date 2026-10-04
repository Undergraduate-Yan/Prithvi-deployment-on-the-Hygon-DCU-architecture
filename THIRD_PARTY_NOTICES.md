# Third-party resources

| Resource | Use in this study | Distribution in this repository |
|---|---|---|
| Prithvi-EO-2.0 | Pretrained encoder and downstream model construction | Acquisition instructions and identities; no pretrained weights |
| TerraTorch / TorchGeo / PyTorch / Lightning | Task models, data modules, training and export | Referenced as dependencies; no vendored package tree |
| ONNX / ONNX Runtime | Graph representation, quantization and runtime | Research integration code; no vendor runtime binaries |
| Hygon DTK / HIP / MIGraphX integration | K100 compilation and execution | Environment description and build interfaces; no vendor SDK or image |
| Sen1Floods11 | Flood imagery and labels | Selection identifiers and derived numerical records; no raster payloads |
| CloudSEN12+ | Cloud imagery and labels | Source locations, selection identifiers and derived numerical records; no raster payloads |
| NumPy / rasterio / OpenCV / PyArrow / Matplotlib | Numerical processing, raster input and diagnostic plots | Referenced as dependencies |

Each third-party resource retains its own license, attribution requirements and redistribution conditions. The study code's license notice does not grant additional rights to these resources. Consult the license distributed with the exact component or dataset revision used.

No third-party weights, container images, runtime libraries or local credentials are distributed as repository payloads.
