# K100 runtime and environment separation

| Component | Study setting |
|---|---|
| Accelerator | Hygon K100 AI accelerator |
| Host/container OS | Ubuntu 20.04 / Ubuntu 22.04.5 |
| Inference ONNX Runtime | 1.19.2, vendor MIGraphX integration |
| MIGraphX | 5.1.0 |
| DTK / HIP | 25.04.2 / 6.3.25405 |
| PyTorch / ONNX for preparation | 2.5.1 / 1.22.0 |
| Offline quantization ONNX Runtime | 1.26.0, separate environment |
| C++ | C++17; CMake 3.20 or later as specified by the source build |

`requirements-analysis.txt` supplies packages for offline analysis. Auxiliary versions absent from the study records are not presented as a complete lockfile. The flood preparation source identifies TerraTorch 1.2.8; the model dependency list is a starting specification, not a claim of cross-version equivalence.

Install the inference software through the applicable Hygon distribution. A generic PyPI ONNX Runtime wheel does not provide the paper's vendor stack. Keep offline quantization and measured inference environments separate.

The retained flood exporter additionally requires CUDA for its PyTorch-to-ONNX equivalence comparison. It is a model-preparation procedure, not a K100 inference command. Supply an appropriate preparation environment rather than changing its numerical equivalence conditions.

The CMake files expose `K100_DTK_ROOT`; its source default is `/opt/dtk-25.04.2`. Build the desired runner in a dedicated directory:

```bash
cmake -S cpp -B build/flood -DPRITHVI_TASK=flood -DK100_DTK_ROOT=/opt/dtk-25.04.2
cmake --build build/flood
cmake -S cpp -B build/cloud -DPRITHVI_TASK=cloud -DK100_DTK_ROOT=/opt/dtk-25.04.2
cmake --build build/cloud
```

Expected executable locations are `build/flood/flood/k100_pipeline_benchmark` and `build/cloud/cloud/k100_pipeline_benchmark`. The optional compilation support executable is `build/<task>/lsmod`. The bundled support source implements the study's container compatibility behavior; assess its suitability for the target container before using it as the compiler shim.

Compilation host memory and inference device memory are different quantities. The controlled timing protocol is recorded in `configs/benchmark/controlled_latency.json`. Container images, drivers, proprietary libraries and compiled binaries are external dependencies.
