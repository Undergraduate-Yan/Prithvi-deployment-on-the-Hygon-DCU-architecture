# Verified runtime fingerprint

The paper's formal environment reports:

| Component | Verified value |
|---|---|
| Accelerator | K100 AI accelerator; 65,520 MiB device memory observed |
| Nodes | Two K100 nodes used for final cross-node evidence |
| Host OS | Ubuntu 20.04 LTS; Linux 5.4.0-216-generic |
| Host memory | 62 GiB visible RAM |
| Container base | Ubuntu 22.04.5 |
| Python | 3.10.12 |
| PyTorch | 2.5.1 |
| HIP | 6.3.25405 |
| DTK | 25.04.2 |
| MIGraphX | 5.1.0 + `das.opt1.dtk25042` |
| ONNX Runtime | 1.19.2 + `das.opt1.dtk25042` |
| Formal image ID | `sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01` |
| Batch | 1 |
| Formal timing | 30 warm-ups, three fresh processes, 100 timed calls per process |

The vendor ONNX Runtime/MIGraphX integration is not reproducible with an arbitrary PyPI ONNX Runtime wheel. Capture and compare runtime fingerprints with the supplied scripts before accepting a new run.

