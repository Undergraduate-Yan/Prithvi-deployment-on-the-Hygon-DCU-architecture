# External artifacts

Large or separately licensed files are not stored in this Git repository. Keep them under a local root such as:

```text
<artifact-root>/
  checkpoint/best.ckpt
  sources/prithvi300_upernet_fp16.onnx
  sources/prithvi300_upernet_int8_backbone_qdq.onnx
  admitted/full_fp16_candidate.onnx
  admitted/full_fp16_candidate.mxr
  runtime/lsmod
  dataset/phase11_frozen_test90/
  bundles/prithvi-k100-m5-v1/
  bundles/prithvi-k100-fp16-full-v1/
```

Update only the `path` fields in a private copy of `required_artifacts.example.json`; do not alter expected hashes to make a mismatched file pass.

```bash
python scripts/verify_external_artifacts.py \
  --manifest artifacts/required_artifacts.example.json \
  --root <artifact-root>
```

## Storage options

- Private object storage or institutional cloud with immutable object versions.
- GitHub/GitLab Release assets for redistributable artifacts.
- Git LFS only when quota and licensing are understood.
- Offline shared filesystem for vendor images and MXR caches.

Never publish SSH credentials, private host addresses, expiring signed URLs, or restricted vendor packages. If a cache is rebuilt on another node/runtime, assign it a new identity and do not overwrite the original manifest.

