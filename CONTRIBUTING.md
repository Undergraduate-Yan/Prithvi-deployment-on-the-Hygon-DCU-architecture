# Contributing

Changes should preserve the distinction between frozen evidence and new experiments.

- Never overwrite an existing evidence directory; create a new run root.
- Record model, cache, protocol, container, and script identities with SHA256.
- Keep strict cross-provider diagnostics separate from 90-image task acceptance.
- Do not claim INT8 acceleration from ONNX datatypes, provider placement, or kernel names alone.
- Run `python scripts/repo_check.py` and the relevant static tests before proposing a change.
- New metrics must identify the execution topology, timing scope, warm-up count, repetition count, independent trial count, and device-memory definition.
- Do not submit secrets, proprietary binaries, model weights, datasets, MXR caches, or raw user paths.

