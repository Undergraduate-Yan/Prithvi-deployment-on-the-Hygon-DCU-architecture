# K100 minimal deployment toolkit

This directory contains a candidate-agnostic packaging and acceptance framework for either:

- one admitted 25-segment M0--M5 candidate selected by the later Pareto decision; or
- the frozen single-session FP16-full deployment baseline.

It does **not** select the winning candidate. Packaging proves only static completeness and byte identity. Runtime, cache portability, stability, recovery, kernel precision, and deployment readiness are separate claims.

## 1. Capture and lock the origin runtime

Run inside the exact official DTK 25.04.2 / MIGraphX 5.1.0 image. The host launcher should export the actual inspected image digest, rather than a tag:

```bash
export PHASE11_K100_IMAGE_ID='sha256:<64 lowercase hex digits>'
LSMOD_SHIM='/frozen/evidence/tools/lsmod'
python3 capture_k100_runtime_fingerprint.py \
  --official-image-id "${PHASE11_K100_IMAGE_ID}" \
  --lsmod-shim "${LSMOD_SHIM}" \
  --output /new/output/origin_runtime.json
```

`LSMOD_SHIM` is mandatory and must be the frozen static executable with size `819664` bytes and SHA256 `9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12`. The capture tool verifies it and prepends its directory to `PATH` **before** importing ONNX Runtime or Torch. If `PHASE11_K100_IMAGE_ID` is absent, the fingerprint is retained only as a diagnostic; the bundle builder and all formal runtime acceptance refuse it.

## 2. Build one immutable bundle

The fixed input must be exactly float32 `1x6x224x224`. A `.npz` input must contain only the key `input`. It must already have the six-band `1e-4` scaling. The expected output must be an `.npz` containing exactly float32 `logits` (`1x2x224x224`) and uint8 `prediction` (`1x224x224`).

For whichever 25-segment candidate wins the later Pareto decision:

```bash
python3 build_k100_bundle.py \
  --kind segment25 \
  --bundle-id prithvi-k100-<candidate>-v1 \
  --source-root /work \
  --source-manifest /work/candidates/<candidate>/mixed_precision_manifest.final.json \
  --task-admission /work/admission/<candidate>/test90_three_run_summary.json \
  --runtime-fingerprint /new/output/origin_runtime.json \
  --lsmod-shim "${LSMOD_SHIM}" \
  --fixed-input /fixed/fixed_input.npy \
  --expected-output /fixed/expected_output.npz \
  --official-image-id "${PHASE11_K100_IMAGE_ID}" \
  --evidence performance=/evidence/formal_performance_summary.json \
  --output-dir /new/bundles/prithvi-k100-<candidate>-v1
```

For FP16 full:

```bash
python3 build_k100_bundle.py \
  --kind fp16_full \
  --bundle-id prithvi-k100-fp16-full-v1 \
  --source-root /frozen/fp16-full-root \
  --model /frozen/fp16-full-root/prithvi300_upernet_fp16_compat.onnx \
  --cache /frozen/fp16-full-root/prithvi300_upernet_fp16_compat.mxr \
  --source-manifest /frozen/fp16-full-root/evidence/build_report.json \
  --single-admission /frozen/fp16-full-root/evidence/single_result.json \
  --fp16-task-result /frozen/fp16-full-root/evidence/fresh_fixed90_result.json \
  --task-admission /frozen/fp16-full-root/evidence/confirmation_summary.json \
  --runtime-fingerprint /new/output/origin_runtime.json \
  --lsmod-shim "${LSMOD_SHIM}" \
  --fixed-input /fixed/fixed_input.npy \
  --expected-output /fixed/expected_output.npz \
  --official-image-id "${PHASE11_K100_IMAGE_ID}" \
  --output-dir /new/bundles/prithvi-k100-fp16-full-v1
```

The builder prints the exact `manifest.json` size and SHA256. Preserve that detached value with the experiment report.

For a segmented bundle, the source manifest, task-admission result, all three raw result/NPZ/CSV triplets, all 25 ONNX files, and all 25 MXR files must resolve below `--source-root`; absolute escapes, traversal and symlinks are rejected. The aggregate/common/evaluator generator identities and each raw result are revalidated, so a hand-written summary cannot substitute for three admitted runs. The manifest must be a cache-finalized M0--M5 manifest with the fixed block mapping, repaired `fpn4` head lineage, and matching passed fixed-90 evidence.

The three-run summary records each raw artifact using its original absolute container path (normally `/work/run/...`). Build inside a container where the experiment tree is mounted at that **same absolute path**, and choose a `--source-root` that contains those paths (for example `/work`, if the frozen summary says `/work/run/...`). The builder intentionally does not rebase these records: a renamed or differently mounted evidence tree fails closed and must not be worked around by editing the signed summary.

FP16 full is fail-closed to the frozen compatibility baseline. The builder requires the exact admitted model (`638970735` bytes, SHA256 `8ad6b71482be31ebcc1d5322a9671cc15d44113d0d2597ed715a98e0dc1089f7`), paired MXR (`708193607` bytes, SHA256 `9fe8985dc8ce4a3cf9d829ad66a91de41bb67b22e01b8181c43bf15c74f7dd0a`), locked build report, failed strict single-sample record, fresh 90-sample result and task-confirmation summary. It also inspects internal FLOAT16 initializers/Casts and FP32 LayerNorm islands. An arbitrary FP32 ONNX plus MXR cannot be relabelled as FP16. All six source artifacts must remain below `--source-root`. The task pass does not rewrite the frozen strict-logits failure and does not itself claim deployment completion.

## 3. Static verification and one inference

Verification must write its receipt outside the read-only bundle:

```bash
python3 /bundle/tools/verify_k100_bundle.py \
  --bundle /bundle \
  --expected-manifest-sha256 '<detached manifest SHA256>' \
  --receipt /run/static_verification.json

python3 /bundle/tools/infer_k100.py \
  --bundle /bundle \
  --input /input/example.npy \
  --output /run/example_output.npz \
  --device 0 \
  --expected-manifest-sha256 '<detached manifest SHA256>'
```

The inference CLI never casts or reshapes invalid input and never applies scaling or normalization. It writes float32 logits and a uint8 class map. It fails on a pre-existing output path.

## 4. Runtime acceptance

Each output directory must be new and outside `/bundle`.

```bash
/bundle/tools/run_cold_start_5.sh \
  --bundle /bundle --expected-manifest-sha256 '<detached manifest SHA256>' \
  --output-dir /run/cold_start_5 --device 0

/bundle/tools/run_stability_60min.sh \
  --bundle /bundle --expected-manifest-sha256 '<detached manifest SHA256>' \
  --output-dir /run/stability_60min --device 0

/bundle/tools/run_recovery_3.sh \
  --bundle /bundle --expected-manifest-sha256 '<detached manifest SHA256>' \
  --output-dir /run/recovery_3 --device 0

# Cross-node smoke is launched from the host with the command in Section 5.
# Do not call run_cross_node_smoke.sh from a container whose PATH was not set
# by docker run before the entrypoint.
```

Run the host launcher independently on K100-2 and K100-3. A smoke failure must not trigger an in-place cache rewrite. Recompile the affected MXR files from the locked ONNX on that node, create a **new** bundle/manifest with new SHA values, then rerun admission.

The 60-minute run samples every 5 seconds. It preserves raw `hy-smi --showuse --showmemuse --showtemp --showpower` and `hy-smi --showmeminfo vram` output plus available sysfs counters. Every sample must contain parsed temperature, power, and absolute VRAM-used measurements. Consecutive timestamps must remain between 4.0 and 7.5 seconds. Non-empty command output alone cannot satisfy the gate. Admission additionally requires at least 3600 seconds, zero inference errors, zero non-finite outputs, and zero fixed-sample prediction drift.

## 5. Host container contract

A formal host invocation keeps the bundle read-only and all receipts in a separate writable directory. The launcher passes the locked PATH through `docker run -e`, before the image entrypoint or its `LD_PRELOAD` initializer can import Torch/ONNX Runtime or call `lsmod`:

```bash
IMAGE='sha256:<locked digest>'
BUNDLE='/absolute/host/bundle'
RUN='/absolute/host/acceptance/smoke_k100_3'  # must not exist
python3 /uploaded-tools/launch_cross_node_smoke_container.py \
  --bundle "${BUNDLE}" \
  --expected-manifest-sha256 '<detached manifest SHA256>' \
  --output-dir "${RUN}" \
  --official-image-id "${IMAGE}" \
  --host-hyhal /opt/hyhal \
  --device 0 \
  --node-label K100-3
```

The launcher emits `container_launch_attestation.json` and proves that Docker received exactly `PATH=/bundle/tools/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin` before entrypoint. It resolves a symlinked `/opt/hyhal` to its real host path, creates the formal output root as the invoking host user before Docker, isolates root-owned container files under `container_smoke/`, and then copies the exact formal result/fingerprint bytes to the host-owned root. Merely exporting PATH inside `run_cross_node_smoke.sh` is too late on the known K100-3 host-hyhal path and is not formal evidence. `infer_k100.py` still independently re-verifies the locked 819664-byte static shim before importing ONNX Runtime. No external `lsmod` binary or new `LD_PRELOAD` is introduced.

The earlier exact-image/host-hyhal attempt with PATH injected only inside the wrapper, and the no-hyhal HIP-100 diagnostic, are invalid environmental attempts. Preserve their original logs in a separate exclusion root; neither may be copied into a formal smoke directory or counted as a deployment gate.

CPU-only safety regression tests can be run before upload:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 test_deployment_static.py
```

They cover source-root traversal rejection, M5 mapping/head-lineage rejection, mixed raw-run/generator identity checks, FP16 frozen lineage rejection, incomplete telemetry rejection, excessive sampling gaps, and acceptance-output isolation.

## Claim boundary

| Layer | What the tools can establish | What they do not establish |
|---|---|---|
| Static bundle | complete payload path set, sizes, SHA256, input/output contract | any K100 execution |
| Fixed-sample smoke | cache load, no CPU fallback, finite output, prediction SHA | strict CPU/MIGraphX logits equality |
| 60-minute/recovery | sustained execution, telemetry, prediction stability, reload | native INT8 kernel precision |
| Kernel evidence | must come from the separate `hipprof` I8II/HBH analysis | cannot be inferred from MIGraphX placement |

The historical INT8 strict-logits failure is immutable. The default mixed bundle records native INT8 kernel status as `unverified`; supplying a generic evidence file does not silently promote that claim. FP16 full remains the deployment default until a mixed candidate passes all task, kernel, size, latency, stability, and cross-node upgrade gates.
