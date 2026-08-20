# K100 final deployment acceptance aggregator

`aggregate_final_deployment_acceptance.py` is a read-only, fail-closed final
aggregator for the immutable M5 and FP16-full bundles. It does not execute a
model, alter a bundle, edit an old receipt, repair a cache, or select the Pareto
winner. It publishes only into a new output directory.

## Required evidence layout

Each candidate uses the same locked bundle on both nodes and must preserve:

```text
<acceptance-root>/
  static_verification.json
  cold_start_5/result.json
  recovery_3/result.json
  smoke_k100_2/
    result.json
    target_runtime_fingerprint.json
  smoke_k100_3/
    result.json
    target_runtime_fingerprint.json
    container_launch_attestation.json
    container_launcher_stdout.log
    container_launcher_stderr.log
    container_smoke/
      result.json
      target_runtime_fingerprint.json
  stability_60min/
    result.json
    telemetry_5s.jsonl
```

The K100-3 launch attestation is mandatory. The exact bundle image remains
locked; no alternate image or no-hyhal result can substitute for formal smoke.

Two known failed environmental attempts live under a separate immutable root:

```text
<exclusion-root>/
  environment_exclusion_ledger.json
  runs/<frozen file or directory tree for late-PATH attempt>
  runs/<frozen file or directory tree for no-hyhal HIP-100 attempt>
```

The ledger schema is
`phase11_k100_invalid_environmental_run_exclusions_v1`, status is
`locked_invalid_environmental_runs_excluded`, and it contains exactly these
two failure modes:

- `bdc7_with_node3_host_hyhal_lsmod_recursion_before_model`, stage
  `runtime_import_before_model_inference`;
- `bdc7_without_host_hyhal_hip100_no_rocm_device`, stage
  `model_session_creation_before_successful_inference`.

Each entry must lock a permitted M5/FP16 manifest SHA, the official image ID,
K100-3, at least one locked file or directory tree below the exclusion root,
and literal
`accepted_environment=false`, `counts_toward_acceptance=false`,
`model_execution_started=false`, `successful_model_inferences=0`. The ledger
claims must state that the runs satisfy no gate, old receipts are not
overwritten, and a fresh formal rerun is required. The aggregator checks the
detached ledger SHA and re-hashes every referenced artifact.

An empty failed output directory is valid evidence of “no receipt was
produced”: the builder records its canonical empty tree SHA. Adding any file
later changes that tree identity and fails aggregation.

The later launcher permission bug is a different class of evidence: its inner
model smoke passed, but the old launcher could not write the outer attestation
because Docker created a root-owned output directory. Keep it in a separate
operational-failure root and lock it once:

```bash
python3 build_operational_tool_failure_ledger.py \
  --operational-root /absolute/old_launcher_failure_root \
  --failed-launcher-dir /absolute/old_launcher_failure_root/<preserved_failed_dir> \
  --candidate-label M5 \
  --manifest-sha256 '<M5 manifest SHA>' \
  --official-image-id 'sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01' \
  --output /absolute/old_launcher_failure_root/operational_tool_failure_ledger.json
```

This ledger explicitly records `passed_observation_only` for the inner smoke,
missing outer attestation, and `counts_toward_acceptance=false`. It cannot
replace the required fresh launcher run.

Create it once with the supplied builder; both artifact paths and the new
ledger must stay below the same exclusion root:

```bash
python3 build_environment_exclusion_ledger.py \
  --exclusion-root /absolute/invalid_environment_runs \
  --late-path-artifact /absolute/invalid_environment_runs/runs/late_path.log \
  --nohyhal-hip100-artifact /absolute/invalid_environment_runs/runs/nohyhal_hip100.log \
  --manifest-sha256 '<M5 manifest SHA used by both attempts>' \
  --official-image-id 'sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01' \
  --output /absolute/invalid_environment_runs/environment_exclusion_ledger.json
```

## Formal K100-3 launcher

The precise failure mechanism occurs before the in-container wrapper runs: an
image `LD_PRELOAD` initializer calls `lsmod`. Therefore an `export PATH=...`
inside the wrapper is too late. Run the supplied host launcher, which injects
the locked PATH through `docker run -e` before entrypoint and records the exact
Docker argv:

```bash
IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
BUNDLE='/absolute/host/bundle'
MANIFEST_SHA='<detached lowercase manifest SHA256>'
SMOKE='/absolute/host/acceptance/smoke_k100_3'  # must not exist

python3 launch_cross_node_smoke_container.py \
  --bundle "${BUNDLE}" \
  --expected-manifest-sha256 "${MANIFEST_SHA}" \
  --output-dir "${SMOKE}" \
  --official-image-id "${IMAGE}" \
  --host-hyhal /opt/hyhal \
  --device 0 \
  --node-label K100-3
```

The launcher passes exactly:

```text
PATH=/bundle/tools/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
PHASE11_K100_PREENTRYPOINT_PATH_ATTESTATION=docker_run_environment_before_entrypoint_v1
```

It also verifies the immutable Docker image ID, detached manifest SHA and
static `lsmod` identity (819664 bytes,
`9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12`),
mounts the bundle and `/opt/hyhal` read-only, and writes a launch attestation
whose `smoke_result` identity must match the formal result. A launcher failure
creates only failed evidence and never becomes a gate.

`/opt/hyhal` may be a symlink. The launcher records the requested path, resolves
its real path before Docker, and mounts the resolved directory. It atomically
creates the formal output root as the invoking host user before Docker; the
container writes only `container_smoke/`, after which the host copies the exact
result/fingerprint bytes to the formal root and writes logs plus attestation.

Existing passed v2 smoke directories that contain only result/fingerprint and
stdout/stderr remain useful auxiliary observations, but they do not prove the
outer Docker argv. Preserve them unchanged. The final aggregator intentionally
accepts only a fresh new-directory run made by this launcher; do not append a
retrospective boolean attestation to an old v2 directory.

Do not use the successful no-hyhal fingerprint as smoke: the subsequent HIP
100 failure proved that configuration has no usable K100 device. If a separate
K100-3 60-minute stability run is later required, use the same exact image,
host-hyhal mount and pre-entrypoint PATH contract; never substitute a no-hyhal
run merely because telemetry is easier to collect.

## Final aggregation command

```bash
M5_BUNDLE=/absolute/bundles/prithvi-k100-m5-v1
FP16_BUNDLE=/absolute/bundles/prithvi-k100-fp16-full-v1
M5_ACCEPT=/absolute/receipts/m5
FP16_ACCEPT=/absolute/receipts/fp16
EXCLUSION_ROOT=/absolute/invalid_environment_runs
EXCLUSION_LEDGER="${EXCLUSION_ROOT}/environment_exclusion_ledger.json"
OPERATIONAL_ROOT=/absolute/old_launcher_failure_root
OPERATIONAL_LEDGER="${OPERATIONAL_ROOT}/operational_tool_failure_ledger.json"

# Prefer detached SHA values printed by the original builders/ledger creation.
M5_SHA=$(sha256sum "${M5_BUNDLE}/manifest.json" | awk '{print $1}')
FP16_SHA=$(sha256sum "${FP16_BUNDLE}/manifest.json" | awk '{print $1}')
EXCLUSION_SHA=$(sha256sum "${EXCLUSION_LEDGER}" | awk '{print $1}')
OPERATIONAL_SHA=$(sha256sum "${OPERATIONAL_LEDGER}" | awk '{print $1}')

python3 aggregate_final_deployment_acceptance.py \
  --m5-bundle "${M5_BUNDLE}" \
  --m5-manifest-sha256 "${M5_SHA}" \
  --m5-acceptance-root "${M5_ACCEPT}" \
  --fp16-bundle "${FP16_BUNDLE}" \
  --fp16-manifest-sha256 "${FP16_SHA}" \
  --fp16-acceptance-root "${FP16_ACCEPT}" \
  --environment-exclusion-root "${EXCLUSION_ROOT}" \
  --environment-exclusion-ledger "${EXCLUSION_LEDGER}" \
  --environment-exclusion-ledger-sha256 "${EXCLUSION_SHA}" \
  --operational-failure-root "${OPERATIONAL_ROOT}" \
  --operational-failure-ledger "${OPERATIONAL_LEDGER}" \
  --operational-failure-ledger-sha256 "${OPERATIONAL_SHA}" \
  --output-dir /absolute/new/final_acceptance_v1 \
  --write-markdown
```

All six input roots must be distinct and non-overlapping. The output must be a new normal leaf
outside every input root, with an existing parent. Missing or tampered evidence
produces a new failed aggregate and exit code 2; it never overwrites evidence.

## Mandatory gates

For both M5 and FP16:

1. The detached manifest matches and every bundle payload path, size and SHA is
   re-hashed, including the origin runtime fingerprint.
2. Static verification locks the matching bundle, execution kind, manifest and
   payload totals.
3. Cold start contains exactly five successful fresh-process inferences with
   exact image, MIGraphX, no CPU fallback, locked static `lsmod`, strict I/O and
   fixed-prediction evidence.
4. Recovery contains exactly three active SIGTERM/SIGKILL cycles followed by
   three successful fresh reloads.
5. K100-2 and K100-3 have distinct hostnames, exact node labels, an origin-
   identical portable runtime fingerprint, successful cache inference and the
   fixed prediction. K100-3 additionally requires the validated Docker-level
   pre-entrypoint PATH/host-hyhal launch attestation.
6. Stability observes at least 3600 seconds with at least 720 telemetry samples,
   all intervals 4.0--7.5 seconds, temperature/power/absolute VRAM in every
   sample, at least one inference, and zero errors, non-finite outputs or fixed-
   prediction drifts. The raw JSONL must exactly reproduce the locked summary.
7. The two known invalid environmental attempts are separately identity-locked
   and explicitly excluded from every gate.
8. The known host-output permission failure is locked in its independent
   operational-tool-failure ledger and explicitly excluded from every gate.

The output directory contains JSON and CSV plus optional Markdown, each with an
exact SHA printed by the tool. A passed aggregate proves operational deployment
acceptance only. It does not repair the immutable strict-logits failure, prove
native INT8 kernel precision, establish an INT8/FP16 hardware peak comparison,
or promote M5 over FP16 full.
