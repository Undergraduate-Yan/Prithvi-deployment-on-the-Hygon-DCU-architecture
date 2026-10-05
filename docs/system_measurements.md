# System measurement procedures

The programs distinguish controlled model-call latency, device resource measurements, initialization/recovery and sustained execution. Use the exact task artifacts and development input. A new experiment must establish its own hardware isolation, finite outputs and provider placement; historical PASS fields are not substitutes.

## Host and container preparation

Use Linux with the Hygon K100 vendor stack from `environment/k100_runtime.md`, Bash, Docker, GNU time, NumPy, sysfs telemetry and permission to observe KFD process cgroups. `np.trapezoid` in the original flood energy summarizer requires NumPy 2.0 or later; run analysis separately from a vendor runtime environment if necessary.

All code, inputs, resolved model/cache paths, plans, outputs and the compatibility shim must be inside one workspace mounted at the identical absolute path in the container. A workspace under `/var/tmp` also satisfies the original compile-matrix controller. None of these scripts install the runtime or download an image.

Set these environment variables to real locations on the target host:

```bash
export PRITHVI_WORKSPACE=/var/tmp/prithvi-reproduction
export PRITHVI_REPO="$PRITHVI_WORKSPACE/prithvi-hygon-deployment"
export PRITHVI_SYSTEM_ROOT="$PRITHVI_WORKSPACE/flood-system"
export PRITHVI_CONTAINER_IMAGE=sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01
export PRITHVI_SHIM="$PRITHVI_REPO/build/flood/lsmod"
```

The image identity above is the retained study image, which must already be available through an authorized distribution. Set `PRITHVI_DEVICE_SYSFS` and `PRITHVI_HWMON` to the actual K100 device and its hwmon directory after inspecting the host; card and hwmon numbers are not portable. The sensors are read-only. Do not change the power cap to satisfy a check: a different cap means a different measurement condition.

Build the C++ runner and shim as described in `environment/k100_runtime.md`. Confirm their actual paths from the build outputs. Flood resource scripts use the Python bundle runner and the three task variants. Cloud system scripts use the C++ cloud runner. These measurement populations are not interchangeable.

## Flood inputs and execution

The retained NPY input has SHA256 `7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80`, size 1,204,352 bytes, and FP32 shape `[1,6,224,224]`.

```bash
python scripts/prepare_system_inputs.py flood --artifact-root "$PRITHVI_WORKSPACE/artifacts" --input "$PRITHVI_WORKSPACE/inputs/configuration_validation_input_00.npy" --input-sha256 7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80 --output-dir "$PRITHVI_SYSTEM_ROOT"
bash scripts/system/flood_vram.sh
bash scripts/system/flood_energy.sh
bash scripts/system/flood_cold_start.sh
bash scripts/system/flood_recovery.sh
```

Each script refuses an existing output directory. VRAM uses 20 Hz sampling and 500 calls; energy uses five valid trials with 50 warm-ups and 1000 measured calls at 10 Hz. Startup uses ten valid fresh processes with filesystem caches retained; it is not disk-cold startup. Recovery uses five valid cycles of terminating a container created by the script, waiting for device idle, then reloading. Do not run recovery against an unrelated production workload.

The default output directories below are original interface names. They do not indicate newly obtained evidence until execution and aggregation succeed:

```bash
python scripts/system_experiments.py flood-vram-summary --input-root "$PRITHVI_SYSTEM_ROOT/07_vram_lineage_v2" --output-dir outputs/flood-vram --node K100-host
python scripts/system_experiments.py flood-energy-summary --input-root "$PRITHVI_SYSTEM_ROOT/08_energy" --output-dir outputs/flood-energy --node K100-host
python scripts/system_experiments.py flood-cold_start-summary --input-root "$PRITHVI_SYSTEM_ROOT/09_cold_start" --output-dir outputs/flood-startup --node K100-host
python scripts/system_experiments.py flood-recovery-summary --input-root "$PRITHVI_SYSTEM_ROOT/10_recovery" --output-dir outputs/flood-recovery --node K100-host
```

Cold-start/recovery aggregators accept repeated `--input-root` arguments for additional independently recorded attempts. Preserve rejected trials and their exclusion reasons; do not rename an invalid attempt to make it count. Exclusion rules, sample counts, input identity and cgroup ownership are checked by the aggregation code.

Run stability separately for each of `Mono-FP32`, `Mono-FP16-Opt` and `MP-RCS-Opt`, using the corresponding `00_configs/*.json`. For example:

```bash
bash scripts/system/flood_stability.sh Mono-FP32 "$PRITHVI_SYSTEM_ROOT/00_configs/Mono_FP32.json" "$PRITHVI_SYSTEM_ROOT/02_input/configuration_validation_input_00.npy" "$PRITHVI_WORKSPACE/stability-fp32"
python scripts/system_experiments.py flood-stability-summary --raw-dir "$PRITHVI_WORKSPACE/stability-fp32" --node K100-host --output-dir outputs/flood-stability-fp32
```

The fixed duration is 3600 seconds, telemetry cadence is five seconds, and outputs include inference timings and prediction hashes. Functional stability and isolated system attribution are separate checks. Preserve the actual node for each variant; do not pool a shared-node run into an isolated comparison.

## Cloud plans and latency

`prepare_system_inputs.py cloud` resolves each model/cache against its portable identity manifest and produces `cloud_candidates.json`. Supply the retained normalized development pack and its expected SHA256 from the original input evidence; do not substitute evaluation scenes.

```bash
python scripts/prepare_system_inputs.py cloud --artifact-root "$PRITHVI_WORKSPACE/artifacts" --input "$PRITHVI_WORKSPACE/inputs/cloud_development.npy" --input-sha256 "$CLOUD_DEVELOPMENT_SHA256" --output-dir "$PRITHVI_WORKSPACE/cloud-resolved"
python scripts/system_experiments.py cloud-prepare --config "$PRITHVI_WORKSPACE/cloud-resolved/cloud_candidates.json" --dv-inputs "$PRITHVI_WORKSPACE/inputs/cloud_development.npy" --runner "$PRITHVI_REPO/build/cloud/cloud/k100_pipeline_benchmark" --output-root "$PRITHVI_WORKSPACE/cloud-entry"
```

Set `PRITHVI_CLOUD_ASSETS` to a directory containing `runner/k100_pipeline_benchmark`, built from the packaged cloud source. Set `CPU` to an allowed host CPU. The cloud plan preparation exports the first retained development tile and inspects static tensor interfaces.

```bash
bash scripts/system/cloud_latency.sh "$PRITHVI_WORKSPACE/cloud-entry" "$PRITHVI_WORKSPACE/cloud-latency" "$PRITHVI_SHIM" "$CPU"
```

This runs three 14-session variants, five fresh processes each, 50 warm-ups and 200 calls. Provider profiles must show MIGraphX execution with no CPU kernels. Use the same execution settings on each comparison node.

## Cloud auxiliary and sustained measurements

```bash
bash scripts/system/cloud_auxiliary.sh "$PRITHVI_WORKSPACE/cloud-entry" "$PRITHVI_WORKSPACE/cloud-aux-fp32" "$PRITHVI_SHIM" "$CPU" "$PRITHVI_REPO/src/system/telemetry.py" "$PRITHVI_WORKSPACE/cloud-latency" Cloud-RCS-FP32-Compat
bash scripts/system/cloud_stability.sh "$PRITHVI_WORKSPACE/cloud-entry" "$PRITHVI_WORKSPACE/cloud-stability-fp16" "$PRITHVI_SHIM" "$CPU" "$PRITHVI_REPO/src/system/telemetry.py" "$PRITHVI_WORKSPACE/cloud-latency" Cloud-RCS-FP16-Opt-v2
```

`cloud_auxiliary.sh` includes VRAM, five energy trials, ten startup trials, five kill/reload trials and stability. Execute it for both accepted roles when a complete resource matrix is needed. `cloud_stability.sh` provides a separate full sustained run when stability is measured on a different clean node. Its measured-time gate remains at least 3600 seconds; the conservative iteration budget and 5000 warm-ups follow the retained procedure. A pilot estimate alone is not proof of sufficient duration.

Use `src/system/cloud_summary.py` functions `latency_summary`, `provider_summary` and `aux_summary` for scoped evidence inspection, or its CLI for the complete two-node matrix:

```bash
python scripts/system_experiments.py cloud-summary --help
```

The full CLI requires `--node NAME=LATENCY_ROOT`, `--aux NAME=AUX_ROOT`, a genuine `--kernel-gate`, `--compile-gate`, and a new `--output-root`. It enforces two nodes × three latency roles, two accepted auxiliary roles, and valid resource gates. Do not fabricate the gate JSONs for partial runs. Mixed precision remains rejected by the task agreement criterion even if its latency succeeds.

## Measurement interpretation

- Flood paper energy is gross Joules per completed call. Cloud energy is incremental above idle per tile. Report both scopes explicitly.
- The retained cloud aggregator integrates the active telemetry sequence around the container procedure and divides by 1000 measured calls; its sequence can include initialization and warm-up. Preserve and report that interval, inspect it against the runner timestamps, and do not silently present it as pure steady-state model-call energy. The retained flood aggregator instead clips/interpolates to the runner's exact measured window.
- Integrate power only across the defined measurement interval. Samples must bracket that interval; check cadence, units, missing readings and cgroup ownership.
- Host RAM during compilation is not inference VRAM. Telemetry peaks and process RSS are different measurements.
- A one-hour sustained loop includes checks, telemetry and application overhead; its throughput is not the controlled model-call benchmark.
- Fault-injection and shell orchestration are Linux/container procedures. On an unsupported workstation, report BLOCKED with missing requirements rather than simulating a hardware PASS.
