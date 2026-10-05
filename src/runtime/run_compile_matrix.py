#!/usr/bin/env python3
'Controlled compilation settings and memory-budget comparison.'

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


EXACT_IMAGE = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
GIB = 1 << 30
CONFIGS = {
    "C0": {
        "description": "frozen runtime baseline without GPU compile parallel limit",
        "model": "raw",
        "intra_op_threads": 4,
        "environment": {
            "ORT_MIGRAPHX_EXHAUSTIVE_TUNE": "0",
            "MIGRAPHX_TIME_PASSES": "1",
        },
    },
    "C1": {
        "description": "C0 plus GPU compile parallelism limited to one",
        "model": "raw",
        "intra_op_threads": 4,
        "environment": {
            "ORT_MIGRAPHX_EXHAUSTIVE_TUNE": "0",
            "MIGRAPHX_TIME_PASSES": "1",
            "MIGRAPHX_GPU_COMPILE_PARALLEL": "1",
        },
    },
    "C2": {
        "description": "C1 plus CPU thread and glibc allocator limits",
        "model": "raw",
        "intra_op_threads": 1,
        "environment": {
            "ORT_MIGRAPHX_EXHAUSTIVE_TUNE": "0",
            "MIGRAPHX_TIME_PASSES": "1",
            "MIGRAPHX_GPU_COMPILE_PARALLEL": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
        },
    },
    "C3": {
        "description": "C2 using the separately audited clean merged graph",
        "model": "clean",
        "intra_op_threads": 1,
        "environment": {
            "ORT_MIGRAPHX_EXHAUSTIVE_TUNE": "0",
            "MIGRAPHX_TIME_PASSES": "1",
            "MIGRAPHX_GPU_COMPILE_PARALLEL": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=True, **kwargs)


def docker_inspect(container: str, template: str) -> str:
    return run(["docker", "inspect", "--format", template, container], capture_output=True).stdout.strip()


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def gpu_busy_percent() -> int:
    candidates = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
    values = []
    for path in candidates:
        try:
            values.append(int(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            continue
    return max(values, default=0)


def read_csv_peak(path: Path, field: str, mode: str = "max") -> Optional[int]:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        values = [int(row[field]) for row in csv.DictReader(stream) if row.get(field)]
    if not values:
        return None
    return min(values) if mode == "min" else max(values)


def parse_time_v(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing"}
    pairs = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.strip().split(":", 1)
            pairs[key] = value.strip()
    maximum_rss_kib = pairs.get("Maximum resident set size (kbytes)")
    return {
        "status": "captured",
        "maximum_resident_set_size_bytes": int(maximum_rss_kib) * 1024 if maximum_rss_kib else None,
        "elapsed_wall_clock": pairs.get("Elapsed (wall clock) time (h:mm:ss or m:ss)"),
        "user_time_seconds": pairs.get("User time (seconds)"),
        "system_time_seconds": pairs.get("System time (seconds)"),
        "exit_status": pairs.get("Exit status"),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_preflight(args: argparse.Namespace) -> dict[str, object]:
    image_id = run(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], capture_output=True
    ).stdout.strip()
    if image_id != EXACT_IMAGE or args.image != EXACT_IMAGE:
        raise RuntimeError(f"exact image identity mismatch: {args.image} -> {image_id}")
    if args.host_memory_budget_gib != 48:
        raise RuntimeError("formal C0-C3 matrix budget is frozen at 48 GiB")
    if not Path("/usr/bin/time").is_file():
        raise RuntimeError("host GNU /usr/bin/time is required for the frozen resource record")
    available = mem_available_bytes()
    if available < 54 * GIB:
        raise RuntimeError(f"preflight MemAvailable {available / GIB:.3f} GiB is below 54 GiB")
    busy = gpu_busy_percent()
    if busy != 0:
        raise RuntimeError(f"accelerator is not idle: gpu_busy_percent={busy}")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "image_id": image_id,
        "mem_available_bytes": available,
        "gpu_busy_percent": busy,
        "cgroup_version": 1,
        "host_memory_budget_bytes": args.host_memory_budget_gib * GIB,
        "active_stop_threshold_bytes":
        (args.host_memory_budget_gib - args.safety_stop_margin_gib) * GIB,
        "safety_stop_margin_bytes": args.safety_stop_margin_gib * GIB,
        "container_hard_limit_bytes": args.container_hard_limit_gib * GIB,
        "formal_90_image_test_used": False,
    }


def run_one(config_id: str, args: argparse.Namespace) -> dict[str, object]:
    config = CONFIGS[config_id]
    preflight = verify_preflight(args)
    run_dir = args.output_root / config_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite prior attempt: {run_dir}")
    run_dir.mkdir(parents=True)
    model = args.raw_model if config["model"] == "raw" else args.clean_model
    model.resolve(strict=True)
    args.feeds.resolve(strict=True)
    args.shim.resolve(strict=True)
    container = f"phase2b-{config_id.lower()}-{args.run_id.lower()}"
    compile_output = run_dir / "compile_output"
    cache = run_dir / f"{config_id}.mxr"
    time_v = run_dir / "time_v.txt"
    command = [
        "docker", "create", "--name", container,
        "--entrypoint", "/usr/bin/time",
        "--ulimit", "stack=-1:-1",
        "--memory", f"{args.container_hard_limit_gib}g",
        "--memory-swap", f"{args.container_hard_limit_gib}g",
        "--pids-limit", "1024", "--network", "none",
        "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
        "--ipc=host", "--shm-size=8g",
        "-v", "/opt/hyhal:/opt/hyhal:ro", "-v", "/var/tmp:/var/tmp:rw",
        "-v", "/usr/bin/time:/usr/bin/time:ro",
        "-v", f"{args.shim}:/usr/bin/kmod:ro",
        "-w", str(args.tools_root),
    ]
    for key, value in config["environment"].items():
        command.extend(["-e", f"{key}={value}"])
    command.extend(
        [
            args.image, "-v", "-o", str(time_v), "/usr/bin/python3",
            str(args.tools_root / "compile_flood_session.py"),
            "--config-id", config_id, "--model", str(model), "--feeds", str(args.feeds),
            "--cache", str(cache), "--shim", str(args.shim),
            "--output-dir", str(compile_output),
            "--intra-op-threads", str(config["intra_op_threads"]),
        ]
    )
    write_json(
        run_dir / "attempt_manifest.json",
        {
            "schema": "journal_phase2b_host_attempt_v1",
            "status": "created",
            "run_id": args.run_id,
            "config_id": config_id,
            "description": config["description"],
            "hardware": "海光 K100 AI 加速卡",
            "formal_90_image_test_used": False,
            "preflight": preflight,
            "model": identity(model),
            "feeds": identity(args.feeds),
            "environment": config["environment"],
            "docker_create_argv": command,
        },
    )
    run(command, capture_output=True)
    started_monotonic = time.monotonic()
    try:
        try:
            run(["docker", "start", container], capture_output=True)
        except subprocess.CalledProcessError as exc:
            write_json(
                run_dir / "controller_start_failure.json",
                {
                    "schema": "journal_phase2b_controller_start_failure_v1",
                    "status": "controller_failed_before_compile",
                    "config_id": config_id,
                    "returncode": exc.returncode,
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                    "formal_90_image_test_used": False,
                },
            )
            raise
        host_pid = int(docker_inspect(container, "{{.State.Pid}}"))
        cgroup_csv = run_dir / "cgroup_memory.csv"
        rss_csv = run_dir / "process_tree_rss.csv"
        cgroup_collector = subprocess.Popen(
            [
                "/bin/bash", str(args.tools_root / "collect_cgroup_memory.sh"),
                "--container", container, "--output", str(cgroup_csv),
                "--budget-bytes",
                str((args.host_memory_budget_gib - args.safety_stop_margin_gib) * GIB),
                "--interval-seconds", str(args.sample_interval_seconds),
            ]
        )
        rss_collector = subprocess.Popen(
            [
                sys.executable, str(args.tools_root / "../benchmarking/collect_compile_memory.py"),
                "--root-pid", str(host_pid), "--output", str(rss_csv),
                "--interval-seconds", str(args.sample_interval_seconds),
            ]
        )
        timed_out = False
        while docker_inspect(container, "{{.State.Running}}") == "true":
            if time.monotonic() - started_monotonic >= args.timeout_seconds:
                timed_out = True
                (run_dir / "timeout.stop_reason").write_text("compile_timeout\n", encoding="utf-8")
                subprocess.run(["docker", "stop", "--time", "5", container], check=False)
                break
            time.sleep(5)
        exit_code = int(docker_inspect(container, "{{.State.ExitCode}}"))
        oom_killed = docker_inspect(container, "{{.State.OOMKilled}}") == "true"
        state_error = docker_inspect(container, "{{.State.Error}}")
        logs = subprocess.run(
            ["docker", "logs", container], capture_output=True, text=True, check=False
        )
        (run_dir / "container.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        for collector in (cgroup_collector, rss_collector):
            try:
                collector.wait(timeout=15)
            except subprocess.TimeoutExpired:
                collector.terminate()
                collector.wait(timeout=5)
        budget_stopped = cgroup_csv.with_suffix(cgroup_csv.suffix + ".stop_reason").exists()
        compile_result_path = compile_output / "result.json"
        compile_result = (
            json.loads(compile_result_path.read_text(encoding="utf-8"))
            if compile_result_path.exists()
            else None
        )
        summary = {
            "schema": "journal_phase2b_compile_probe_result_v1",
            "status": "passed"
            if exit_code == 0
            and not budget_stopped
            and not timed_out
            and compile_result is not None
            and compile_result.get("status") == "passed"
            else "failed",
            "run_id": args.run_id,
            "config_id": config_id,
            "description": config["description"],
            "hardware": "海光 K100 AI 加速卡",
            "formal_90_image_test_used": False,
            "preflight": preflight,
            "model": identity(model),
            "feeds": identity(args.feeds),
            "environment": config["environment"],
            "exit_code": exit_code,
            "oom_killed": oom_killed,
            "state_error": state_error,
            "budget_stopped": budget_stopped,
            "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started_monotonic,
            "host_memory_budget_bytes": args.host_memory_budget_gib * GIB,
            "active_stop_threshold_bytes":
            (args.host_memory_budget_gib - args.safety_stop_margin_gib) * GIB,
            "safety_stop_margin_bytes": args.safety_stop_margin_gib * GIB,
            "container_hard_limit_bytes": args.container_hard_limit_gib * GIB,
            "peak_cgroup_memory_bytes": read_csv_peak(cgroup_csv, "memory_max_usage_bytes"),
            "peak_process_tree_rss_bytes": read_csv_peak(rss_csv, "process_tree_rss_bytes"),
            "minimum_mem_available_bytes": read_csv_peak(cgroup_csv, "mem_available_bytes", "min"),
            "maximum_cgroup_failcnt": read_csv_peak(cgroup_csv, "memory_failcnt"),
            "time_v": parse_time_v(time_v),
            "mxr": identity(cache) if cache.is_file() else None,
            "compile_result": compile_result,
        }
        write_json(run_dir / "result.json", summary)
        return summary
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--raw-model", type=Path, required=True)
    parser.add_argument("--clean-model", type=Path, required=True)
    parser.add_argument("--feeds", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--configs", default="C0,C1,C2,C3")
    parser.add_argument("--image", default=EXACT_IMAGE)
    parser.add_argument("--host-memory-budget-gib", type=int, default=48)
    parser.add_argument("--safety-stop-margin-gib", type=int, default=1)
    parser.add_argument("--container-hard-limit-gib", type=int, default=49)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    args = parser.parse_args()
    requested = [item.strip() for item in args.configs.split(",") if item.strip()]
    if any(item not in CONFIGS for item in requested):
        parser.error(f"unknown config in {requested}")
    input_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if input_manifest["status"] != "prepared_not_compiled":
        raise RuntimeError("input manifest is not the frozen prepared probe")
    if input_manifest["selection_boundary"]["formal_90_image_test_used"] is not False:
        raise RuntimeError("formal test contamination detected")
    if args.container_hard_limit_gib <= args.host_memory_budget_gib:
        raise RuntimeError("hard limit must be above the active stop threshold")
    if not 0 < args.safety_stop_margin_gib < args.host_memory_budget_gib:
        raise RuntimeError("safety stop margin must be positive and below the formal budget")
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = Path("/var/tmp/prithvi-compile.lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another compilation controller holds the host lock") from exc
        results = []
        for config_id in requested:
            result = run_one(config_id, args)
            results.append(result)
            write_json(
                args.output_root / "matrix_status.json",
                {
                    "schema": "journal_phase2b_matrix_status_v1",
                    "status": "running" if config_id != requested[-1] else "completed",
                    "run_id": args.run_id,
                    "requested_configs": requested,
                    "completed_configs": [item["config_id"] for item in results],
                    "results": results,
                    "formal_90_image_test_used": False,
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
