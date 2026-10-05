#!/usr/bin/env python3
'Flood sustained inference with synchronized telemetry.'
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import argparse, json, subprocess, sys, time
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=3600.0)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--sampler", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started_ns = time.time_ns()
    stop = args.output_dir / "sampler.stop"
    telemetry = args.output_dir / "telemetry.csv"
    sampler = subprocess.Popen([sys.executable, str(args.sampler), "--stop", str(stop), "--output", str(telemetry), "--interval", "5"])
    command = [
        sys.executable, str(args.runner), "--mode", "stability", "--label", args.label,
        "--config", str(args.config), "--input", str(args.input), "--output-dir", str(args.output_dir / "run"),
        "--duration-seconds", str(args.duration_seconds),
    ]
    runner_started_ns = runner_ended_ns = None
    runner_rc = 1
    failure = None
    try:
        time.sleep(5)
        if sampler.poll() is not None:
            raise RuntimeError("Telemetry sampler exited before inference")
        runner_started_ns = time.time_ns()
        result = subprocess.run(command)
        runner_rc = result.returncode
        runner_ended_ns = time.time_ns()
        time.sleep(5)
    except BaseException as error:
        failure = repr(error)
    finally:
        stop.touch()
        try:
            sampler_rc = sampler.wait(timeout=30)
        except subprocess.TimeoutExpired:
            sampler.terminate()
            try:
                sampler_rc = sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler_rc = sampler.wait()
    payload = {
        "schema": "journal_phase6_integrated_stability_wrapper_v1", "status": "PASSED" if runner_rc == 0 and sampler_rc == 0 and failure is None else "FAILED",
        "hardware": "海光 K100 AI 加速卡", "variant": args.label, "requested_duration_seconds": args.duration_seconds,
        "wrapper_started_unix_ns": started_ns, "runner_started_unix_ns": runner_started_ns,
        "runner_ended_unix_ns": runner_ended_ns, "wrapper_ended_unix_ns": time.time_ns(),
        "runner_exit_code": runner_rc, "sampler_exit_code": sampler_rc, "failure": failure,
        "telemetry_interval_seconds": 5, 
    }
    (args.output_dir / "wrapper_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if payload["status"] == "PASSED" else 1

if __name__ == "__main__": raise SystemExit(main())
