#!/usr/bin/env python3
'Summarize flood vram measurements from raw records.'
from __future__ import annotations
import argparse
import csv, hashlib, json, statistics
from pathlib import Path

VARIANTS=("Mono_FP32","Mono_FP16_Opt","MP_RCS_Opt")

def identity(path): return {"path":str(path),"size_bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path, action='append')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--node', required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    roots = [p.resolve(strict=True) for p in args.input_root]
    if len(roots) != 1: raise ValueError("one input root required")
    root = roots[0]
    phase_root = args.output_dir
    if (root/"RUN_STATUS").read_text(encoding="utf8").strip()!="PASSED":raise RuntimeError("remote VRAM run not passed")
    rows=[];audit=[]
    for variant in VARIANTS:
        d=root/variant; samples_path=d/"vram_samples.csv"; events_path=d/"run/stage_events.jsonl"; result_path=d/"run/result.json"
        with samples_path.open(newline="",encoding="utf8") as f:samples=[{**r,"unix_ns":int(r["unix_ns"]),"vram_used_bytes":int(r["vram_used_bytes"])} for r in csv.DictReader(f)]
        events={r["stage"]:int(r["unix_ns"]) for r in (json.loads(line) for line in events_path.read_text(encoding="utf8").splitlines())}
        needed={"process_started","sessions_loaded","first_inference","steady_state","complete"}
        if set(events)!=needed:raise RuntimeError(f"stage drift {variant}: {set(events)}")
        launch=int((d/"container_launch_unix_ns.txt").read_text());exit_ns=int((d/"container_exit_unix_ns.txt").read_text())
        idle=[s for s in samples if s["unix_ns"]<launch]; post=[s for s in samples if s["unix_ns"]>=exit_ns]
        if not idle or not post:raise RuntimeError(f"idle/post samples missing: {variant}")
        idle_median=statistics.median(s["vram_used_bytes"] for s in idle)
        windows={
            "process_started_to_sessions_loaded":(events["process_started"],events["sessions_loaded"]),
            "sessions_loaded_to_first_inference":(events["sessions_loaded"],events["first_inference"]),
            "post_first_inference_snapshot":(events["first_inference"],events["steady_state"]),
            "steady_state":(events["steady_state"],events["complete"]),
        }
        for stage,(start,end) in windows.items():
            values=[s for s in samples if start<=s["unix_ns"]<end]
            if not values:
                values=[min(samples,key=lambda s:abs(s["unix_ns"]-start))]
                alignment="nearest_20hz_sample"
            else:alignment="within_stage_window"
            used=[s["vram_used_bytes"] for s in values]
            rows.append({"variant":variant,"hardware":"海光 K100 AI 加速卡","stage":stage,"sample_count":len(values),
                         "median_vram_used_bytes":statistics.median(used),"max_vram_used_bytes":max(used),
                         "median_increment_vs_idle_bytes":statistics.median(used)-idle_median,"max_increment_vs_idle_bytes":max(used)-idle_median,
                         "idle_median_vram_used_bytes":idle_median,"vram_total_bytes":int(samples[0]["vram_total_bytes"]),"alignment":alignment})
        result=json.loads(result_path.read_text(encoding="utf8"))
        if result["status"]!="PASSED" or not all(result["gates"].values()):raise RuntimeError(f"runner gate failed: {variant}")
        pre_pids={s["kfd_pids"] for s in idle};post_pids={s["kfd_pids"] for s in post}
        container_id=(d/"container_id.txt").read_text(encoding="utf8").strip()
        active=[s for s in samples if events["process_started"]<=s["unix_ns"]<exit_ns]
        foreign=[]
        for sample in active:
            for pid,cgroup in json.loads(sample["kfd_cgroups_json"]).items():
                if container_id not in cgroup: foreign.append({"pid":pid,"cgroup":cgroup,"unix_ns":sample["unix_ns"]})
        if pre_pids!={""} or post_pids!={""} or foreign:raise RuntimeError(f"exclusive KFD cgroup gate failed: {variant} foreign={foreign[:3]}")
        audit.extend(identity(p) for p in (samples_path,events_path,result_path,d/"hysmi_before.txt",d/"hysmi_after.txt",d/"container_id.txt"))
    path=args.output_dir/"vram_results.csv"
    with path.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    payload={"schema":"journal_phase6_vram_summary_v1","status":"PASSED","sampling_hz":20,"rows":rows,
             "interpretation_boundary":"Measured stage-level device VRAM increments only; no weight/activation/workspace decomposition is inferred.",
             "exclusive_kfd_process_gate":True,"source_files":audit}
    (args.output_dir/"vram_summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":"PASSED","rows":len(rows)}));return 0
if __name__=="__main__":raise SystemExit(main())
