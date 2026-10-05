#!/usr/bin/env python3
'Flood process-cold initialization and first-result measurement.'
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import flood_runner as bundle

EXPECTED={"size_bytes":1204352,"sha256":"7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80"}
def arrhash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def event(out,stage):
 row={"stage":stage,"unix_ns":time.time_ns(),"monotonic_ns":time.monotonic_ns()};(out/"stage.txt").write_text(stage+"\n",encoding="utf8")
 with (out/"stage_events.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(row)+"\n")
 return row
def main():
 p=argparse.ArgumentParser();p.add_argument("--label",required=True);p.add_argument("--config",type=Path,required=True);p.add_argument("--input",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 process_start=time.perf_counter();config_id_start=time.perf_counter();config_id=bundle.identity(a.config);input_id=bundle.identity(a.input);identity_gate_s=time.perf_counter()-config_id_start
 if {k:input_id[k] for k in EXPECTED}!=EXPECTED:raise RuntimeError("input identity drift")
 raw=np.load(a.input,allow_pickle=False)
 if raw.dtype!=np.float32 or raw.shape!=(1,6,224,224) or not np.isfinite(raw).all():raise RuntimeError("input contract")
 raw=np.ascontiguousarray(raw);a.output_dir.mkdir(parents=True);event(a.output_dir,"process_started")
 original=bundle.verify_identity;records=[]
 def measured(row,label):
  started=time.perf_counter();value=original(row,label);records.append({"label":label,"path":str(value),"seconds":time.perf_counter()-started,"size_bytes":row["size_bytes"],"sha256":row["sha256"]});return value
 bundle.verify_identity=measured
 try:runner=bundle.K100BundleRunner(a.config,device=0)
 finally:bundle.verify_identity=original
 event(a.output_dir,"sessions_loaded");logits,pred,first_ms=runner.run(raw);first_event=event(a.output_dir,"first_inference")
 if not np.isfinite(logits).all():raise RuntimeError("nonfinite")
 process_ttfr=time.perf_counter()-process_start;hash_seconds=sum(x["seconds"] for x in records);cache_hash=sum(x["seconds"] for x in records if x["label"].startswith("cache"));model_hash=sum(x["seconds"] for x in records if x["label"].startswith("model"))
 result={"schema":"journal_phase6_process_cold_start_v1","status":"PASSED","hardware":"海光 K100 AI 加速卡","variant":a.label,"config":config_id,"input":input_id,"data_role":"fixed configuration input identified by SHA256","cold_start_definition":"process cold start; filesystem page cache was not flushed",
  "runtime":{**runner.runtime,"load_seconds":runner.load_seconds,"first_inference_ms":first_ms},"timing":{"config_and_input_identity_gate_seconds":identity_gate_s,"artifact_hash_verification_seconds":hash_seconds,"model_hash_verification_seconds":model_hash,"mxr_hash_verification_seconds":cache_hash,"runtime_plus_cached_session_initialization_seconds":runner.load_seconds-hash_seconds,"mxr_load_separate_measurement":"not separable from ORT/MIGraphX cached Session initialization","process_time_to_first_result_seconds":process_ttfr,"first_result_unix_ns":first_event["unix_ns"]},
  "artifact_identity_records":records,"baseline":{"logits_sha256":arrhash(logits),"prediction_sha256":arrhash(pred)},"gates":{"finite":True,"cpu_fallback_disabled":runner.runtime["cpu_fallback_disabled"]}}
 (a.output_dir/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8");print(json.dumps({"status":"PASSED","variant":a.label,"ttfr_s":process_ttfr}),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
