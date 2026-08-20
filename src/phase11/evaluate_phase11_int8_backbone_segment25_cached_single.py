#!/usr/bin/env python3
"""Run all 25 strict MIGraphX caches sequentially on the locked real sample."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

SOURCE=(368_047_490,"edd2c3e69b64986dc6d0b78fe32230f9a9f7eb0396c2d77fdcb6bf4c59506a7a")
REPORT=(21_342,"e43daaa5dd8914c3632d0548b3f41f3d8c7d3b28f7be0bf391949b686532bb1e")
PARITY=(14_138,"5c66f94a447a3d08fe0024686e2a0dc2d8e74f16d0f10951cc49e4ed81568a6f")
SAMPLE=(4_820_344,"4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62")
RAW_SHA="23bd7b08aa08352cfa52b0086c3a6aac18b8b0e6a63f3cdbcae7409c667aa62c"
LABELS=tuple(f"encoder_block_{i:02d}" for i in range(24))+("upernet_decoder_head",)
MGX="MIGraphXExecutionProvider";CPU="CPUExecutionProvider"


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for x in iter(lambda:f.read(8<<20),b""):h.update(x)
 return h.hexdigest()


def lock(path:Path,expected:tuple[int,str])->dict:
 path=path.resolve(strict=True);d={"path":str(path),"size_bytes":path.stat().st_size,"sha256":sha(path)}
 if (d["size_bytes"],d["sha256"])!=expected:raise RuntimeError(f"identity drift: {d}")
 return d


def load_image(path:Path)->np.ndarray:
 obj=torch.load(path,map_location="cpu",weights_only=False);found=[]
 def visit(v):
  if torch.is_tensor(v) and v.ndim==4 and tuple(v.shape[1:])==(6,224,224):found.append(v)
  elif isinstance(v,dict):
   for z in v.values():visit(z)
  elif isinstance(v,(list,tuple)):
   for z in v:visit(z)
 visit(obj);x=np.ascontiguousarray(found[0][:1].numpy(),dtype=np.float32)
 if hashlib.sha256(x.tobytes()).hexdigest()!=RAW_SHA:raise RuntimeError("raw input drift")
 return x


def opts(ort,strict=False,profile=None):
 o=ort.SessionOptions();o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL;o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;o.intra_op_num_threads=4;o.inter_op_num_threads=1
 if strict:o.add_session_config_entry("session.disable_cpu_ep_fallback","1")
 if profile is not None:o.enable_profiling=True;o.profile_file_prefix=str(profile)
 return o


def array_record(x):
 x=np.ascontiguousarray(x);return {"shape":list(x.shape),"dtype":str(x.dtype),"finite":bool(np.isfinite(x).all()),"sha256":hashlib.sha256(x.tobytes()).hexdigest()}


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
 result={"status":"failed","variant":"int8_backbone_compat_25_static_batch1_cached_host_staged_single","claims":{"all_25_segments_strict_migraphx_placement":False,"strict_end_to_end_numeric_admission":False,"task_accuracy_90":False,"device_resident_intersegment_io":False,"performance":False,"native_int8_kernel_verified":False}}
 try:
  source=a.root/"build/candidate.onnx";segments=a.root/"segment25_static_build_cpu";sample=a.root/"sample_and_logits.pt"
  ids={"source":lock(source,SOURCE),"segment_report":lock(segments/"segment_build_report.json",REPORT),"cpu_parity":lock(segments/"cpu_sequential_parity.json",PARITY),"sample":lock(sample,SAMPLE)}
  report=json.loads((segments/"segment_build_report.json").read_text());rows=report["segments"]
  if [r["label"] for r in rows]!=list(LABELS):raise RuntimeError("segment order drift")
  models=[];caches=[];frozen=[]
  for i,row in enumerate(rows):
   model=segments/"models"/Path(row["path"]).name;ids[row["label"]]=lock(model,(int(row["size_bytes"]),row["sha256"]));models.append(model)
   base=a.root/"segment25_static_cache54g_segment0" if i==0 else a.root/"segment25_static_remaining_caches"/f"segment_{i:02d}"
   result_path=base/"output/result.json";d=json.loads(result_path.read_text());cache=base/f"segment_{i:02d}.mxr"
   if d.get("target_segment_index")!=i or d.get("target_segment_label")!=LABELS[i]:raise RuntimeError(f"result lineage drift {i}")
   counts=d.get("profile",{}).get("provider_event_counts",{})
   if int(counts.get(MGX,0))<=0 or int(counts.get(CPU,0))!=0:raise RuntimeError(f"frozen placement failed {i}")
   cmeta=d["compiled_cache"];lock(cache,(int(cmeta["size_bytes"]),cmeta["sha256"]));caches.append(cache);frozen.append({"index":i,"result":{"path":str(result_path),"size_bytes":result_path.stat().st_size,"sha256":sha(result_path)},"cache":cmeta,"descriptive_comparison":d.get("comparison")})
  import onnxruntime as ort
  if ort.__version__!="1.19.2" or MGX not in ort.get_available_providers():raise RuntimeError("runtime drift")
  raw=load_image(sample);cpu=ort.InferenceSession(str(source),sess_options=opts(ort),providers=[CPU]);cpu_logits=np.asarray(cpu.run(["logits"],{"image":raw})[0]);del cpu;gc.collect()
  retained={};value=raw;runtime_rows=[];profiles=[]
  for i,(model,cache) in enumerate(zip(models,caches)):
   os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"]="0";os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"]="1";os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"]=str(cache.resolve());os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"]=str(cache.resolve())
   prefix=a.output_dir/f"segment_{i:02d}_cached_profile";started=perf_counter();session=ort.InferenceSession(str(model),sess_options=opts(ort,True,prefix),providers=[(MGX,{"device_id":0})]);session.disable_fallback();create=perf_counter()-started
   feeds={session.get_inputs()[0].name:value} if i<24 else {item.name:retained[item.name] for item in session.get_inputs()};out=session.get_outputs()[0].name;started=perf_counter();value=np.asarray(session.run([out],feeds)[0]);infer=perf_counter()-started
   profile=Path(session.end_profiling()).resolve(strict=True);events=json.loads(profile.read_text());counts=Counter(str(e["args"]["provider"]) for e in events if e.get("cat")=="Node" and e.get("args",{}).get("provider"))
   if counts[MGX]<=0 or counts[CPU]!=0:raise RuntimeError(f"live cached placement failed {i}: {counts}")
   if i in (5,11,17,23):retained[out]=value.copy()
   runtime_rows.append({"index":i,"label":LABELS[i],"session_creation_seconds":create,"inference_seconds_diagnostic":infer,"output":array_record(value),"provider_event_counts":dict(counts)})
   profiles.append({"index":i,"path":str(profile),"size_bytes":profile.stat().st_size,"sha256":sha(profile)});del session;gc.collect()
  logits=value;diff=np.abs(cpu_logits.astype(np.float64)-logits.astype(np.float64));p0=np.argmax(cpu_logits,1);p1=np.argmax(logits,1);comparison={"mae":float(diff.mean()),"max_abs":float(diff.max()),"rmse":float(np.sqrt(np.mean(diff*diff))),"p99_abs":float(np.percentile(diff,99)),"pixel_class_agreement":float(np.mean(p0==p1)),"changed_pixels":int(np.count_nonzero(p0!=p1))}
  gates={"all_finite":bool(np.isfinite(cpu_logits).all() and np.isfinite(logits).all()),"all_25_live_profiles_migraphx_positive_cpu_zero":len(runtime_rows)==25,"mae_le_1e_3":comparison["mae"]<=1e-3,"max_abs_le_5e_2":comparison["max_abs"]<=5e-2,"pixel_class_agreement_ge_99_9pct":comparison["pixel_class_agreement"]>=.999}
  paired=a.output_dir/"cpu_and_cached_migraphx_logits.npz";np.savez_compressed(paired,image=raw,cpu_logits=cpu_logits,migraphx_logits=logits)
  result.update({"identities":ids,"frozen_cache_evidence":frozen,"runtime":{"onnxruntime":ort.__version__,"intersegment_transport":"numpy_host_staging"},"segments":runtime_rows,"profiles":profiles,"cpu_logits":array_record(cpu_logits),"migraphx_logits":array_record(logits),"comparison":comparison,"gates":gates,"paired_logits":{"path":str(paired),"size_bytes":paired.stat().st_size,"sha256":sha(paired)},"status":"passed" if all(gates.values()) else "failed"});result["claims"]["all_25_segments_strict_migraphx_placement"]=gates["all_25_live_profiles_migraphx_positive_cpu_zero"];result["claims"]["strict_end_to_end_numeric_admission"]=all(gates.values())
 except Exception as e:result["failure"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
 path=a.output_dir/"result.json";path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n");print(json.dumps(result,indent=2,ensure_ascii=False));raise SystemExit(0 if result["status"]=="passed" else 2)


if __name__=="__main__":main()
