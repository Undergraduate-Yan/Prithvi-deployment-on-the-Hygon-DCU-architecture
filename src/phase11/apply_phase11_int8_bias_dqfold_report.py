#!/usr/bin/env python3
"""Recreate a frozen Day-7 bias-DQ fold from its exact sidecar records."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, onnx
from onnx import numpy_helper
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--frozen-report",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--report",type=Path,required=True);a=ap.parse_args()
 frozen=json.loads(a.frozen_report.read_text());src=frozen["source_model"]
 if a.source.stat().st_size!=src["size_bytes"] or sha(a.source)!=src["sha256"]:raise RuntimeError("source drift")
 m=onnx.load(str(a.source),load_external_data=False);onnx.checker.check_model(m);init={x.name:x for x in m.graph.initializer};targets={x["node_name"]:x for x in frozen["mutation"]["fold_records"]};removed=[];added=[]
 for n in m.graph.node:
  if n.name not in targets:continue
  if n.op_type!="DequantizeLinear" or len(n.input)!=3 or len(n.output)!=1:raise RuntimeError("target signature drift")
  q=np.asarray(numpy_helper.to_array(init[n.input[0]]));s=np.asarray(numpy_helper.to_array(init[n.input[1]]));z=np.asarray(numpy_helper.to_array(init[n.input[2]]));value=np.ascontiguousarray((q.astype(np.int64)-z.astype(np.int64)).astype(np.float32)*s.astype(np.float32))
  rec=targets[n.name]
  if hashlib.sha256(value.tobytes()).hexdigest()!=rec["folded_fp32_bias"]["sha256"] or n.output[0]!=rec["output_initializer_name"]:raise RuntimeError("folded value drift")
  added.append(numpy_helper.from_array(value,name=n.output[0]));removed.append(n.name)
 if set(removed)!=set(targets):raise RuntimeError("target set drift")
 keep=[n for n in m.graph.node if n.name not in targets];del m.graph.node[:];m.graph.node.extend(keep);m.graph.initializer.extend(added);onnx.checker.check_model(m);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(a.output));re=onnx.load(str(a.output),load_external_data=False);onnx.checker.check_model(re)
 out={"status":"created_static_pass","source":{"size_bytes":a.source.stat().st_size,"sha256":sha(a.source)},"frozen_report":{"size_bytes":a.frozen_report.stat().st_size,"sha256":sha(a.frozen_report)},"removed_bias_dq":len(removed),"output":{"size_bytes":a.output.stat().st_size,"sha256":sha(a.output)},"claims":{"strict_migraphx":False,"task_accuracy_90":False,"performance":False}}
 a.report.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
