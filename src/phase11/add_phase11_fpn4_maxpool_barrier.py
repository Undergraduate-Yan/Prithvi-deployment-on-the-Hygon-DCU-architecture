#!/usr/bin/env python3
"""Add the single validated fpn4 MaxPool graph-output barrier."""
import argparse, hashlib, json
from pathlib import Path
import onnx
from onnx import TensorProto, helper
TENSOR="/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0";NODE="/task/model/decoder/fpn4/fpn4.0/MaxPool"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("source",type=Path);ap.add_argument("output",type=Path);ap.add_argument("report",type=Path);a=ap.parse_args();m=onnx.load(str(a.source),load_external_data=False)
 hits=[n for n in m.graph.node if n.name==NODE and TENSOR in n.output]
 if len(hits)!=1 or any(x.name==TENSOR for x in m.graph.output):raise RuntimeError("barrier identity drift")
 m.graph.output.append(helper.make_tensor_value_info(TENSOR,TensorProto.FLOAT,[1,1024,7,7]));onnx.checker.check_model(m);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(a.output));re=onnx.load(str(a.output),load_external_data=False);onnx.checker.check_model(re)
 out={"status":"created_static_pass","source":{"size_bytes":a.source.stat().st_size,"sha256":sha(a.source)},"output":{"size_bytes":a.output.stat().st_size,"sha256":sha(a.output),"outputs":[x.name for x in re.graph.output]},"barrier":{"node":NODE,"tensor":TENSOR}}
 a.report.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
