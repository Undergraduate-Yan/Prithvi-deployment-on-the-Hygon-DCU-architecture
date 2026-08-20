#!/usr/bin/env python3
"""Create a verbose ORT/MIGraphX session to expose full-graph rejection reasons."""
import argparse
from pathlib import Path
import onnxruntime as ort

ort.set_default_logger_severity(0)

p = argparse.ArgumentParser()
p.add_argument("model", type=Path)
p.add_argument("--strict", action="store_true")
a = p.parse_args()
so = ort.SessionOptions()
so.log_severity_level = 0
so.log_verbosity_level = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
if a.strict:
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
providers = [("MIGraphXExecutionProvider", {"device_id": 0})]
if not a.strict:
    providers.append("CPUExecutionProvider")
print("available", ort.get_available_providers(), flush=True)
print("requested", providers, flush=True)
s = ort.InferenceSession(str(a.model), sess_options=so, providers=providers)
print("created", s.get_providers(), flush=True)
