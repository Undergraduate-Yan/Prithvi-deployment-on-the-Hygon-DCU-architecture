from pathlib import Path
import csv,math
import numpy as np
r=Path(__file__).resolve().parent
def rows(name):return list(csv.DictReader((r/name).open(encoding="utf-8-sig",newline="")))
event={(x["variant"],x["event_id"]):x for x in rows("per_event_metrics.csv")}
expected={(x["variant"],x["metric"]):x for x in rows("paired_event_bootstrap.csv")}
groups=sorted(set(x[1] for x in event));rng=np.random.RandomState(42);cache={};checks=0
for v in ["RCS13_FP16_OPT","MP_RCS_OPT"]:
 for m in ["miou","water_iou","boundary_water_iou","pixel_accuracy","agreement"]:
  pairs=[(g,float(event[v,g][m])-float(event["RCS13_FP32",g][m])) for g in groups]
  key=tuple(g for g,d in pairs if math.isfinite(d));values=np.array([d for g,d in pairs if math.isfinite(d)])
  if key not in cache:cache[key]=rng.randint(0,len(key),size=(10000,len(key)))
  means=values[cache[key]].mean(axis=1);target=expected[v,m]
  for name,value in [("mean_difference",values.mean()),("ci95_low",np.percentile(means,2.5)),("ci95_high",np.percentile(means,97.5)),("one_sided95_lcb",np.percentile(means,5))]:
   if abs(float(target[name])-value)>1e-12:raise RuntimeError("Reproduction mismatch: "+v+" "+m+" "+name)
   checks+=1
print("Reproduced",checks,"event-weighted statistics from",len(groups),"groups; no inference or raw labels read.")
