"""Matched cloud resource and full-scene measurement worker; never loads labels."""
from pathlib import Path
import argparse,ctypes,csv,hashlib,json,os,resource,threading,time
import numpy as np
from PIL import Image

ROOT=Path('/var/tmp/rs_revision_20261112/system_control')
MEAN=np.asarray([.20592650340892424,.21923400858467046,.22272650518804932,.34347781061352434,.25785957500381157,.202518440999147],np.float32)[:,None,None]
STD=np.asarray([.26007981612252296,.24578255272067706,.24758331278577964,.20353990322067164,.15844321837668404,.1511271839787069],np.float32)[:,None,None]
STARTS=[0,224,448,672,896,1120,1344,1568,1776]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return time.perf_counter_ns()*1e-9
class Monitor:
 def __init__(self):self.rows=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True)
 def loop(self):
  os.sched_setaffinity(threading.get_native_id(),{3});tick=0
  while not self.stop.is_set():
   start=now();mem={}
   for l in Path('/proc/self/status').read_text().splitlines():
    if l.startswith(('VmRSS:','VmHWM:')):k,v=l.split(':');mem[k]=int(v.split()[0])*1024
   row={'time_s':start,'unix_s':time.time(),'device_bytes':int(Path('/sys/class/drm/card1/device/mem_info_vram_used').read_text()),'rss_bytes':mem.get('VmRSS',0),'hwm_bytes':mem.get('VmHWM',0),'power_w':'','temperature_c':''}
   if tick%2==0:
    row['power_w']=int(Path('/sys/class/drm/card1/device/hwmon/hwmon2/power1_average').read_text())*1e-6
    row['temperature_c']=int(Path('/sys/class/drm/card1/device/hwmon/hwmon2/temp1_input').read_text())*.001
   self.rows.append(row);tick+=1;self.stop.wait(max(0,.05-(now()-start)))
 def energy(self,a,b,idle):
  rows=[r for r in self.rows if r['power_w']!=''];x=np.array([r['time_s'] for r in rows]);y=np.array([r['power_w'] for r in rows]);ix=(x>a)&(x<b)
  assert len(x)>1 and x[0]<=a and x[-1]>=b
  xx=np.r_[a,x[ix],b];yy=np.interp(xx,x,y)
  gross=float(np.trapz(yy,xx));return gross,gross-idle*(b-a)
 def peak(self,a,b):
  rows=[r for r in self.rows if a<=r['time_s']<=b]
  return max([r['device_bytes'] for r in rows] or [0]),max([r['rss_bytes'] for r in rows] or [0])

def main():
 p=argparse.ArgumentParser();p.add_argument('--precision',choices=['fp32','fp16'],required=True);p.add_argument('--run',required=True);p.add_argument('--seconds',type=float,default=60);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 os.sched_setaffinity(0,{2});out=ROOT/'results'/a.run;out.mkdir(parents=True,exist_ok=False)
 mon=Monitor();mon.thread.start();idle_a=now();time.sleep(1 if a.smoke else 5);idle_b=now()
 idle_rows=[r for r in mon.rows if idle_a<=r['time_s']<=idle_b]
 idle_power=float(np.mean([r['power_w'] for r in idle_rows if r['power_w']!='']));idle_mem=int(np.median([r['device_bytes'] for r in idle_rows]))
 lib=ctypes.CDLL(str(ROOT/'build/librs_scene_runner.so'));lib.rs_create.restype=ctypes.c_void_p;lib.rs_create.argtypes=[ctypes.c_char_p,ctypes.c_char_p];lib.rs_error.restype=ctypes.c_char_p
 fp=ctypes.POINTER(ctypes.c_float);lib.rs_run.argtypes=[ctypes.c_void_p,fp,fp];lib.rs_run.restype=ctypes.c_int;lib.rs_destroy.argtypes=[ctypes.c_void_p]
 load_a=now();handle=lib.rs_create(str(ROOT/'plans'/f'cloud_{a.precision}.tsv').encode(),b'');load_b=now()
 if not handle:raise RuntimeError(lib.rs_error().decode())
 x=np.fromfile(ROOT/'cloud_input.raw',np.float32).reshape(1,6,224,224);logits=np.empty((1,4,224,224),np.float32)
 def call(t):
  if lib.rs_run(handle,t.ctypes.data_as(fp),logits.ctypes.data_as(fp)):raise RuntimeError(lib.rs_error().decode())
  if not np.isfinite(logits).all():raise ValueError('Nonfinite output')
 for _ in range(2 if a.smoke else 50):call(x)
 logits.tofile(out/'smoke_tile_logits.f32')
 reference=np.argmax(logits,axis=1);calls=0;drift=0;tile_a=now()
 while now()-tile_a < (1 if a.smoke else a.seconds):
  call(x);calls+=1
  if calls%100==0:drift+=int(np.count_nonzero(np.argmax(logits,axis=1)!=reference))
 tile_b=now();scenes=json.loads((ROOT/'scene_list.json').read_text());scene_rows=[]
 for item in scenes[:1] if a.smoke else scenes:
  sid=item['scene_index'];start=now()
  with np.load(item['path'],allow_pickle=False) as z: image=z['image']
  read_end=now();assert image.shape==(6,2000,2000) and image.dtype==np.float32
  normalized=np.ascontiguousarray((image-MEAN)/STD,np.float32);normalize_end=now()
  accumulator=np.zeros((4,2000,2000),np.float32);weight=np.zeros((2000,2000),np.uint8);tile_s=0.;stitch_s=now()-normalize_end
  for y in STARTS:
   for xx in STARTS:
    t0=now();tile=np.ascontiguousarray(normalized[:,y:y+224,xx:xx+224][None]);tile_s+=now()-t0
    t0=now();call(tile);tile_s+=now()-t0
    t0=now();accumulator[:,y:y+224,xx:xx+224]+=logits[0];weight[y:y+224,xx:xx+224]+=1;stitch_s+=now()-t0
  assert weight.min()==1 and weight.max()==4
  t0=now();accumulator/=weight[None];stitch_s+=now()-t0
  product_a=now();mask=np.argmax(accumulator,axis=0).astype(np.uint8)
  product=out/f'scene_{sid:03d}_mask.png';Image.fromarray(mask).save(product)
  dump(out/f'scene_{sid:03d}_product.json',{'source':item['path'],'source_sha256':item['sha256'],'shape':[2000,2000],'classes':['clear sky','thick cloud','thin cloud','cloud shadow'],'georeferencing':'not available in retained NPZ; no geographic coordinates invented','label_loaded':False,'precision':a.precision})
  end=now();row={'scene_index':sid,'start_s':start,'end_s':end,'read_s':read_end-start,'normalize_s':normalize_end-read_end,'tile_calls':81,'tile_processing_s':tile_s,'stitch_s':stitch_s,'product_s':end-product_a,'total_s':end-start,'product_path':str(product),'product_sha256':sha(product),'source_sha256':item['sha256']};scene_rows.append(row)
  del image,normalized,accumulator,weight,mask,tile
 time.sleep(.25);end=now();mon.stop.set();mon.thread.join()
 tile_gross,tile_net=mon.energy(tile_a,tile_b,idle_power);tile_peak,tile_rss=mon.peak(tile_a,tile_b)
 for row in scene_rows:
  gross,net=mon.energy(row['start_s'],row['end_s'],idle_power);peak,rss=mon.peak(row['start_s'],row['end_s'])
  row.update({'gross_device_energy_j':gross,'above_idle_device_energy_j':net,'device_peak_bytes':peak,'device_peak_increment_bytes':peak-idle_mem,'host_peak_rss_bytes':rss})
 with (out/'telemetry.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(mon.rows[0]));w.writeheader();w.writerows(mon.rows)
 with (out/'scenes.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(scene_rows[0]));w.writeheader();w.writerows(scene_rows)
 result={'precision':a.precision,'run':a.run,'smoke':a.smoke,'status':'passed','idle_power_w':idle_power,'idle_device_bytes':idle_mem,'load_s':load_b-load_a,'tile_calls':calls,'tile_duration_s':tile_b-tile_a,'tile_gross_device_energy_j_per_call':tile_gross/calls,'tile_above_idle_device_energy_j_per_call':tile_net/calls,'tile_device_peak_bytes':tile_peak,'tile_device_peak_increment_bytes':tile_peak-idle_mem,'tile_host_peak_rss_bytes':tile_rss,'scene_count':len(scene_rows),'tile_prediction_drift_pixels_checked':drift,'host_VmHWM_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'memory_sample_hz':20,'power_sample_hz':10,'whole_computer_energy_measured':False,'labels_loaded':False,'original_files_modified':False}
 assert drift==0;dump(out/'summary.json',result);lib.rs_destroy(handle);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
