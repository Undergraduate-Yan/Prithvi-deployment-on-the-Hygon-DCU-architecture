"""Run the unmodified frozen benchmark and record its process peak RSS."""
from pathlib import Path
import json,os,resource,subprocess,sys,time
os.sched_setaffinity(0,{2})
out=Path(sys.argv[1]);start=time.time();p=subprocess.run(sys.argv[2:])
out.write_text(json.dumps({'returncode':p.returncode,'max_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'scope':'unmodified benchmark child process including startup','started_unix':start,'ended_unix':time.time()}))
sys.exit(p.returncode)
