"""Resumable overnight coordinator, no Drive unmount while logs are open."""
import json,time,subprocess,sys,fcntl,traceback
from pathlib import Path
from diverse_v2_data import write_json,prepare_nuscenes
from diverse_v2_models import smoke

ROOT=Path('/content/MyDrive/MyDrive');OUT=ROOT/'stage3_vjepa_runs/diverse_v2_20260917'
CODE=Path(__file__).parent;OUT.mkdir(parents=True,exist_ok=True)
LOCK=open('/content/diverse_v2_job.lock','w')
fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)

def wait_for(paths):
    started=time.time();last=0
    while not all(p.exists() and json.loads(p.read_text()).get('verified') for p in paths):
        if time.time()-last>120:
            missing=[p.name for p in paths if not p.exists()]
            write_json(OUT/'status.json',{'phase':'waiting_for_verified_collection','missing':missing,'time':time.time()})
            print('V2_WAITING_FOR_COLLECTION',missing,flush=True);last=time.time()
        assert time.time()-started<12*3600,'Collection exceeded 12 hours; inspect download logs'
        time.sleep(30)

try:
    smoke()
    source=ROOT/'stage3_data/diverse_v2_20260917'
    wait_for([source/'collection_report.json'])
    cfg=json.loads((ROOT/'stage3_vjepa_runs/three_heads_20260917/config.json').read_text())
    work=Path('/content/diverse_v2_work');work.mkdir(exist_ok=True)
    if not (OUT/'nuscenes_rows.json').exists():
        write_json(OUT/'status.json',{'phase':'preparing_nuscenes_while_a2d2_downloads','time':time.time()})
        rows=prepare_nuscenes(ROOT,OUT,work,cfg);write_json(OUT/'nuscenes_rows.json',rows)
        print('V2_NUSCENES_ALIGNED_RECORDS',len(rows),flush=True)
    wait_for([source/'a2d2_collection_report.json',source/'20190401121727_stream_report.json',source/'20190401145936_stream_report.json'])
    for name in ['diverse_v2_train.py','diverse_v2_finish.py']:
        subprocess.run([sys.executable,'-u',str(CODE/name),'--root',str(ROOT)],cwd=CODE,check=True)
except Exception:
    err=traceback.format_exc();print(err,flush=True);write_json(OUT/'failure.json',{'error':err,'time':time.time()});raise
