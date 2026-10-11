"""Locked, resumable V3 audit -> extraction -> training -> checked submissions."""
import sys,subprocess,fcntl,traceback,time,shutil
from pathlib import Path
from diverse_v3_data import OUT,ROOT
from diverse_v2_data import write_json

OUT.mkdir(parents=True,exist_ok=True)
lock=open('/content/diverse_v3_training.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
code=Path(__file__).parent
try:
    shutil.copytree(code,OUT/'code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.zip','*.ipynb'))
    for script in ['diverse_v3_data.py','diverse_v3_train.py','diverse_v3_finish.py']:
        subprocess.run([sys.executable,'-u',str(code/script)],cwd=code,check=True)
except Exception:
    err=traceback.format_exc();write_json(OUT/'failure.json',{'time':time.time(),'error':err});print(err,flush=True);raise
