"""Wait for verified feature preparation, then train and package exactly once."""
from pathlib import Path
import fcntl,os,subprocess,sys,time,json,traceback
OUT=Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
lock=open('/content/l2d500_20260927/continuation.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def record(phase,**kw):
 obj=dict(phase=phase,time=time.time(),pid=os.getpid(),**kw);tmp=OUT/'continuation_status.tmp';tmp.write_text(json.dumps(obj,indent=2));tmp.replace(OUT/'continuation_status.json');print(json.dumps(obj),flush=True)
try:
 record('waiting_for_features')
 while not (OUT/'features_complete.json').exists():
  if (OUT/'feature_failure.json').exists():raise RuntimeError('Feature preparation failed; inspect feature_failure.json before retrying')
  time.sleep(30)
 assert (OUT/'camera_review.json').exists(),'Forward-camera visual verification missing'
 if not (OUT/'training_complete.json').exists():
  record('training');subprocess.run([sys.executable,'-u',str(OUT/'train_l2d.py')],check=True)
 if not (OUT/'submission/complete.json').exists():
  record('packaging');subprocess.run([sys.executable,'-u',str(OUT/'package_l2d.py')],check=True)
 record('FROZEN_SUBMISSION_COMPLETE_LORA_PENDING')
except Exception:
 record('FAILED',traceback=traceback.format_exc());raise
