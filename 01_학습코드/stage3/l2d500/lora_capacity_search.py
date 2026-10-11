"""Find the largest tested independent sequence batch near 40GiB; isolated OOM recovery."""
from pathlib import Path
import subprocess,sys,json,time
R=Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
results=[]
def trial(b,m=32):
 tag=f'b{b}_m{m}'
 with open(R/f'lora_capacity_{tag}.log','w') as log:
  p=subprocess.run([sys.executable,'-u',str(R/'lora_capacity_probe.py'),str(b),str(m)],stdout=log,stderr=subprocess.STDOUT)
 file=R/f'lora_capacity_{tag}.json'
 entry={'batch':b,'clip_microbatch':m,'exit_code':p.returncode,'time':time.time()}
 if p.returncode==0 and file.exists():entry.update(json.loads(file.read_text())['measurements'][-1])
 else:entry['error_tail']=(R/f'lora_capacity_{tag}.log').read_text()[-1800:]
 results.append(entry);(R/'lora_capacity_search.json').write_text(json.dumps({'phase':'testing','results':results},indent=2));print(entry,flush=True)
 return entry
for b in [4,8,10,12]:
 e=trial(b)
 if e['exit_code']!=0 or e.get('peak_reserved_GiB',40)>38.5:
  lower=max([x['batch'] for x in results if x['exit_code']==0 and x.get('peak_reserved_GiB',40)<=38.5],default=2)
  for n in range(b-1,lower,-1):
   e=trial(n)
   if e['exit_code']==0 and e.get('peak_reserved_GiB',40)<=38.5:break
  break
valid=[x for x in results if x['exit_code']==0 and x.get('peak_reserved_GiB',40)<=38.5]
selected=max(valid,key=lambda x:x['batch']) if valid else None
(R/'lora_capacity_search.json').write_text(json.dumps({'phase':'complete','results':results,'selected':selected,'note':'Transient AdamW step included, no trained checkpoint saved. Effective48 uses variable final microbatch and loss weighting by sequence count. Training augmentation validation remains required.'},indent=2))
