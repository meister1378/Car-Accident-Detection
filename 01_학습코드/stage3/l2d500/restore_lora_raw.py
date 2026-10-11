"""Restore only immutable original train/val manifest images, without relabeling."""
from pathlib import Path
import json,tarfile,zipfile,hashlib,shutil,traceback,time
from collections import defaultdict,Counter
from PIL import Image,ImageOps
import cv2
ROOT=Path('/content/MyDrive/MyDrive'); OUT=ROOT/'stage3_vjepa_runs/l2d500_w256_b48_20260927'; DATA=ROOT/'stage3_data'
def save(name,obj):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def main():
 manifest=ROOT/'stage3_vjepa_runs/diverse_v3_20260917/manifest.json'
 rows=[r for r in json.loads(manifest.read_text())['records'] if r['split'] in ('train','val')]
 paths={d:{Path(p) for r in rows if r['domain']==d for p in r['images']} for d in {r['domain'] for r in rows}}
 def status(phase):
  counts={d:{'required':len(pp),'present':sum(p.exists() for p in pp)} for d,pp in paths.items()}
  save('lora_raw_status.json',{'phase':phase,'time':time.time(),'counts':counts,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()});print(phase,counts,flush=True)
 status('START')
 # The source preparation stored padded 384px images for A2D2 and nuScenes.
 for domain,archives in [('a2d2',list((DATA/'diverse_v2_20260917').glob('*selected.tar'))+list((DATA/'diverse_v2_20260917').glob('*frontcenter.tar'))),('nuscenes',list((DATA/'diverse_v3_20260917').glob('*candidates.tar'))+[DATA/'diverse_driving_20260916/v1.0-trainval01_blobs_camera.tgz',DATA/'diverse_v2_20260917/v1.0-trainval02_blobs_camera.tgz'])]:
  wanted=defaultdict(list)
  for p in paths.get(domain,set()):
   if not p.exists():wanted[p.name].append(p)
  for archive in archives:
   if not wanted:break
   status('READ '+archive.name)
   with tarfile.open(archive,'r|*') as tf:
    for member in tf:
     if not member.isfile() or Path(member.name).name not in wanted:continue
     targets=wanted[Path(member.name).name]
     # Prevent basename collisions by also requiring the archive suffix path.
     targets=[p for p in targets if str(p).endswith(member.name.lstrip('./')) or '/'+member.name.lstrip('./') in str(p)]
     if not targets:continue
     with tf.extractfile(member) as f,Image.open(f) as im:
      padded=ImageOps.pad(im.convert('RGB'),(384,384),color=(124,116,104))
      for p in targets:p.parent.mkdir(parents=True,exist_ok=True);padded.save(p)
     wanted[Path(member.name).name]=[p for p in wanted[Path(member.name).name] if not p.exists()]
     if not wanted[Path(member.name).name]:del wanted[Path(member.name).name]
   status('DONE '+archive.name)
 # KITTI retains its original pixel data.
 bydrive=defaultdict(list)
 for p in paths.get('kitti',set()):
  if not p.exists():bydrive[p.parts[-4]].append(p)
 for drive,targets in bydrive.items():
  candidates=list((DATA/'kitti_zips').glob(drive+'.zip'))+list((DATA/'kitti_extra50gb_20260916').glob(drive+'.zip'))
  if not candidates:continue
  with zipfile.ZipFile(candidates[0]) as z:
   names=set(z.namelist())
   for p in targets:
    name='/'.join(p.parts[-5:])
    if name not in names:continue
    p.parent.mkdir(parents=True,exist_ok=True)
    with z.open(name) as f,p.open('wb') as out:shutil.copyfileobj(f,out)
  status('KITTI '+drive)
 # comma source prefix hashes and frame indices are the original preparation contract.
 comma=defaultdict(set)
 for p in paths.get('comma2k19',set()):
  if not p.exists():comma[p.parent].add(int(p.stem))
 for archive in sorted((DATA/'diverse_v3_20260917').glob('comma2k19_Chunk_*.zip')):
  with zipfile.ZipFile(archive) as z:
   for name in z.namelist():
    if not name.endswith('/video.hevc'):continue
    token=hashlib.sha256(name.rsplit('/',1)[0].encode()).hexdigest()[:16]
    dirs=[p for p in comma if p.name==token]
    if not dirs:continue
    for dest in dirs:
     dest.mkdir(parents=True,exist_ok=True);video=dest/'video.hevc'
     if not video.exists():
      with z.open(name) as f,video.open('wb') as out:shutil.copyfileobj(f,out)
     cap=cv2.VideoCapture(str(video));i=0;wanted=comma[dest]
     while i<=max(wanted):
      ok,frame=cap.read()
      if not ok:break
      if i in wanted:
       im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));ImageOps.pad(im,(384,384),color=(124,116,104)).save(dest/f'{i:06d}.jpg',quality=92)
      i+=1
     cap.release();status('COMMA '+token)
 status('AUDIT')
 missing={d:[str(p) for p in pp if not p.exists()] for d,pp in paths.items()}
 save('lora_raw_missing.json',missing)
 assert not any(missing.values()),{d:len(v) for d,v in missing.items()}
 save('lora_raw_complete.json',{'time':time.time(),'records':len(rows),'frames':{d:len(pp) for d,pp in paths.items()},'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
if __name__=='__main__':
 try:main()
 except Exception:save('lora_raw_failure.json',{'traceback':traceback.format_exc()});raise
