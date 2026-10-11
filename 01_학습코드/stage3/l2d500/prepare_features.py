"""Resumeable L2D front camera acquisition and temporally coherent feature views."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,random,io,traceback
from collections import defaultdict
import numpy as np
import requests
import torch
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
OUT=Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
WORK=Path('/content/l2d500_20260927');SOURCE=OUT.parent/'diverse_v3_20260917'
sys.path.insert(0,str(SOURCE/'code'))
from model import load_encoder,extract_sequence

def save(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def status(phase,**kw):
 save('status.json',dict(phase=phase,time=time.time(),**kw));print(phase,json.dumps(kw),flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def download(rel,revision):
 p=WORK/rel;p.parent.mkdir(parents=True,exist_ok=True);receipt=p.with_suffix('.verified.json')
 if p.exists() and receipt.exists() and json.loads(receipt.read_text())['bytes']==p.stat().st_size:return p
 for attempt in range(4):
  try:
   tmp=p.with_suffix('.partial')
   with requests.get(f'https://huggingface.co/datasets/yaak-ai/L2D-v3/resolve/{revision}/{rel}',stream=True,timeout=(30,120)) as r:
    r.raise_for_status()
    with open(tmp,'wb') as f:
     for b in r.iter_content(4*1024*1024):f.write(b)
    expected=r.headers.get('Content-Length')
    if expected:assert tmp.stat().st_size==int(expected)
   assert tmp.stat().st_size>1000;tmp.replace(p)
   receipt.write_text(json.dumps({'bytes':p.stat().st_size,'sha256':sha(p),'revision':revision}));return p
  except Exception:
   if attempt==3:raise
   time.sleep(5*(attempt+1))
def main():
 torch.set_num_threads(4)
 selected=json.loads((OUT/'selected_episodes.json').read_text());records=selected['records'];cfg=selected['config']
 assert len(records)==500
 revisionfile=OUT/'source_revision.json'
 if revisionfile.exists():revision=json.loads(revisionfile.read_text())['revision']
 else:
  q=requests.get('https://huggingface.co/api/datasets/yaak-ai/L2D-v3',timeout=60);q.raise_for_status();revision=q.json()['sha']
  save('source_revision.json',{'revision':revision,'metadata_sha256':{str(p.relative_to(WORK)):sha(p) for p in WORK.rglob('*.parquet')},'note':'Metadata downloaded from main before revision recorded; hashes retained.'})
 groups=defaultdict(list)
 for r in records:groups[r['video_path']].append(r)
 (OUT/'clips').mkdir(exist_ok=True);(OUT/'features').mkdir(exist_ok=True)
 encoder=None;done=0
 for rel,rs in groups.items():
  missing=[r for r in rs if not (OUT/'clips'/(r['id']+'.mp4')).exists()]
  if missing:
   status('downloading_camera_chunk',path=rel,episodes_complete=done,total=500)
   source=download(rel,revision)
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate:format=duration','-of','json',str(source)]))
   rate=probe['streams'][0]['avg_frame_rate'];a,b=map(float,rate.split('/'));assert abs(a/b-10)<.01,(rel,rate)
   for r in missing:
    target=OUT/'clips'/(r['id']+'.mp4');tmp=WORK/(r['id']+'.partial.mp4')
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(r['video_start']),'-i',str(source),'-frames:v',str(r['frames']),'-an','-c:v','libx264','-crf','17','-preset','fast',str(tmp)],check=True)
    check=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(tmp)]))
    assert int(check['streams'][0]['nb_read_frames'])==r['frames'],r['id']
    import shutil
    partial=target.with_suffix('.partial.mp4');shutil.copyfile(tmp,partial);assert sha(partial)==sha(tmp);partial.replace(target);tmp.unlink()
  for r in rs:
   for view in range(2 if r['split']=='train' else 1):
    dest=OUT/'features'/(r['id']+f'_v{view}.npz')
    if dest.exists():
     with np.load(dest) as z:assert z['features'].shape==(r['frames'],3072) and np.array_equal(z['accel'],r['accel'])
     continue
    status('extracting_features',episode=r['id'],view=view,episodes_complete=done,total=500)
    if encoder is None:encoder=load_encoder(SOURCE/'code/vendor/vjepa2',OUT.parent/'kitti_50gb_v1/vjepa21_vitb_official.pt',torch.device('cuda'))
    import cv2
    cap=cv2.VideoCapture(str(OUT/'clips'/(r['id']+'.mp4')));images=[]
    while True:
     ok,im=cap.read()
     if not ok:break
     images.append(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
    cap.release();assert len(images)==r['frames']
    rng=random.Random(int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16));flip=bool(view and rng.random()<.5)
    bright,contrast,color,blur=rng.uniform(.6,1.4),rng.uniform(.7,1.35),rng.uniform(.5,1.4),rng.uniform(0,1.1);quality=rng.randint(50,90)
    def load(i):
     im=images[i].copy()
     if view:
      if flip:im=ImageOps.mirror(im)
      im=ImageEnhance.Brightness(im).enhance(bright);im=ImageEnhance.Contrast(im).enhance(contrast);im=ImageEnhance.Color(im).enhance(color).filter(ImageFilter.GaussianBlur(blur))
      buf=io.BytesIO();im.save(buf,format='JPEG',quality=quality);buf.seek(0);im=Image.open(buf).convert('RGB')
     return im
    f=extract_sequence(load,len(images),encoder,cfg,torch.device('cuda'));assert np.isfinite(f).all()
    ss=np.asarray(r['steer']);ss=2-ss if flip else ss
    tmp=dest.with_suffix('.partial.npz');np.savez_compressed(tmp,features=f.astype(np.float16),accel=r['accel'],steer=ss);tmp.replace(dest)
    save(r['id']+f'_v{view}_augmentation.json',{'flip':flip,'brightness':bright,'contrast':contrast,'color':color,'blur':blur,'jpeg_quality':quality,'applied':bool(view),'shared_across_sequence':True})
    if done==0:
     preview=Image.new('RGB',(768,384));preview.paste(ImageOps.pad(images[len(images)//2],(384,384)),(0,0));preview.paste(ImageOps.pad(load(len(images)//2),(384,384)),(384,0));preview.save(OUT/f'camera_preview_v{view}.jpg')
    del images,f
   done+=1;status('episode_features_complete',episodes_complete=done,total=500)
 save('features_complete.json',{'episodes':done,'time':time.time(),'train_views':2,'val_views':1,'revision':revision})
 status('FEATURES_COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:
  save('feature_failure.json',{'time':time.time(),'traceback':traceback.format_exc()});status('FEATURES_FAILED');raise
