from pathlib import Path
import requests,json,time,hashlib,shutil,zipfile,tarfile,traceback,math,gc
from collections import Counter
ROOT=Path('/content/MyDrive/MyDrive'); OUT=ROOT/'stage3_data/diverse_v3_20260917'; OUT.mkdir(parents=True,exist_ok=True)
LOCAL=Path('/content/diverse_v3');LOCAL.mkdir(exist_ok=True)
def save(name,x):
 p=OUT/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def digest(p,alg='sha256'):
 h=hashlib.new(alg)
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fetch(url,p,size):
 if p.exists() and p.stat().st_size==size:return
 part=p.with_suffix(p.suffix+'.part')
 for attempt in range(5):
  try:
   off=part.stat().st_size if part.exists() else 0
   if off>=size:part.rename(part.with_name(part.name+'.rejected.'+str(time.time_ns())));off=0
   assert shutil.disk_usage(LOCAL).free>size-off+10_000_000_000
   with requests.get(url,headers={'Range':f'bytes={off}-'} if off else {},stream=True,timeout=(30,120)) as r:
    r.raise_for_status()
    if r.status_code==206:assert r.headers['Content-Range'].startswith(f'bytes {off}-')
    else:off=0
    last=time.time()
    with part.open('ab' if off else 'wb') as f:
     for b in r.iter_content(8*1024*1024):
      if b:f.write(b);off+=len(b)
      if time.time()-last>45:print('V3_DOWNLOAD',p.name,off,size,flush=True);last=time.time()
   assert part.stat().st_size==size
   part.replace(p);return
  except Exception:
   if attempt==4:raise
   time.sleep(3)
def copy_verified(p):
 assert sum(x.stat().st_size for x in OUT.iterdir() if x.is_file())+p.stat().st_size<60_000_000_000
 sha=digest(p);dest=OUT/p.name
 if not(dest.exists() and dest.stat().st_size==p.stat().st_size and digest(dest)==sha):
  tmp=dest.with_suffix(dest.suffix+'.copying');shutil.copyfile(p,tmp);assert digest(tmp)==sha;tmp.replace(dest)
 return {'file':p.name,'bytes':p.stat().st_size,'sha256':sha}
def run():
 # Existing results and splits are references only; no training or relabelling here.
 old=ROOT/'stage3_vjepa_runs/diverse_v2_20260917'
 assert json.loads((old/'status.json').read_text())['phase']=='COMPLETE_REMOTE_HASH_VERIFIED'
 save('baseline.json',{'model':'decoupled_ltc','user_reported_contest_score':.723,'prior_split_contract':json.loads((old/'split_contract.json').read_text()),'do_not_modify_existing_splits':True})
 api='https://huggingface.co/api/datasets/commaai/comma2k19'
 rr=requests.get(api,timeout=60);rr.raise_for_status();rev=rr.json()['sha']
 rr=requests.get(api+'/tree/'+rev+'/raw_data',timeout=60);rr.raise_for_status();hf={x['path']:x for x in rr.json()}
 jobs=[]
 for n,vehicle in [(1,'Toyota RAV4'),(3,'Honda Civic')]:
  info=hf[f'raw_data/Chunk_{n}.zip'];jobs.append({'dataset':'comma2k19','file':f'comma2k19_Chunk_{n}.zip','url':f'https://huggingface.co/datasets/commaai/comma2k19/resolve/{rev}/'+info['path'],'bytes':info['size'],'sha256':info['lfs']['oid'],'vehicle':vehicle,'revision':rev})
 base='https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0/'
 for n in [3,6,8,10]:
  name=f'v1.0-trainval{n:02d}_blobs_camera.tgz';r=requests.head(base+name,timeout=60);r.raise_for_status()
  jobs.append({'dataset':'nuscenes','file':name,'url':base+name,'bytes':int(r.headers['Content-Length']),'etag':r.headers['ETag'].strip(chr(34))})
 assert sum(j['bytes'] for j in jobs)<100_000_000_000
 save('collection_plan.json',{'jobs':jobs,'transfer_bytes':sum(j['bytes'] for j in jobs),'stored_budget_bytes':60_000_000_000,'scope':'collection only; alignment and labels must pass before training','diversity':'new dataset and two vehicle models; nuScenes additional cities/night/rain audited from descriptions, not assumed from archive number'})
 print('V3_PLAN',[(j['file'],j['bytes']) for j in jobs],flush=True)
 src=ROOT/'stage3_data/diverse_driving_20260916'
 tables={}
 with tarfile.open(src/'v1.0-trainval_meta.tgz','r|gz') as t:
  for m in t:
   bn=Path(m.name).name
   if bn in ['scene.json','sample.json','sample_data.json','log.json']:tables[bn[:-5]]=json.load(t.extractfile(m))
 scenes={s['token']:s for s in tables['scene']};logs={s['token']:s for s in tables['log']};samples={s['token']:s['scene_token'] for s in tables['sample']}
 frames={s['filename']:samples[s['sample_token']] for s in tables['sample_data'] if '/CAM_FRONT/' in s['filename']}
 del tables;gc.collect()
 with zipfile.ZipFile(src/'can_bus.zip') as z:can_names=set(Path(n).name for n in z.namelist())
 contract=json.loads((old/'split_contract.json').read_text());protected=set(contract['excluded_mini_logs'])|{k for k,v in contract['nuScenes_log_split'].items() if v!='train'}
 all_reports=[]
 for j in jobs:
  reportfile=j['file']+'.report.json'
  if (OUT/reportfile).exists():
   prior=json.loads((OUT/reportfile).read_text())
   if prior.get('verified') and (OUT/prior['file']).exists():all_reports.append(prior);continue
  save('status.json',{'phase':'downloading','job':j,'time':time.time()})
  p=LOCAL/j['file'];fetch(j['url'],p,j['bytes'])
  if j['dataset']=='comma2k19':
   assert digest(p)==j['sha256'],'Official HF LFS SHA256 mismatch'
   with zipfile.ZipFile(p) as z:
    assert z.testzip() is None
    names=z.namelist();videos=[n for n in names if n.endswith('/video.hevc')]
    routes=sorted({n.rsplit('/',2)[0] for n in videos})
    sensor_names=[n for n in names if any(k in n for k in ['car_speed','steering_angle','frame_times','gyro'])]
    assert videos and sensor_names
   report=copy_verified(p);report.update(dataset=j['dataset'],vehicle=j['vehicle'],routes=routes,segments=len(videos),sensor_entries=len(sensor_names),sensor_examples=sensor_names[:20],environment='California I-280 commute; weather/lighting not yet audited',alignment_validated=False,verified=True,source=j)
  else:
   etag=j['etag'];parts=int(etag.rsplit('-',1)[1]);partsize=math.ceil(j['bytes']/parts/(1024*1024))*1024*1024
   hs=[]
   with p.open('rb') as f:
    for b in iter(lambda:f.read(partsize),b''):hs.append(hashlib.md5(b).digest())
   calculated=hashlib.md5(b''.join(hs)).hexdigest()+'-'+str(len(hs));assert calculated==etag,('multipart ETag mismatch',calculated,etag)
   h=requests.head(j['url'],timeout=60);h.raise_for_status();assert h.headers['ETag'].strip(chr(34))==etag
   q=LOCAL/(j['file'].replace('_blobs_camera.tgz','_front_sensor_candidates.tar'))
   counts=Counter()
   with tarfile.open(p,'r|gz') as t,tarfile.open(q,'w') as w:
    for m in t:
     name=m.name.removeprefix('./');token=frames.get(name)
     if not token or not m.isfile():continue
     scene=scenes[token]
     if scene['name']+'_pose.json' not in can_names:continue
     # Preserve all images; protect old evaluation groups from later training.
     w.addfile(m,t.extractfile(m));counts[token]+=1
   assert counts
   rows=[]
   for token,count in counts.items():
    s=scenes[token];d=s['description'].lower();g=s['log_token'];rows.append({'scene':s['name'],'log_token':g,'location':logs[g]['location'],'description':s['description'],'night':'night' in d,'rain_mentioned':'rain' in d,'front_frames':count,'sensor_pose_present':True,'existing_split':contract['nuScenes_log_split'].get(g,'new'),'protected_from_training':g in protected})
   report=copy_verified(q);report.update(dataset='nuscenes',verified=True,source_etag_verified=True,source=j,scenes=rows,alignment_validated=False)
   # Delete only our disposable source archive after verified selected-copy preservation.
   assert p.resolve().parent==LOCAL.resolve();p.unlink()
  save(reportfile,report);all_reports.append(report);save('collection_report.json',all_reports)
  print('V3_VERIFIED',j['file'],report['bytes'],flush=True)
 summary={'datasets':['KITTI','A2D2','nuScenes','comma2k19'],'new_comma_vehicles':sorted({r['vehicle'] for r in all_reports if r['dataset']=='comma2k19'}),'new_comma_segments':sum(r['segments'] for r in all_reports if r['dataset']=='comma2k19'),'nuscenes_scene_inventory':[s for r in all_reports if r['dataset']=='nuscenes' for s in r['scenes']],'alignment_validated':False,'next_step':'timestamp/unit/sign and class coverage audit; keep current LTC and all prior holdouts fixed'}
 save('diversity_inventory.json',summary);save('status.json',{'phase':'COLLECTED_PENDING_FLUSH','time':time.time()});print('V3_COLLECTION_DONE',flush=True)
try:run()
except Exception:
 save('failure.json',{'error':traceback.format_exc(),'time':time.time()});raise
