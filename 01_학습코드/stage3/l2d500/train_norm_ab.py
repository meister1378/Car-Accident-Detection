"""48 independent sequences, rotating exact five-domain quotas; frozen encoder."""
from pathlib import Path
import sys,os,json,time,random,hashlib,importlib.util,concurrent.futures,traceback,gc
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from ltc_task_normalization import set_intermediate_normalization, MODES
DATA=Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
MODE=sys.argv[1];assert MODE in MODES
OUT=DATA/'norm_ab_w256_15ep'/MODE;RUNS=DATA.parent;SOURCE=RUNS/'diverse_v3_20260917'
sys.path.insert(0,str(SOURCE/'code'))
from diverse_v2_models import probabilities
spec=importlib.util.spec_from_file_location('prior_train',RUNS/'v3_w256_B_15ep_20260926/train_three_experiments.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
BATCH=48;T=256;EPOCHS=15;STEPS=300;SEED=927548;LR=5e-5
def save(name,obj):
 p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def status(phase,**kw):save('training_status.json',dict(phase=phase,time=time.time(),**kw));print(phase,json.dumps(kw),flush=True)
def pt(name,obj):
 p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp.pt');torch.save(obj,tmp);tmp.replace(p)
def main():
 assert (DATA/'features_complete.json').exists()
 if (OUT/'training_complete.json').exists():return
 torch.set_num_threads(4);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 old=json.loads((SOURCE/'manifest.json').read_text());new=json.loads((DATA/'selected_episodes.json').read_text());assert old['config']==new['config'];cfg=old['config']
 records=[r for r in old['records'] if r['split'] in ['train','val']]+new['records']
 train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val'];domains=sorted({r['domain'] for r in train});assert len(domains)==5
 assert not {(r['domain'],r['group']) for r in train}&{(r['domain'],r['group']) for r in val}
 base=DATA/'trained/head.pt';base_sha=prior.sha(base);ck=torch.load(base,map_location='cpu',weights_only=True)
 assert base_sha=='11069af0aa0e2941d57d74b9445d1940e3b72de5830588177a92c0859463113f'
 model=prior.make_wide().cuda();model.load_state_dict(ck['members'][0]['model'],strict=True)
 set_intermediate_normalization(model,MODE)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
 contract={'epochs':EPOCHS,'steps_per_epoch':STEPS,'batch':BATCH,'window_frames':T,'trainable_parameters':sum(p.numel() for p in model.parameters()),'warm_start_sha256':base_sha,'lr':LR,'end_lr':LR/10,'sampler':'five datasets: rotating 10/10/10/9/9 windows; 48 slots per dataset every5 updates. Per dataset acceleration class anchor probabilities .35/.35/.20/.10, random anchor from training frames. Report actual whole-window labels separately.','augmentation':'cached coherent image views, feature channel mask .10 with .5 probability, Gaussian feature noise .025. Time order unchanged.','selection':'original four-domain eight validation task MacroF1 mean; L2D heldout separate','limitations':'single-seed paired comparison; development validation reused; sensor-derived labels; no hidden-test accuracy claim; auxiliary two intermediate channels retain legacy joint normalization','seed':SEED,'intermediate_normalization':MODE,'manifest_sha256':prior.sha(SOURCE/'manifest.json'),'l2d_selection_sha256':prior.sha(DATA/'selected_episodes.json'),'normalization_code_sha256':prior.sha(Path(__file__).with_name('ltc_task_normalization.py')),'normalization_scope':'only channels0:4 and4:7 are normalized within their own groups;7:9 unchanged;same9channel architecture and loss'}
 save('training_contract.json',contract);status('loading_features')
 arrays={}
 def read(rv):
  r,v=rv;p=(DATA/'features'/(r['id']+f'_v{v}.npz')) if r['domain']=='l2d' else prior.cache(r,v)
  with np.load(p) as f:z={k:f[k].copy() for k in ['features','accel','steer']}
  assert z['features'].shape==(len(r['accel']),3072) and np.isfinite(z['features']).all() and np.array_equal(z['accel'],r['accel'])
  if v==0:assert np.array_equal(z['steer'],r['steer'])
  return (r['id'],v),z
 tasks=[(r,v) for r in train for v in [0,1]]+[(r,0) for r in val]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for key,z in pool.map(read,tasks):arrays[key]=z
 pools={d:{} for d in domains}
 for d in domains:
  for c in range(4):
   entries=[]
   for r in train:
    if r['domain']!=d:continue
    indices=np.flatnonzero(np.asarray(r['accel'])==c)[::4]
    entries.extend((r['id'],int(i)) for i in indices)
   if entries:pools[d][c]=entries
 aa=np.concatenate([np.asarray(r['accel']) for r in train]);ss=np.concatenate([arrays[r['id'],v]['steer'] for r in train for v in [0,1]]);am=np.repeat([False],1)
 aa2=np.concatenate([arrays[r['id'],v]['accel'] for r in train for v in [0,1]])
 def weight(count):
  w=1/np.sqrt(np.maximum(count,1));return torch.tensor(np.clip(w/w.mean(),.5,2),device='cuda',dtype=torch.float32)
 aw=weight(np.bincount(aa,minlength=4));sw=weight(np.bincount(ss[aa2!=3],minlength=3))
 def evaluate():
  model.eval();results={}
  with torch.inference_mode():
   for d in domains:
    ats=[];sts=[];aps=[];sps=[];speeds=[]
    for r in val:
     if r['domain']!=d:continue
     z=arrays[r['id'],0];a,s=probabilities(model,'decoupled_ltc',z['features'],'cuda');ats.append(z['accel']);sts.append(z['steer']);aps.append(a.numpy().argmax(1));sps.append(s.numpy().argmax(1));speeds.append(r['speed_mps'])
    a,s,ap,sp,speed=map(np.concatenate,[ats,sts,aps,sps,speeds]);mv=a!=3
    results[d]={'accel':prior.metric(a,ap,4),'moving_steer':prior.metric(s[mv],sp[mv],3),'speed_subsets':{}}
    for tag,mask in [('low',speed<15),('high',speed>=15)]:
     moving=mask&mv;results[d]['speed_subsets'][tag]={'accel':prior.metric(a[mask],ap[mask],4),'moving_steer':prior.metric(s[moving],sp[moving],3)}
  score=float(np.mean([results[d][k]['macro_f1'] for d in domains if d!='l2d' for k in ['accel','moving_steer']]))
  return results,score
 if not (OUT/'baseline_validation.json').exists():
  status('baseline_validation');metrics,score=evaluate();save('baseline_validation.json',{'validation':metrics,'selection_score':score})
 torch.manual_seed(SEED+10000);torch.cuda.manual_seed_all(SEED+10000)
 opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.02);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=LR/10)
 start=0;history=[];best=-1
 if (OUT/'trained/resume.pt').exists():
  z=torch.load(OUT/'trained/resume.pt',map_location='cpu',weights_only=False);assert z['normalization']==MODE and z['warm_start_sha256']==base_sha;model.load_state_dict(z['model']);opt.load_state_dict(z['optimizer']);sched.load_state_dict(z['scheduler']);start=z['next_epoch'];history=z['history'];best=z['best'];torch.set_rng_state(z['torch']);torch.cuda.set_rng_state_all(z['cuda']);del z
 for ep in range(start,EPOCHS):
  rng=random.Random(SEED+ep);model.train();losses=[];counts={d:np.zeros(4,dtype=np.int64) for d in domains};slots={d:0 for d in domains};started=time.time();torch.cuda.reset_peak_memory_stats();schedule_hash=hashlib.sha256();cuda_start_sha=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest();first_batch_sha=None
  for step in range(STEPS):
   selected=[]
   for di,d in enumerate(domains):
    quota=9+int((di-step)%5<3);slots[d]+=quota;classes=list(pools[d]);weights=[ [.35,.35,.20,.10][c] for c in classes]
    for _ in range(quota):
     c=rng.choices(classes,weights)[0];rid,center=rng.choice(pools[d][c]);v=rng.randrange(2);z=arrays[rid,v];n=len(z['accel']);lo=max(0,min(center-rng.randrange(T),n-T));hi=min(n,lo+T);selected.append((d,z,lo,hi));schedule_hash.update(json.dumps((d,rid,v,lo,hi),separators=(',',':')).encode())
   assert len(selected)==BATCH
   xx=np.zeros((BATCH,3072,T),np.float32);a=np.zeros((BATCH,T),np.int64);s=np.zeros_like(a);mask=np.zeros((BATCH,1,T),bool)
   for j,(d,z,lo,hi) in enumerate(selected):
    n=hi-lo;xx[j,:,:n]=z['features'][lo:hi].T;a[j,:n]=z['accel'][lo:hi];s[j,:n]=z['steer'][lo:hi];mask[j,0,:n]=True;counts[d]+=np.bincount(a[j,:n],minlength=4)
   x=torch.from_numpy(xx).cuda();a=torch.from_numpy(a).cuda();s=torch.from_numpy(s).cuda();mask=torch.from_numpy(mask).cuda()
   choose=torch.rand(BATCH,1,1,device='cuda')<.5;fm=(torch.rand(BATCH,3072,1,device='cuda')>.1)/.9;x=x*torch.where(choose,fm,torch.ones_like(fm));x=(x+torch.randn_like(x)*.025)*mask
   if step==0:first_batch_sha=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
   opt.zero_grad(set_to_none=True);output=model(x,mask);loss=prior.batched_loss(output,a,s,mask,aw,sw);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
   if (step+1)%25==0:status('training',epoch=ep+1,step=step+1,total_steps=STEPS,seconds=time.time()-started)
  assert len(set(slots.values()))==1
  del x,a,s,mask,output,loss,fm,choose;sched.step();peak=torch.cuda.max_memory_allocated()/2**30;status('validation',epoch=ep+1);metrics,score=evaluate()
  row={'epoch':ep+1,'loss':float(np.mean(losses)),'selection_score':score,'validation':metrics,'epoch_seconds':time.time()-started,'peak_allocated_GiB':peak,'schedule_sha256':schedule_hash.hexdigest(),'first_augmented_batch_sha256':first_batch_sha,'cuda_rng_start_sha256':cuda_start_sha,'cuda_rng_end_sha256':hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),'actual_sampling':{d:{'windows':slots[d],'accel_counts':counts[d].tolist()} for d in domains}};history.append(row)
  checkpoint={'input_dim':3072,'config':cfg,'model_dim':256,'intermediate_normalization':MODE,'members':[{'kind':('decoupled_ltc' if MODE=='joint9' else 'taskwise_ltc_keep_aux'),'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}],'metrics':row}
  pt(f'trained/epoch_{ep+1:02d}.pt',checkpoint)
  if score>best:best=score;pt('trained/head.pt',checkpoint)
  save('trained/history.json',history);pt('trained/resume.pt',{'normalization':MODE,'warm_start_sha256':base_sha,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'next_epoch':ep+1,'history':history,'best':best,'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
  status('epoch_complete',epoch=ep+1,score=score,best=best);gc.collect()
 assert prior.sha(base)==base_sha
 save('training_complete.json',{'selected':max(history,key=lambda x:x['selection_score']),'head_sha256':prior.sha(OUT/'trained/head.pt'),'baseline':json.loads((OUT/'baseline_validation.json').read_text())['selection_score']});status('TRAINING_COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:
  save('training_failure.json',{'time':time.time(),'traceback':traceback.format_exc()});status('TRAINING_FAILED');raise
