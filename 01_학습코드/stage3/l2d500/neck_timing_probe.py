from pathlib import Path
import sys,json,time,gc,traceback,random
import numpy as np
import torch
from torch import nn
R=Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
sys.path.insert(0,str(R))
import train_l2d as base
prior=base.prior
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.manual_seed(927)
class Neck(nn.Module):
 def __init__(self,depth):
  super().__init__();self.norm=nn.LayerNorm(3072);self.down=nn.Conv1d(3072,512,1)
  self.blocks=nn.ModuleList([nn.Conv1d(512,512,3,padding=2**i,dilation=2**i) for i in range(depth)])
  self.up=nn.Conv1d(512,3072,1);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
 def forward(self,x,m):
  z=self.down(self.norm(x.transpose(1,2)).transpose(1,2))*m
  for block in self.blocks:z=(z+torch.nn.functional.gelu(block(z)))*m
  return (x+self.up(z))*m
class Model(nn.Module):
 def __init__(self,depth,state):
  super().__init__();self.head=prior.make_wide();self.head.load_state_dict(state);self.neck=Neck(depth) if depth else None
 def forward(self,x,m):return self.head(self.neck(x,m) if self.neck else x,m)
def main():
 old=json.loads((base.SOURCE/'manifest.json').read_text());new=json.loads((R/'selected_episodes.json').read_text())
 records=[r for r in old['records']+new['records'] if r['split']=='train'];domains=sorted({r['domain'] for r in records})
 rng=random.Random(927);selected=[]
 for i,d in enumerate(domains):selected+=rng.choices([r for r in records if r['domain']==d],k=10 if i<3 else 9)
 started=time.perf_counter();data=[]
 for r in selected:
  p=R/'features'/(r['id']+'_v0.npz') if r['domain']=='l2d' else prior.cache(r,0)
  with np.load(p) as f:data.append({k:f[k].copy() for k in ['features','accel','steer']})
 load_seconds=time.perf_counter()-started
 state=torch.load(R/'trained/head.pt',map_location='cpu',weights_only=True)['members'][0]['model']
 result={'gpu':torch.cuda.get_device_name(),'batch':48,'window':256,'precision':'float32; TF32 off, same as frozen trainer','loaded_records':len(data),'initial_feature_read_seconds':load_seconds,'steps_per_epoch':300,'epochs':15,'measurements':[]}
 def batch():
  xx=np.zeros((48,3072,256),np.float32);aa=np.zeros((48,256),np.int64);ss=aa.copy();mm=np.zeros((48,1,256),bool)
  for j,z in enumerate(data):
   n=min(256,len(z['accel']));lo=rng.randrange(max(1,len(z['accel'])-n+1));xx[j,:,:n]=z['features'][lo:lo+n].T;aa[j,:n]=z['accel'][lo:lo+n];ss[j,:n]=z['steer'][lo:lo+n];mm[j,0,:n]=True
  return [torch.from_numpy(v).cuda() for v in [xx,aa,ss,mm]]
 for depth in [0,2,4]:
  model=Model(depth,state).cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.02);aw=torch.ones(4,device='cuda');sw=torch.ones(3,device='cuda');times=[];losses=[]
  torch.cuda.reset_peak_memory_stats()
  for step in range(23):
   torch.cuda.synchronize();t=time.perf_counter();x,a,s,m=batch()
   choose=torch.rand(48,1,1,device='cuda')<.5;fm=(torch.rand(48,3072,1,device='cuda')>.1)/.9;x=(x*torch.where(choose,fm,torch.ones_like(fm))+torch.randn_like(x)*.025)*m
   opt.zero_grad(set_to_none=True);out=model(x,m);loss=prior.batched_loss(out,a,s,m,aw,sw);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize()
   if step>=3:times.append(time.perf_counter()-t);losses.append(float(loss))
  row={'depth':depth,'trainable_parameters':sum(p.numel() for p in model.parameters()),'neck_parameters':sum(p.numel() for p in model.neck.parameters()) if depth else 0,'measured_steps':len(times),'step_mean_seconds':float(np.mean(times)),'step_median_seconds':float(np.median(times)),'step_p90_seconds':float(np.percentile(times,90)),'epoch300_train_seconds':float(np.mean(times)*300),'15epoch_train_minutes':float(np.mean(times)*4500/60),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'last_loss':losses[-1]}
  model.eval();torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode():
   for _ in range(10):out=model(x,m)
  torch.cuda.synchronize();row['eval_batch48_seconds']=(time.perf_counter()-t)/10
  result['measurements'].append(row);(R/'neck_timing_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
  del model,opt,x,a,s,m,out,loss,choose,fm;gc.collect();torch.cuda.empty_cache()
 result['complete']=True;result['limits']='Timing probe only; 48 sampled real train records, uniform class loss weights; CPU window assembly, H2D, feature augmentation, forward, backward and AdamW included. Full-cache initial load, complete validation and checkpoint/ZIP IO excluded. No trained weights saved.'
 (R/'neck_timing_probe.json').write_text(json.dumps(result,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  (R/'neck_timing_probe_failure.txt').write_text(traceback.format_exc());raise
