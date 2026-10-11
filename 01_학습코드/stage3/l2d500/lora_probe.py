"""Actual rank16 image-to-LTC gradient probe; does not save a trained checkpoint."""
from pathlib import Path
import sys,json,time,math,traceback,importlib.util
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import cv2
OUT=Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927');SOURCE=OUT.parent/'diverse_v3_20260917'
sys.path.insert(0,str(SOURCE/'code'))
from model import load_encoder,image_tensor
spec=importlib.util.spec_from_file_location('prior',OUT.parent/'v3_w256_B_15ep_20260926/train_three_experiments.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
class LoRALinear(nn.Module):
 def __init__(self,base,rank=16,alpha=16):
  super().__init__();self.base=base.requires_grad_(False);self.scale=alpha/rank
  self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device));self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.A),self.B)*self.scale
def main():
 torch.set_num_threads(4);torch.manual_seed(92716)
 enc=load_encoder(SOURCE/'code/vendor/vjepa2',OUT.parent/'kitti_50gb_v1/vjepa21_vitb_official.pt',torch.device('cuda'))
 targets=[(name,module) for name,module in enc.named_modules() if isinstance(module,nn.Linear) and (name.endswith('.attn.qkv') or name.endswith('.attn.proj'))]
 assert len(targets)==24,[(n,type(m).__name__) for n,m in targets]
 for name,module in targets:
  parent,leaf=name.rsplit('.',1);setattr(enc.get_submodule(parent),leaf,LoRALinear(module))
 enc.use_activation_checkpointing=True;enc.eval()
 head=prior.make_wide().cuda();ck=torch.load(OUT/'trained/head.pt',map_location='cpu',weights_only=True);head.load_state_dict(ck['members'][0]['model']);head.train()
 selected=json.loads((OUT/'selected_episodes.json').read_text());r=next(r for r in selected['records'] if r['split']=='train' and r['frames']>=64)
 cap=cv2.VideoCapture(str(OUT/'clips'/(r['id']+'.mp4')));frames=[]
 for i in range(32):
  ok,im=cap.read();assert ok;frames.append(image_tensor(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),384))
 cap.release();report={'rank':16,'alpha':16,'targets':[n for n,_ in targets],'lora_parameters':sum(p.numel() for p in enc.parameters() if p.requires_grad),'head_parameters':sum(p.numel() for p in head.parameters()),'episode':r['id'],'probe_only':True,'measurements':[]}
 for n in [16,32]:
  enc.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.time()
  x=torch.stack(frames[:n],dim=1)[None].cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   tokens=enc(x);grid=tokens.float().reshape(1,n//2,24,24,-1)
   features=torch.cat([grid[:,:,ys:ye,xs:xe].mean((2,3)) for ys,ye,xs,xe in [(0,12,0,12),(0,12,12,24),(12,24,0,12),(12,24,12,24)]],-1).transpose(1,2)
   # align_corners=False gives original tubelet centers at .5,2.5,...
   features=F.interpolate(features,size=n,mode='linear',align_corners=False)
   mask=torch.ones(1,1,n,device='cuda',dtype=torch.bool);a=torch.tensor(r['accel'][:n],device='cuda')[None];s=torch.tensor(r['steer'][:n],device='cuda')[None]
   outputs=head(features,mask);loss=prior.batched_loss(outputs,a,s,mask,torch.ones(4,device='cuda'),torch.ones(3,device='cuda'))
  loss.backward();torch.cuda.synchronize()
  grads={name:float(p.grad.norm()) for name,p in enc.named_parameters() if p.requires_grad and p.grad is not None};assert grads and all(math.isfinite(v) for v in grads.values()) and max(grads.values())>0
  assert all(p.grad is None for p in enc.parameters() if not p.requires_grad)
  report['measurements'].append({'frames':n,'seconds':time.time()-started,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'loss':float(loss.detach()),'nonzero_adapter_gradients':sum(v>0 for v in grads.values())})
  del x,tokens,grid,features,outputs,loss,mask,a,s
 (OUT/'lora_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  (OUT/'lora_probe_failure.json').write_text(traceback.format_exc());raise
