"""Measure a true 256-frame LTC window with original overlapping encoder clips."""
from lora_probe import *
def main():
 torch.set_num_threads(4);torch.manual_seed(92716)
 enc=load_encoder(SOURCE/'code/vendor/vjepa2',OUT.parent/'kitti_50gb_v1/vjepa21_vitb_official.pt',torch.device('cuda'))
 targets=[(n,m) for n,m in enc.named_modules() if isinstance(m,nn.Linear) and (n.endswith('.attn.qkv') or n.endswith('.attn.proj'))]
 assert len(targets)==24
 for name,module in targets:
  parent,leaf=name.rsplit('.',1);setattr(enc.get_submodule(parent),leaf,LoRALinear(module))
 enc.use_activation_checkpointing=True;enc.eval()
 head=prior.make_wide().cuda();ck=torch.load(OUT/'trained/head.pt',map_location='cpu',weights_only=True);head.load_state_dict(ck['members'][0]['model']);head.train()
 selected=json.loads((OUT/'selected_episodes.json').read_text());r=next(r for r in selected['records'] if r['split']=='train' and r['frames']>=256)
 cap=cv2.VideoCapture(str(OUT/'clips'/(r['id']+'.mp4')));frames=[]
 for i in range(256):
  ok,im=cap.read();assert ok;frames.append(image_tensor(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),384))
 cap.release();frames=torch.stack(frames);measurements=[]
 for micro in [4,8]:
  enc.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.time()
  total=torch.zeros(256,3072,device='cuda');weight=torch.zeros(256,1,device='cuda')
  for offset in range(0,32,micro):
   starts=[s*8 for s in range(offset,min(32,offset+micro))]
   clips=torch.stack([frames[torch.arange(s,s+16).clamp_max(255)].permute(1,0,2,3) for s in starts]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):tokens=enc(clips)
   grid=tokens.float().reshape(len(starts),8,24,24,-1)
   f=torch.cat([grid[:,:,ys:ye,xs:xe].mean((2,3)) for ys,ye,xs,xe in [(0,12,0,12),(0,12,12,24),(12,24,0,12),(12,24,12,24)]],-1)
   expanded=F.interpolate(f.transpose(1,2),size=16,mode='linear',align_corners=False).transpose(1,2)
   for j,s in enumerate(starts):
    n=min(16,256-s);ix=torch.arange(s,s+n,device='cuda');total=total.index_add(0,ix,expanded[j,:n]);weight[ix]+=1
  features=(total/weight).T[None];mask=torch.ones(1,1,256,device='cuda',dtype=torch.bool)
  a=torch.tensor(r['accel'][:256],device='cuda')[None];s=torch.tensor(r['steer'][:256],device='cuda')[None]
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=prior.batched_loss(head(features,mask),a,s,mask,torch.ones(4,device='cuda'),torch.ones(3,device='cuda'))
  loss.backward();torch.cuda.synchronize();elapsed=time.time()-started
  gradients=[p.grad for p in enc.parameters() if p.requires_grad and p.grad is not None]
  assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.norm()>0 for g in gradients)
  measurements.append({'window_frames':256,'encoder_clip_frames':16,'encoder_stride':8,'encoder_clips':32,'clip_microbatch':micro,'seconds':elapsed,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'loss':float(loss.detach()),'projected_48_window_compute_seconds':elapsed*48})
  (OUT/'lora_window_probe.json').write_text(json.dumps({'probe_only':True,'episode':r['id'],'measurements':measurements},indent=2));print(measurements[-1],flush=True)
  del total,weight,features,loss,clips,tokens,grid,f,expanded,gradients
if __name__=='__main__':
 try:main()
 except Exception:(OUT/'lora_window_probe_failure.json').write_text(traceback.format_exc());raise
