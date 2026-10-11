"""Independent episode batching benchmark; no optimizer or checkpoint mutations."""
from lora_probe import *
B=int(sys.argv[1]);MICRO=int(sys.argv[2]);TAG=f"b{B}_m{MICRO}"
def encode(enc,frames,micro):
 b,n=frames.shape[:2];total=torch.zeros(b*n,3072,device='cuda');weight=torch.zeros(b*n,1,device='cuda')
 jobs=[(j,s) for s in range(0,n,8) for j in range(b)]
 for offset in range(0,len(jobs),micro):
  part=jobs[offset:offset+micro]
  clips=torch.stack([frames[j,torch.arange(s,s+16).clamp_max(n-1)].permute(1,0,2,3) for j,s in part]).cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):tokens=enc(clips)
  grid=tokens.float().reshape(len(part),8,24,24,-1)
  f=torch.cat([grid[:,:,ys:ye,xs:xe].mean((2,3)) for ys,ye,xs,xe in [(0,12,0,12),(0,12,12,24),(12,24,0,12),(12,24,12,24)]],-1)
  f=F.interpolate(f.transpose(1,2),size=16,mode='linear',align_corners=False).transpose(1,2)
  for i,(j,s) in enumerate(part):
   k=min(16,n-s);ix=torch.arange(j*n+s,j*n+s+k,device='cuda');total=total.index_add(0,ix,f[i,:k]);weight[ix]+=1
 return (total/weight).reshape(b,n,3072).transpose(1,2)
def main():
 torch.set_num_threads(4);torch.manual_seed(92716)
 enc=load_encoder(SOURCE/'code/vendor/vjepa2',OUT.parent/'kitti_50gb_v1/vjepa21_vitb_official.pt',torch.device('cuda'))
 for name,module in list(enc.named_modules()):
  if isinstance(module,nn.Linear) and (name.endswith('.attn.qkv') or name.endswith('.attn.proj')):
   parent,leaf=name.rsplit('.',1);setattr(enc.get_submodule(parent),leaf,LoRALinear(module))
 enc.use_activation_checkpointing=True;enc.eval()
 head=prior.make_wide().cuda();ck=torch.load(OUT/'trained/head.pt',map_location='cpu',weights_only=True);head.load_state_dict(ck['members'][0]['model']);head.eval()
 rows=[r for r in json.loads((OUT/'selected_episodes.json').read_text())['records'] if r['split']=='train' and r['frames']>=256][:B]
 assert len(rows)==B;frames=[]
 for r in rows:
  cap=cv2.VideoCapture(str(OUT/'clips'/(r['id']+'.mp4')));ff=[]
  for i in range(256):
   ok,im=cap.read();assert ok;ff.append(image_tensor(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),384))
  cap.release();frames.append(torch.stack(ff))
 frames=torch.stack(frames);measurements=[];reference=None
 opt=torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad]+list(head.parameters()),lr=1e-5)
 for b,micro in [(B,MICRO)]:
  enc.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.time()
  features=encode(enc,frames[:b],micro)
  first=features[0].detach().cpu()
  if reference is None:reference=first
  error=float((reference-first).abs().max());relative=float((reference-first).norm()/reference.norm())
  assert relative<.01,('batch changed first sequence features',relative)
  mask=torch.ones(b,1,256,device='cuda',dtype=torch.bool);a=torch.tensor([r['accel'][:256] for r in rows[:b]],device='cuda');s=torch.tensor([r['steer'][:256] for r in rows[:b]],device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=prior.batched_loss(head(features,mask),a,s,mask,torch.ones(4,device='cuda'),torch.ones(3,device='cuda'))
  loss.backward();opt.step();torch.cuda.synchronize();elapsed=time.time()-started
  grads=[p.grad for p in enc.parameters() if p.requires_grad and p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.norm()>0 for g in grads)
  measurements.append({'independent_sequences':b,'frames_per_sequence':256,'encoder_clip_batch':micro,'seconds':elapsed,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'first_sequence_max_abs_difference':error,'first_sequence_relative_l2':relative,'projected48_compute_seconds':elapsed*48/b})
  (OUT/f'lora_capacity_{TAG}.json').write_text(json.dumps({'probe_only':True,'head_eval_for_determinism':True,'episodes':[r['id'] for r in rows],'measurements':measurements},indent=2));print(measurements[-1],flush=True)
  del features,loss,mask,a,s,grads
if __name__=='__main__':
 try:main()
 except Exception:(OUT/f'lora_capacity_{TAG}_failure.txt').write_text(traceback.format_exc());raise
