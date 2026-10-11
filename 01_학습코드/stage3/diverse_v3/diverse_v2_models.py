"""Decoupled acceleration/steering heads with optional motion regression."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class MotionTCN(nn.Module):
    def __init__(self,dim=3072,motion=False):
        super().__init__();self.motion=motion
        self.norm=nn.LayerNorm(dim)
        self.project=nn.Conv1d(dim*(3 if motion else 1),192,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(192,192,3,padding=d,dilation=d),nn.GELU(),nn.Dropout(.2),nn.Conv1d(192,192,1)) for d in [1,2,4,8,16,32]])
        self.output=nn.Conv1d(192,9,1)
    def forward(self,x):
        x=self.norm(x)
        if self.motion:
            # Temporal differences carry appearance change, never sensor inputs.
            prev=torch.cat([x[:1].expand(2,-1),x[:-2]],0) if len(x)>2 else x
            nxt=torch.cat([x[2:],x[-1:].expand(2,-1)],0) if len(x)>2 else x
            x=torch.cat([x,x-prev,nxt-x],-1)
        x=self.project(x.T[None])
        for b in self.blocks:x=x+b(x)
        return [self.output(x)[0].T]

def make_model(kind,dim=3072):
    if kind in ['balanced_tcn','motion_tcn']:return MotionTCN(dim,kind=='motion_tcn')
    if kind=='decoupled_ltc':
        sys.path.insert(0,str(Path(__file__).parent/'vendor/LTContext'))
        from ltc.model.ltcontext import LTC
        c=NS(INPUT_DIM=dim,NUM_CLASSES=9,ATTENTION=NS(NUM_ATTN_HEADS=1,DROPOUT=.15),
            LTC=NS(NUM_LAYERS=7,MODEL_DIM=96,NUM_STAGES=3,DIM_REDUCTION=2,CONV_DILATION_FACTOR=2,
            WINDOWED_ATTN_W=32,LONG_TERM_ATTN_G=32,USE_INSTANCE_NORM=True,DROPOUT_PROB=.15,CHANNEL_MASKING_PROB=.15))
        return LTC(c)
    raise ValueError(kind)

def forward(model,kind,x):
    if kind=='decoupled_ltc':
        return [y[0].T for y in model(x.T[None],torch.ones(1,1,len(x),dtype=torch.bool,device=x.device))]
    return model(x)

def loss(model,kind,x,a,s,reg,aw,sw):
    outputs=forward(model,kind,x);moving=a!=3;ls=[]
    for z in outputs:
        la=F.cross_entropy(z[:,:4],a,weight=aw,label_smoothing=.03)
        ls_=F.cross_entropy(z[moving,4:7],s[moving],weight=sw,label_smoothing=.03) if moving.any() else z[:,4:7].sum()*0
        l=la+ls_
        if kind=='motion_tcn':l=l+.15*F.smooth_l1_loss(z[:,7:],reg)
        # Modest label-preserving smoothness only inside unchanged true states.
        if len(a)>1:
            stable=(a[1:]==a[:-1])&((a[1:]==3)|(s[1:]==s[:-1]))
            if stable.any():
                p=F.log_softmax(z[:,:4],-1)
                l=l+.015*(p[1:][stable]-p[:-1].detach()[stable]).square().clamp(max=16).mean()
        ls.append(l)
    return sum(ls)/len(ls)

@torch.inference_mode()
def probabilities(model,kind,features,device,core=256,context=96):
    model.eval();aa=[];ss=[]
    for start in range(0,len(features),core):
        end=min(start+core,len(features));left=max(0,start-context);right=min(len(features),end+context)
        x=torch.as_tensor(features[left:right],dtype=torch.float32,device=device)
        z=forward(model,kind,x)[-1][start-left:end-left]
        aa.append(z[:,:4].softmax(-1).cpu());ss.append(z[:,4:7].softmax(-1).cpu())
    return torch.cat(aa),torch.cat(ss)

def load_members(saved,device):
    members=[]
    for item in saved['members']:
        m=make_model(item['kind'],saved['input_dim']).to(device).eval()
        m.load_state_dict(item['model'],strict=True);members.append((item['kind'],m))
    return members

def predict(members,features,device):
    results=[probabilities(m,k,features,device) for k,m in members]
    # Equal weights fixed before training. No tuning on held-out domains.
    a=torch.stack([r[0] for r in results]).mean(0).argmax(-1)
    s=torch.stack([r[1] for r in results]).mean(0).argmax(-1)
    return a,s

def smoke():
    for kind in ['balanced_tcn','decoupled_ltc','motion_tcn']:
        torch.manual_seed(42);m=make_model(kind,64).cuda();x=torch.randn(65,64,device='cuda')
        a=torch.arange(65,device='cuda')%4;s=torch.arange(65,device='cuda')%3
        reg=torch.zeros(65,2,device='cuda');w=torch.ones(4,device='cuda');sw=torch.ones(3,device='cuda')
        l=loss(m,kind,x,a,s,reg,w,sw);assert torch.isfinite(l);l.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        m.eval();l1=loss(m,kind,x,torch.full_like(a,3),s,reg,w,sw)
        l2=loss(m,kind,x,torch.full_like(a,3),(s+1)%3,reg,w,sw)
        assert torch.allclose(l1,l2), 'STOPPED steering leaked into smoothness'
        s0=torch.zeros_like(s)
        l1=loss(m,kind,x,torch.full_like(a,3),s0,reg,w,sw)
        l2=loss(m,kind,x,torch.full_like(a,3),s0+1,reg,w,sw)
        assert torch.allclose(l1,l2), 'STOPPED steering leaked into loss'
        for n in [1,17,65]:
            p=probabilities(m,kind,x[:n].detach().cpu().numpy(),'cuda');assert p[0].shape==(n,4) and p[1].shape==(n,3)
        print('V2_MODEL_SMOKE_PASS',kind,float(l.detach()),flush=True)
        del m;torch.cuda.empty_cache()
