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

def make_model(kind,dim=3072,model_dim=96):
    if kind in ['balanced_tcn','motion_tcn']:return MotionTCN(dim,kind=='motion_tcn')
    if kind=='decoupled_ltc':
        sys.path.insert(0,str(Path(__file__).parent/'vendor/LTContext'))
        from ltc.model.ltcontext import LTC
        c=NS(INPUT_DIM=dim,NUM_CLASSES=9,ATTENTION=NS(NUM_ATTN_HEADS=1,DROPOUT=.15),
            LTC=NS(NUM_LAYERS=7,MODEL_DIM=model_dim,NUM_STAGES=3,DIM_REDUCTION=2,CONV_DILATION_FACTOR=2,
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
    from extended_ltc_runtime import load_extended_member
    members=[]
    for item in saved['members']:
        if item['kind'] in ('decoupled_ltc','taskwise_ltc_keep_aux'):
            members.append(load_extended_member(saved,item,make_model,device))
        else:
            if 'neck_config' in saved or saved.get('intermediate_normalization','joint9')!='joint9':
                raise ValueError('Extended metadata cannot be applied to this member')
            m=make_model(item['kind'],saved['input_dim'],saved.get('model_dim',96)).to(device).eval()
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


# ---- STAGE3_ENSEMBLE_TTA_PATCH v8 (unified ensemble, extra head kinds tcn/bigru/tfm/mstcn, optional profiles, per-profile per-task weights, per-task TTA, car-aware comma) ----
import json as _e_json
import numpy as _e_np
from scipy.ndimage import uniform_filter1d as _e_uf
try:
    _E_CFG = _e_json.loads(Path(__file__).with_name('postproc.json').read_text())
except Exception:
    _E_CFG = {}
_E_CAR = None


def load_members(saved, device):
    global _E_CAR
    from extended_ltc_runtime import load_extended_member
    _E_CAR = saved.get('car_clf')
    members = []
    for item in saved['members']:
        sub = {'input_dim': saved['input_dim'], 'model_dim': int(item.get('model_dim', saved.get('model_dim', 96)))}
        for k in ('intermediate_normalization', 'neck_config'):
            if k in item: sub[k] = item[k]
        if item['kind'] in ('motion_tcn', 'balanced_tcn', 'bigru', 'tfm', 'mstcn', 'dgru', 'mspool', 'vn_tcn'):
            kind = item['kind']
            if kind in ('motion_tcn', 'balanced_tcn'):
                m = make_model(kind, int(saved['input_dim']))
            else:
                import sys as _e_sys2
                _e_sys2.path.insert(0, str(Path(__file__).parent)); from extra_heads import make_extra
                m = make_extra(kind, int(saved['input_dim']))
            m = m.to(device); m.load_state_dict(item['model'], strict=True); m.eval()
        else:
            kind, m = load_extended_member(sub, item, make_model, device)
        wp = item.get('wprof') or {'default': [float(item.get('weight', 1.0))] * 2}
        members.append((kind, m, wp))
    return members


def _e_pp(pa, ps, cfg):
    pa = pa.float().cpu().numpy().astype(_e_np.float64); ps = ps.float().cpu().numpy().astype(_e_np.float64)
    wa = int(cfg.get('win_accel', 1)); ws = int(cfg.get('win_steer', 1))
    if wa > 1 and len(pa) > 1: pa = _e_uf(pa, wa, axis=0, mode='nearest')
    if ws > 1 and len(ps) > 1: ps = _e_uf(ps, ws, axis=0, mode='nearest')
    stopped = pa.argmax(1) == 3
    la = _e_np.log(_e_np.clip(pa, 1e-8, None)); ls = _e_np.log(_e_np.clip(ps, 1e-8, None))
    la[:, 0] += float(cfg.get('bias_accel', 0.0)); la[:, 1] += float(cfg.get('bias_decel', 0.0))
    ls[:, 0] += float(cfg.get('bias_left', 0.0)); ls[:, 2] += float(cfg.get('bias_right', 0.0))
    qa = _e_np.exp(la - la.max(1, keepdims=True)); qa /= qa.sum(1, keepdims=True)
    qs = _e_np.exp(ls - ls.max(1, keepdims=True)); qs /= qs.sum(1, keepdims=True)
    if cfg.get('lock_stopped', True): qa[stopped] = pa[stopped]
    return torch.from_numpy(qa), torch.from_numpy(qs)


def _e_w(w, prof):
    if isinstance(w, dict):
        v = w.get(prof, w.get('default', [1.0, 1.0])); return float(v[0]), float(v[1])
    w = float(w); return w, w


def _e_avg(members, features, device, prof='default'):
    ws = [_e_w(w, prof) for _, _, w in members]
    ta = sum(x[0] for x in ws); ts = sum(x[1] for x in ws); pa = ps = None
    for (k, m, _), (wa, wst) in zip(members, ws):
        if wa == 0 and wst == 0: continue
        a, s = probabilities(m, k, features, device)
        a = a.float().cpu() * (wa / ta); s = s.float().cpu() * (wst / ts)
        pa = a if pa is None else pa + a; ps = s if ps is None else ps + s
    return pa, ps


def _e_car(features):
    if not _E_CAR: return None
    x = _e_np.asarray(features, dtype=_e_np.float64)
    v = _e_np.concatenate([x.mean(0), x.std(0)])
    z = (v - _e_np.asarray(_E_CAR['mean'])) / _e_np.asarray(_E_CAR['scale'])
    logit = float(z @ _e_np.asarray(_E_CAR['coef']) + _E_CAR['intercept'])
    return _E_CAR['classes'][1] if logit > 0 else _E_CAR['classes'][0]


def _e_profile(frame_size, features):
    base = {k: v for k, v in _E_CFG.items() if k not in ('profiles', 'cars')}
    if frame_size is not None:
        fs = [int(frame_size[0]), int(frame_size[1])]
        for p in _E_CFG.get('profiles', []):
            if fs in [[int(v) for v in s] for s in p.get('sizes', [])]:
                cfg = dict(base, **{k: v for k, v in p.items() if k not in ('name', 'sizes', 'car')}); car = None
                if p.get('car'):
                    car = _e_car(features)
                    if car is not None: cfg = dict(cfg, **_E_CFG.get('cars', {}).get(car, {}))
                return cfg, p['name'], car
    return base, 'default', None


def predict(members, features, device, features_flip=None, frame_size=None):
    import os as _e_os, sys as _e_sys
    members = [x if len(x) == 3 else (x[0], x[1], 1.0) for x in members]
    cfg, prof, car = _e_profile(frame_size, features)
    pa, ps = _e_avg(members, features, device, prof)
    if features_flip is not None and (cfg.get('tta_accel', True) or cfg.get('tta_steer', True)):
        fa, fs = _e_avg(members, features_flip, device, prof)
        if cfg.get('tta_accel', True): pa = (pa + fa) / 2
        if cfg.get('tta_steer', True): ps = (ps + fs.flip(-1)) / 2   # steer: LEFT<->RIGHT
    if _e_os.environ.get('STAGE3_PROFILE_DEBUG'): print('PROFILE', frame_size, prof, car, file=_e_sys.stderr, flush=True)
    if cfg.get('enabled', False):
        pa, ps = _e_pp(pa, ps, cfg)
    return pa.argmax(-1), ps.argmax(-1)
