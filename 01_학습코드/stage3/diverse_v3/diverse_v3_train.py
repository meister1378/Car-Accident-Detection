"""Frozen VJEPA features, three LTC exports, exact old cache reuse."""
import json,random,hashlib,gc,time,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
from diverse_v3_data import ROOT,OUT,WORK,main as data_main
from diverse_v2_data import write_json
from model import load_encoder,extract_sequence,macro_f1
from diverse_v2_models import make_model,loss,load_members,predict

CODE=Path(__file__).parent;OLD=ROOT/'stage3_vjepa_runs/diverse_v2_20260917'
BASE=ROOT/'stage3_vjepa_runs/three_heads_20260917'
KINDS=['ltc_scratch','ltc_finetune','ltc_ensemble']

def status(phase,**kw):
    write_json(OUT/'status.json',dict(phase=phase,time=time.time(),**kw))

def savept(p,obj):
    p=Path(p);tmp=p.with_suffix('.partial.pt');torch.save(obj,tmp);tmp.replace(p)

def eval_members(members,arrays,index,keys):
    report={}
    for domain in sorted({index[k]['domain'] for k in keys}):
        a=[];ap=[];s=[];sp=[]
        for k in keys:
            if index[k]['domain']!=domain:continue
            it=arrays[k];pa,ps=predict(members,it['features'],'cuda');moving=it['accel']!=3
            a.extend(it['accel']);ap.extend(pa.numpy());s.extend(it['steer'][moving]);sp.extend(ps.numpy()[moving])
        report[domain]={'accel':macro_f1(a,ap,4),'moving_steer':macro_f1(s,sp,3)}
    score=float(np.mean([r[t]['macro_f1'] for r in report.values() for t in ['accel','moving_steer']]))
    return report,score

def cache_path(r,view):
    if r.get('cache_origin')=='v2':
        return (BASE if r['domain']=='kitti' else OLD)/'features'/(r['id']+f'_v{view}.npz')
    return OUT/'features'/(r['id']+f'_v{view}.npz')

def main():
    assert torch.cuda.is_available(),'GPU unavailable'
    torch.set_num_threads(4)
    status('alignment')
    if (OUT/'manifest.json').exists():
        d=json.loads((OUT/'manifest.json').read_text());records,cfg=d['records'],d['config']
    else:records,cfg=data_main()
    for name in ['model.py','prepare.py','diverse_v2_models.py']:
        assert hashlib.sha256((CODE/name).read_bytes()).digest()==hashlib.sha256((OLD/'code'/name).read_bytes()).digest(),f'V2 compatibility differs: {name}'
    assert cfg==json.loads((OLD/'config.json').read_text()),'old cache config changed'
    write_json(OUT/'feature_provenance.json',{'old_cache':'V2 immutable arrays; identical config/model.py/prepare.py/model head code asserted','encoder_checkpoint':str(ROOT/'stage3_vjepa_runs/kitti_50gb_v1/vjepa21_vitb_official.pt'),'label_recipe':'shared prepare.labels including class order, positive-left yaw and STOPPED steering mask','short_windows':'actual available length min(n,requested), no concatenation across gaps, model mask covers actual frames','new_views':'384px padded originals plus train-only coherent mirror/photometric/blur'})
    (OUT/'features').mkdir(exist_ok=True)
    encoder=None
    for ri,r in enumerate(records):
        for view in range(2 if r['split']=='train' else 1):
            path=cache_path(r,view)
            if path.exists():
                with np.load(path) as z:
                    assert z['features'].shape==(len(r['accel']),3072)
                    assert np.array_equal(z['accel'],r['accel'])
                continue
            assert r.get('cache_origin')!='v2',f'Missing immutable old cache: {path}'
            if encoder is None:encoder=load_encoder(CODE/'vendor/vjepa2',ROOT/'stage3_vjepa_runs/kitti_50gb_v1/vjepa21_vitb_official.pt',torch.device('cuda'))
            rng=random.Random(int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16))
            flip=view==1 and rng.random()<.5
            bright,contrast,color,blur=rng.uniform(.65,1.3),rng.uniform(.75,1.25),rng.uniform(.6,1.25),rng.uniform(0,.8)
            def load(i):
                with Image.open(r['images'][i]) as im:im=im.convert('RGB')
                if view:
                    if flip:im=ImageOps.mirror(im)
                    im=ImageEnhance.Brightness(im).enhance(bright)
                    im=ImageEnhance.Contrast(im).enhance(contrast)
                    im=ImageEnhance.Color(im).enhance(color).filter(ImageFilter.GaussianBlur(blur))
                return im
            status('features',record=r['id'],record_index=ri,total_records=len(records),view=view)
            f=extract_sequence(load,len(r['images']),encoder,cfg,torch.device('cuda'))
            assert np.isfinite(f).all()
            s=np.asarray(r['steer']);s=2-s if flip else s
            temp=path.with_suffix('.partial.npz')
            np.savez_compressed(temp,features=f.astype(np.float16),accel=r['accel'],steer=s)
            temp.replace(path);print('V3_FEATURES',ri,len(records),r['id'],view,len(f),flush=True)
    del encoder;gc.collect();torch.cuda.empty_cache()
    arrays={};index={}
    for r in records:
        for view in range(2 if r['split']=='train' else 1):
            k=r['id']+f'_v{view}'
            with np.load(cache_path(r,view)) as z:arrays[k]={n:z[n] for n in ['features','accel','steer']}
            index[k]=r
    train=[k for k in arrays if index[k]['split']=='train'];val=[k for k in arrays if index[k]['split']=='val']
    oldval=[k for k in val if index[k].get('cache_origin')=='v2']
    baseline=torch.load(OLD/'export/decoupled_ltc.pt',map_location='cpu',weights_only=True)
    if not (OUT/'baseline_metrics.json').exists():
        status('baseline_evaluation');members=load_members(baseline,'cuda')
        write_json(OUT/'baseline_metrics.json',{'validation':eval_members(members,arrays,index,val)[0],'common_v2_validation':eval_members(members,arrays,index,oldval)[0]})
        del members;torch.cuda.empty_cache()
    pools={}
    for d in sorted({index[k]['domain'] for k in train}):
        pools[d]={c:[] for c in range(4)}
        for k in train:
            if index[k]['domain']!=d:continue
            for c in range(4):pools[d][c].extend((k,int(i)) for i in np.flatnonzero(arrays[k]['accel']==c)[::8])
        pools[d]={c:p for c,p in pools[d].items() if p}
    rng=random.Random(314);schedule=[]
    for ep in range(30):
        steps=[]
        for step in range(300):
            d=rng.choice(sorted(pools));c=rng.choice(sorted(pools[d]));k,center=rng.choice(pools[d][c])
            n=len(arrays[k]['accel']);length=rng.choice([128,256,384]);start=max(0,min(center-rng.randrange(length),n-length))
            steps.append((k,start,min(n,start+length)))
        schedule.append(steps)
    write_json(OUT/'window_schedule.json',schedule)
    aa=np.concatenate([arrays[k]['accel'] for k in train]);ss=np.concatenate([arrays[k]['steer'] for k in train])
    def weights(count):
        w=1/np.sqrt(np.maximum(count,1));return torch.tensor(np.clip(w/w.mean(),.5,2),device='cuda',dtype=torch.float32)
    aw=weights(np.bincount(aa,minlength=4));sw=weights(np.bincount(ss[aa!=3],minlength=3))
    write_json(OUT/'training_contract.json',{'models':KINDS,'epochs':30,'steps_per_epoch':300,'seed':314,'sampling':'uniform dataset, then accel class, then frame; no temporal reverse','selection':'equal mean of 4 dataset validation accel/moving steer MacroF1','old_holdouts_immutable':True,'test_used_for_selection':False,'augmentation':'one consistent mirrored/photometric/blur view per train clip plus feature noise/channel dropout; no val/test augmentation','ensemble':'equal scratch+finetune fixed before test','labels':'sensor-derived weak labels, not contest ground truth','scratch_lr':.00015,'finetune_lr':.00005})
    for j,kind in enumerate(KINDS[:2]):
        folder=OUT/kind;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():continue
        torch.manual_seed(314+j);random.seed(314+j);np.random.seed(314+j)
        m=make_model('decoupled_ltc').cuda()
        if kind=='ltc_finetune':m.load_state_dict(baseline['members'][0]['model'])
        lr=.00015 if j==0 else .00005
        opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.02)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30,eta_min=lr/10)
        hist=[];best=-1;start_ep=0
        if (folder/'resume.pt').exists():
            ck=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
            m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
            hist=ck['history'];best=ck['best'];start_ep=ck['next_epoch']
            random.setstate(ck['random']);np.random.set_state(ck['numpy']);torch.set_rng_state(ck['torch']);torch.cuda.set_rng_state_all(ck['cuda'])
        for ep in range(start_ep,30):
            m.train();losses=[]
            for k,a,b in schedule[ep]:
                it=arrays[k];sl=slice(a,b);x=torch.tensor(it['features'][sl],device='cuda',dtype=torch.float32)
                if random.random()<.5:x=x*(torch.rand(1,x.shape[1],device='cuda')>.05)/.95
                x=x+torch.randn_like(x)*.015
                at=torch.tensor(it['accel'][sl],device='cuda');st=torch.tensor(it['steer'][sl],device='cuda')
                ll=loss(m,'decoupled_ltc',x,at,st,None,aw,sw);assert torch.isfinite(ll)
                opt.zero_grad();ll.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(ll.detach()))
            sched.step();metrics,score=eval_members([('decoupled_ltc',m)],arrays,index,val)
            row={'epoch':ep+1,'loss':float(np.mean(losses)),'selection_score':score,'validation':metrics};hist.append(row)
            if score>best:
                best=score;savept(folder/'head.pt',{'input_dim':3072,'config':cfg,'members':[{'kind':'decoupled_ltc','model':{k:v.detach().cpu() for k,v in m.state_dict().items()}}],'metrics':row})
            write_json(folder/'history.json',hist)
            savept(folder/'resume.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':hist,'best':best,'next_epoch':ep+1,'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
            status('training',kind=kind,epoch=ep+1,best=best);print('V3_EPOCH',kind,json.dumps(row),flush=True)
        write_json(folder/'complete.json',max(hist,key=lambda r:r['selection_score']))
        del m,opt;gc.collect();torch.cuda.empty_cache()
    export=OUT/'export';export.mkdir(exist_ok=True)
    saved=[]
    for kind in KINDS[:2]:
        shutil.copyfile(OUT/kind/'head.pt',export/(kind+'.pt'));saved.append(torch.load(export/(kind+'.pt'),map_location='cpu',weights_only=True))
    savept(export/'ltc_ensemble.pt',{'input_dim':3072,'config':cfg,'members':[s['members'][0] for s in saved],'metrics':{'equal_weights_fixed':True}})
    # All checkpoints frozen before held-out test is opened.
    reports={};test=[k for k in arrays if index[k]['split']=='test']
    for kind in KINDS:
        members=load_members(torch.load(export/(kind+'.pt'),map_location='cpu',weights_only=True),'cuda')
        reports[kind]={}
        for split,keys in [('validation',val),('common_v2_validation',oldval),('test',test),('legacy_v2_test',[k for k in test if index[k].get('cache_origin')=='v2']),('new_group_test',[k for k in test if index[k].get('cache_origin')!='v2'])]:
            if keys:reports[kind][split]=eval_members(members,arrays,index,keys)[0]
        del members;torch.cuda.empty_cache()
    write_json(OUT/'final_metrics.json',reports);status('TRAINED_EXPORT_PENDING')
    print('V3_TRAINING_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
