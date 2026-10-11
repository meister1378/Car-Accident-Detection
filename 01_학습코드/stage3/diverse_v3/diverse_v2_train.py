"""Three training recipes, shared disjoint datasets, untouched final test."""
import argparse,json,time,random,hashlib,shutil,sys,traceback,gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageEnhance,ImageFilter,ImageOps
from model import load_encoder,extract_sequence,macro_f1
from diverse_v2_data import prepare_all,write_json
from diverse_v2_models import make_model,loss,probabilities,smoke,load_members,predict

KINDS=['balanced_tcn','decoupled_ltc','motion_tcn']

def evaluation(m,kind,arrays,keys,records):
    result={}
    for domain in sorted({records[k]['domain'] for k in keys}):
        aa=[];ap=[];ss=[];sp=[]
        for k in keys:
            if records[k]['domain']!=domain:continue
            r=arrays[k];p,q=probabilities(m,kind,r['features'],'cuda');moving=r['accel']!=3
            aa.extend(r['accel']);ap.extend(p.argmax(-1).numpy());ss.extend(r['steer'][moving]);sp.extend(q.argmax(-1).numpy()[moving])
        result[domain]={'accel':macro_f1(aa,ap,4),'moving_steer':macro_f1(ss,sp,3)}
    score=float(np.mean([v[task]['macro_f1'] for v in result.values() for task in ['accel','moving_steer']]))
    return result,score

def main(root):
    root=Path(root);code=Path(__file__).parent;old=root/'stage3_vjepa_runs/three_heads_20260917'
    out=root/'stage3_vjepa_runs/diverse_v2_20260917';out.mkdir(parents=True,exist_ok=True)
    work=Path('/content/diverse_v2_work');work.mkdir(exist_ok=True)
    def status(phase,**kw):write_json(out/'status.json',{'phase':phase,'time':time.time(),**kw})
    assert torch.cuda.is_available();smoke()
    shutil.copytree(code,out/'code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','__pycache__','*.zip','*.ipynb'))
    cfg=json.loads((old/'config.json').read_text());cfg.update(epochs=30,learning_rate=.00015,seed=42)
    old_cfg=json.loads((old/'config.json').read_text())
    feature_keys=['fps','resolution','clip_frames','clip_stride','repo_commit']
    assert all(cfg[k]==old_cfg[k] for k in feature_keys)
    assert hashlib.sha256((code/'model.py').read_bytes()).digest()==hashlib.sha256((old/'code/model.py').read_bytes()).digest(), 'Old feature extractor differs; rebuild cache'
    write_json(out/'feature_provenance.json',{'reused_from':str(old/'features'),'extractor_sha256':hashlib.sha256((code/'model.py').read_bytes()).hexdigest(),
        'encoder_checkpoint':str(root/'stage3_vjepa_runs/kitti_50gb_v1/vjepa21_vitb_official.pt'),'feature_config':{k:cfg[k] for k in feature_keys},
        'KITTI_views':'original plus previously saved consistent augmented view',
        'new_views':'original plus stronger consistent augmented view; domain augmentations differ explicitly',
        'label_recipe':'centered 5-frame edge-padded speed smoothing; central gradient at 10Hz; accel >.4/<-.4 else constant; STOPPED initialized false, enter speed<.3 exit>=.6; positive yaw LEFT',
        'macro_f1':'fixed classes, absent classes count as zero; STOPPED excluded from all steering losses, weights and metrics'})
    write_json(out/'config.json',cfg)
    status('preparing_diverse_data')
    if (out/'manifest.json').exists():records=json.loads((out/'manifest.json').read_text())['records']
    else:records=prepare_all(root,out,work,cfg)
    cache=out/'features';cache.mkdir(exist_ok=True)
    encoder=None
    for r in records:
        if r['domain']=='kitti':continue
        for view in range(2 if r['split']=='train' else 1):
            path=cache/(r['id']+f'_v{view}.npz')
            if path.exists():
                with np.load(path) as z:assert z['features'].shape==(len(r['images']),3072)
                continue
            if encoder is None:encoder=load_encoder(code/'vendor/vjepa2',root/'stage3_vjepa_runs/kitti_50gb_v1/vjepa21_vitb_official.pt',torch.device('cuda'))
            seed=int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16);rng=random.Random(seed)
            flip=view==1 and rng.random()<.5
            brightness,contrast,color=rng.uniform(.65,1.3),rng.uniform(.75,1.25),rng.uniform(.6,1.25);blur=rng.uniform(0,.8)
            def load(i):
                with Image.open(r['images'][i]) as im:im=im.convert('RGB')
                if view:
                    if flip:im=ImageOps.mirror(im)
                    im=ImageEnhance.Brightness(im).enhance(brightness)
                    im=ImageEnhance.Contrast(im).enhance(contrast)
                    im=ImageEnhance.Color(im).enhance(color).filter(ImageFilter.GaussianBlur(blur))
                return im
            status('features',record=r['id'],split=r['split'],view=view)
            f=extract_sequence(load,len(r['images']),encoder,cfg,torch.device('cuda'))
            s=np.asarray(r['steer']);yaw=np.asarray(r['yaw_left_radps'])
            if flip:s=2-s;yaw=-yaw
            temp=path.with_suffix('.partial.npz')
            np.savez_compressed(temp,features=f.astype(np.float16),accel=r['accel'],steer=s,regression=np.stack([r['acceleration_mps2'],yaw/.1],-1).clip(-5,5))
            temp.replace(path);print('V2_FEATURES',r['id'],view,len(f),flush=True)
    del encoder;gc.collect();torch.cuda.empty_cache()
    arrays={};index={}
    for r in records:
        for view in range(2 if r['split']=='train' else 1):
            k=r['id']+f'_v{view}';p=(old/'features' if r['domain']=='kitti' else cache)/(k+'.npz')
            with np.load(p) as z:arrays[k]={n:z[n] for n in z.files}
            if 'regression' not in arrays[k]:
                yaw=np.asarray(r['yaw_left_radps'])
                if view:
                    rng=random.Random(int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16))
                    if rng.random()<.5:yaw=-yaw
                arrays[k]['regression']=np.stack([r['acceleration_mps2'],yaw/.1],-1).clip(-5,5)
            index[k]=r
    train=[k for k in arrays if index[k]['split']=='train'];val=[k for k in arrays if index[k]['split']=='val']
    # Domain and class balanced centers: draw domain uniformly, then accel class, then center.
    pools={}
    for domain in sorted({index[k]['domain'] for k in train}):
        pools[domain]={c:[] for c in range(4)}
        for k in train:
            if index[k]['domain']!=domain:continue
            for c in range(4):
                ids=np.flatnonzero(arrays[k]['accel']==c)[::8]
                pools[domain][c].extend((k,int(i)) for i in ids)
        pools[domain]={c:v for c,v in pools[domain].items() if v}
    rng=random.Random(42);schedule=[]
    for epoch in range(30):
        steps=[]
        for step in range(240):
            domain=rng.choice(sorted(pools));c=rng.choice(sorted(pools[domain]));k,center=rng.choice(pools[domain][c])
            n=len(arrays[k]['accel']);length=rng.choice([128,256,384]);start=max(0,min(center-rng.randrange(length),n-length))
            steps.append([k,start,min(n,start+length)])
        schedule.append(steps)
    write_json(out/'window_schedule.json',schedule)
    a=np.concatenate([arrays[k]['accel'] for k in train]);s=np.concatenate([arrays[k]['steer'] for k in train])
    def weights(count):
        w=1/np.sqrt(np.maximum(count,1));w=w/w.mean();return torch.tensor(np.clip(w,.5,2),device='cuda',dtype=torch.float32)
    aw=weights(np.bincount(a,minlength=4));sw=weights(np.bincount(s[a!=3],minlength=3))
    write_json(out/'training_contract.json',{'epochs':30,'steps':240,'seed':42,'domain_and_accel_balanced':True,
        'loss':'decoupled weighted CE, smoothing .03; STOPPED masks steer; motion_tcn auxiliary regression',
        'augmentation':'consistent clip brightness/contrast/color/blur+mirror with LEFT/RIGHT swap; feature noise/dropout; no temporal reversal',
        'ensemble_weights':'equal fixed before evaluation','class_weights_accel':aw.tolist(),'class_weights_steer':sw.tolist()})
    comparison={}
    for ki,kind in enumerate(KINDS):
        folder=out/kind;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():comparison[kind]=json.loads((folder/'complete.json').read_text());continue
        torch.manual_seed(42+ki);np.random.seed(42+ki);random.seed(42+ki)
        m=make_model(kind).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.00015,weight_decay=.02)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30,eta_min=.000015)
        hist=[];best=-1
        for ep in range(30):
            m.train();losses=[]
            for k,start,end in schedule[ep]:
                it=arrays[k];sl=slice(start,end)
                x=torch.tensor(it['features'][sl],dtype=torch.float32,device='cuda')
                # Clip-consistent channel dropout plus low feature noise.
                if random.random()<.5:x=x*(torch.rand(1,x.shape[1],device='cuda')>.05)/.95
                x=x+torch.randn_like(x)*.015
                aa=torch.tensor(it['accel'][sl],device='cuda');ss=torch.tensor(it['steer'][sl],device='cuda')
                reg=torch.tensor(it['regression'][sl],device='cuda',dtype=torch.float32)
                ll=loss(m,kind,x,aa,ss,reg,aw,sw);assert torch.isfinite(ll)
                opt.zero_grad();ll.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(ll.detach()))
            sched.step();metrics,score=evaluation(m,kind,arrays,val,index)
            row={'epoch':ep+1,'loss':float(np.mean(losses)),'validation':metrics,'selection_score':score}
            hist.append(row);write_json(folder/'history.json',hist)
            if score>best:
                best=score
                torch.save({'input_dim':3072,'config':cfg,'members':[{'kind':kind,'model':{k:v.detach().cpu() for k,v in m.state_dict().items()}}],'metrics':row},folder/'head.pt')
            status('training',kind=kind,epoch=ep+1,best=best);print('V2_EPOCH',kind,json.dumps(row),flush=True)
        comparison[kind]=max(hist,key=lambda r:r['selection_score']);write_json(folder/'complete.json',comparison[kind]);write_json(out/'comparison.json',comparison)
        del m,opt;gc.collect();torch.cuda.empty_cache()
    # Freeze all selected weights and equal-weight ensemble before final-test inference.
    export=out/'export';export.mkdir(exist_ok=True)
    for kind in KINDS[:2]:shutil.copyfile(out/kind/'head.pt',export/(kind+'.pt'))
    saved=[torch.load(out/k/'head.pt',map_location='cpu',weights_only=True) for k in KINDS]
    torch.save({'input_dim':3072,'config':cfg,'members':[s['members'][0] for s in saved],
        'metrics':{'selection':'equal three-model ensemble fixed before test','members':comparison}},export/'ensemble.pt')
    status('final_test');reports={};test=[k for k in arrays if index[k]['split']=='test']
    for kind in ['balanced_tcn','decoupled_ltc','ensemble']:
        ck=torch.load(export/(kind+'.pt'),map_location='cpu',weights_only=True);members=load_members(ck,'cuda');reports[kind]={}
        for split,keys in [('validation',val),('test',test)]:
            reports[kind][split]={}
            for domain in sorted({index[k]['domain'] for k in keys}):
                at=[];ap=[];st=[];sp=[]
                for k in keys:
                    if index[k]['domain']!=domain:continue
                    it=arrays[k];aa,ss=predict(members,it['features'],'cuda');moving=it['accel']!=3
                    at.extend(it['accel']);ap.extend(aa.numpy());st.extend(it['steer'][moving]);sp.extend(ss.numpy()[moving])
                reports[kind][split][domain]={'accel':macro_f1(at,ap,4),'moving_steer':macro_f1(st,sp,3)}
        print('V2_FINAL_METRICS',kind,json.dumps(reports[kind]),flush=True)
        del members;torch.cuda.empty_cache()
    write_json(out/'final_metrics.json',reports);status('TRAINED_EVALUATED_EXPORT_PENDING')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    try:main(args.root)
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        write_json(Path(args.root)/'stage3_vjepa_runs/diverse_v2_20260917/failure.json',{'error':error,'time':time.time()});raise
