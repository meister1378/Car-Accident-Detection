"""Colab KITTI expansion + matched three-head experiments; no automatic publishing.

All three use joint labels (including the TCN comparator), identical windows,
seed, feature views, and optimizer-step budget. nuScenes stays out of training
until a separate log-disjoint external evaluation manifest has been audited.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from model import load_encoder, extract_sequence, macro_f1
from prepare import prepare
from experiment_heads import make_head, loss_and_predict, encode_labels, decode_labels, predict_sequence

BASE = Path(__file__).resolve().parent


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def smoke():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    for kind in ['tcn', 'fact', 'ltcontext']:
        m = make_head(kind, 64).to(device)
        x = torch.randn(96, 64, device=device)
        y = torch.arange(96, device=device)//10
        loss, pred = loss_and_predict(m, kind, x, y)
        assert torch.isfinite(loss) and pred.shape == (96,)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        m.eval()
        with torch.inference_mode():
            _, p1 = loss_and_predict(m, kind, x)
            _, p2 = loss_and_predict(m, kind, x)
        assert torch.equal(p1, p2)
        print('HEAD_SMOKE_PASS', kind, float(loss.detach()), flush=True)
    a = torch.tensor([0, 1, 2, 3, 3, 3])
    s = torch.tensor([0, 1, 2, 0, 1, 2])
    j = encode_labels(a, s)
    aa, ss = decode_labels(j)
    assert torch.equal(aa, a) and torch.equal(ss[:3], s[:3])
    assert j[-3:].tolist() == [9, 9, 9]
    print('STOPPED_STEERING_MASK_PASS', flush=True)


def main(root):
    assert torch.cuda.is_available(), 'GPU required'
    root = Path(root)
    out = root/'stage3_vjepa_runs/three_heads_20260917'
    out.mkdir(parents=True, exist_ok=True)
    work = Path('/content/stage3_three_heads_work')
    work.mkdir(exist_ok=True)
    save_json(out/'status.json', {'phase': 'smoke_tests', 'started': time.time()})
    smoke()
    # Persist exact executable sources, configs and licenses before training.
    shutil.copytree(BASE, out/'code', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.zip', '*.ipynb'))
    cfg = json.loads((BASE/'config.json').read_text())
    cfg.update(seed=42, epochs=25, learning_rate=.0002)
    save_json(out/'config.json', cfg)
    data = work/'kitti'; data.mkdir(exist_ok=True)
    sources = [root/'stage3_data/kitti_zips', root/'stage3_data/kitti_extra50gb_20260916']
    archives = {p.name:p for folder in sources for p in folder.glob('*.zip')}
    skipped = []
    save_json(out/'status.json', {'phase': 'selective_extraction', 'archives': len(archives)})
    for name, archive in sorted(archives.items()):
        try:
            with zipfile.ZipFile(archive) as z:
                selected = [i for i in z.infolist() if not i.is_dir() and
                    ('/image_02/data/' in i.filename or '/oxts/data/' in i.filename or
                     i.filename.endswith('/image_02/timestamps.txt') or i.filename.endswith('/oxts/timestamps.txt'))]
                if not selected: continue
                assert shutil.disk_usage(work).free > sum(i.file_size for i in selected)+2_000_000_000
                for info in selected:
                    target = (data/info.filename).resolve()
                    assert target.is_relative_to(data.resolve())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(target.suffix+'.partial')
                    with z.open(info) as src, temp.open('wb') as dst: shutil.copyfileobj(src, dst)
                    temp.replace(target)
            print('EXTRACTED', name, len(selected), flush=True)
        except zipfile.BadZipFile as e:
            # Known truncated old archives: never silently call them usable.
            skipped.append({'archive': name, 'reason': str(e)})
            print('EXCLUDED_BAD_ARCHIVE', name, str(e), flush=True)
    save_json(out/'excluded_archives.json', skipped)
    manifest = work/'manifest.json'
    prepare(data, manifest, cfg, {'2011_10_03'})
    shutil.copyfile(manifest, out/'manifest.json')
    shutil.copyfile(manifest.with_suffix('.report.json'), out/'label_report.json')
    records = json.loads(manifest.read_text())['records']
    for split in ['train','val']:
        rr = [r for r in records if r['split']==split]
        a = np.concatenate([r['accel'] for r in rr]); s = np.concatenate([r['steer'] for r in rr])
        assert (np.bincount(a, minlength=4)>0).all(), (split,'missing accel')
        assert (np.bincount(s[a!=3], minlength=3)>0).all(), (split,'missing steer')
    save_json(out/'split_contract.json', {
        'val_dates':['2011_10_03'], 'train_dates':sorted({r['group'] for r in records if r['split']=='train'}),
        'external_evaluation':'nuScenes pending log-disjoint alignment audit; not used for model selection',
        'label_type':'sensor thresholds, not official contest ground truth',
        'joint_classes':'0..8=accel*3+steer; 9=STOPPED independent of steer',
        'all_heads_same_windows':True, 'image_augmentation':'one deterministic per-recording view; same transforms at every frame',
        'epochs':25,'steps_per_epoch':200,'seed':42})
    repo = BASE/'vendor/vjepa2'
    checkpoint = root/'stage3_vjepa_runs/kitti_50gb_v1/vjepa21_vitb_official.pt'
    if not checkpoint.exists():
        candidates = list((root/'stage3_vjepa_runs').rglob('vjepa21_vitb_official.pt'))
        assert len(candidates)==1, 'Resolve official encoder checkpoint explicitly'
        checkpoint = candidates[0]
    cache = out/'features'; cache.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256(manifest.read_bytes()+(BASE/'model.py').read_bytes()+(BASE/'night_experiments.py').read_bytes()).hexdigest()
    if (cache/'signature.json').exists(): assert json.loads((cache/'signature.json').read_text())['sha256']==fingerprint
    save_json(cache/'signature.json', {'sha256':fingerprint})
    encoder = load_encoder(repo, checkpoint, torch.device('cuda'))
    for r in records:
        for view in range(2 if r['split']=='train' else 1):
            path = cache/(r['id']+f'_v{view}.npz')
            if path.exists():
                with np.load(path) as z: assert z['features'].shape==(len(r['images']),3072)
                continue
            rng = random.Random(int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16))
            flip = view==1 and rng.random()<.5
            brightness, contrast, color = rng.uniform(.8,1.2),rng.uniform(.85,1.15),rng.uniform(.85,1.15)
            blur = rng.uniform(0,.5)
            def load(i):
                with Image.open(r['images'][i]) as im: im=im.convert('RGB')
                if view:
                    if flip: im=ImageOps.mirror(im)
                    im=ImageEnhance.Brightness(im).enhance(brightness)
                    im=ImageEnhance.Contrast(im).enhance(contrast)
                    im=ImageEnhance.Color(im).enhance(color)
                    im=im.filter(ImageFilter.GaussianBlur(blur))
                return im
            save_json(out/'status.json',{'phase':'feature_extraction','record':r['id'],'view':view})
            f = extract_sequence(load,len(r['images']),encoder,cfg,torch.device('cuda'))
            assert np.isfinite(f).all()
            s = np.array(r['steer']); s=2-s if flip else s
            temp=path.with_suffix('.partial.npz')
            np.savez_compressed(temp,features=f.astype(np.float16),accel=r['accel'],steer=s)
            temp.replace(path)
            print('FEATURES_SAVED',r['id'],view,len(f),flush=True)
    del encoder; torch.cuda.empty_cache()
    arrays={}
    for r in records:
        for v in range(2 if r['split']=='train' else 1):
            key=r['id']+f'_v{v}'
            with np.load(cache/(key+'.npz')) as z: arrays[key]={k:z[k] for k in z.files}
    train_keys=[r['id']+f'_v{v}' for r in records if r['split']=='train' for v in range(2)]
    val_keys=[r['id']+'_v0' for r in records if r['split']=='val']
    windows=[(key,start) for key in train_keys for start in range(0,len(arrays[key]['accel']),192)
             if len(arrays[key]['accel'])-start>=32]
    schedule_rng = random.Random(42)
    schedule = [[schedule_rng.choice(windows) for _ in range(200)] for _ in range(25)]
    save_json(out/'training_window_schedule.json', schedule)
    summary={}
    for kind in ['tcn','fact','ltcontext']:
        dest=out/kind;dest.mkdir(exist_ok=True)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        m=make_head(kind,3072).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.01)
        best=-1;hist=[]
        for epoch in range(25):
            m.train(); losses=[]
            for step in range(200):
                key,start=schedule[epoch][step];item=arrays[key]
                sl=slice(start,min(start+384,len(item['accel'])))
                x=torch.tensor(item['features'][sl],dtype=torch.float32,device='cuda')
                a=torch.tensor(item['accel'][sl],device='cuda');s=torch.tensor(item['steer'][sl],device='cuda')
                loss,_=loss_and_predict(m,kind,x,encode_labels(a,s))
                assert torch.isfinite(loss), (kind,epoch,step)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                losses.append(float(loss.detach()))
            at=[];ap=[];st=[];sp=[]
            for key in val_keys:
                it=arrays[key];p=predict_sequence(m,kind,it['features'],'cuda');aa,ss=decode_labels(p)
                moving=it['accel']!=3
                at.extend(it['accel']);ap.extend(aa.tolist());st.extend(it['steer'][moving]);sp.extend(ss.numpy()[moving])
            am,sm=macro_f1(at,ap,4),macro_f1(st,sp,3);score=(am['macro_f1']+sm['macro_f1'])/2
            row={'epoch':epoch+1,'loss':float(np.mean(losses)),'accel':am,'moving_steer':sm,'internal_proxy':score}
            hist.append(row);save_json(dest/'history.json',hist)
            if score>best:
                best=score
                torch.save({'kind':kind,'input_dim':3072,'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'config':cfg,'metrics':row,'joint_labels':True},dest/'head.pt')
            save_json(out/'status.json',{'phase':'training','kind':kind,'epoch':epoch+1,'best_internal_proxy':best})
            print('EPOCH',kind,json.dumps(row),flush=True)
        summary[kind]=max(hist,key=lambda r:r['internal_proxy']);save_json(out/'comparison.json',summary)
        del m,opt;torch.cuda.empty_cache()
    save_json(out/'status.json',{'phase':'THREE_HEADS_TRAINED_EXPORT_AND_EXTERNAL_EVAL_PENDING','comparison':summary})
    print('THREE_HEADS_TRAINED_EXPORT_AND_EXTERNAL_EVAL_PENDING',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.smoke:smoke()
    else:
        try:main(args.root)
        except Exception:
            traceback.print_exc()
            if args.root:
                out=Path(args.root)/'stage3_vjepa_runs/three_heads_20260917'
                out.mkdir(parents=True,exist_ok=True)
                save_json(out/'failure.json',{'error':traceback.format_exc(),'time':time.time()})
            raise
