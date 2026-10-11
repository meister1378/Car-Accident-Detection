"""Held-out nuScenes mini sensor-proxy evaluation. Never selects checkpoints."""
import json, tarfile, zipfile, shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from prepare import labels
from model import load_encoder, extract_sequence, macro_f1
from experiment_heads import make_head, predict_sequence, decode_labels


def evaluate(root, run):
    root,run=Path(root),Path(run)
    source=root/'stage3_data/diverse_driving_20260916'
    data=Path('/content/nuscenes_external_mini');data.mkdir(exist_ok=True)
    # Only front camera and mini metadata; no archive path traversal.
    with tarfile.open(source/'v1.0-mini.tgz') as t:
        for member in t:
            if not member.isfile():continue
            if '/CAM_FRONT/' not in member.name and '/v1.0-mini/' not in '/'+member.name:continue
            dest=(data/member.name).resolve();assert dest.is_relative_to(data.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
    metas=list(data.rglob('scene.json'));assert len(metas)==1
    meta=metas[0].parent
    tables={n:json.loads((meta/(n+'.json')).read_text()) for n in ['scene','sample','sample_data','calibrated_sensor','sensor']}
    # Freeze all mini logs, before any model predictions. Training is KITTI only.
    split={'dataset':'nuScenes-mini','purpose':'external_sensor_proxy_only',
           'scenes':[s['name'] for s in tables['scene']],
           'log_tokens':sorted({s['log_token'] for s in tables['scene']}),
           'used_for_model_selection':False,'all_nuscenes_excluded_from_training':True}
    split_path=run/'external_split.json'
    if split_path.exists():assert json.loads(split_path.read_text())==split
    else:split_path.write_text(json.dumps(split,indent=2))
    sensor={s['token']:s for s in tables['sensor']}
    calib={s['token']:s for s in tables['calibrated_sensor']}
    samples={s['token']:s for s in tables['sample']}
    cfg=json.loads((run/'config.json').read_text())
    rows=[];excluded=[]
    with zipfile.ZipFile(source/'can_bus.zip') as z:
        for scene in tables['scene']:
            try:
                matches=[n for n in z.namelist() if n.endswith(scene['name']+'_pose.json')]
                assert len(matches)==1,'CAN pose unavailable'
                poses=json.loads(z.read(matches[0]))
                ts=np.array([p['utime'] for p in poses],dtype=np.float64)/1e6
                vel=np.array([p['vel'][0] for p in poses]);yaw=np.array([p['rotation_rate'][2] for p in poses])
                assert len(ts)>16 and (np.diff(ts)>0).all() and np.isfinite([vel,yaw]).all()
                assert vel.min()>=-.1,'Reverse motion'
                frames=sorted([s for s in tables['sample_data'] if
                    samples[s['sample_token']]['scene_token']==scene['token'] and
                    sensor[calib[s['calibrated_sensor_token']]['sensor_token']]['channel']=='CAM_FRONT'],key=lambda s:s['timestamp'])
                ft=np.array([f['timestamp'] for f in frames],dtype=np.float64)/1e6
                assert len(ft)>16 and (np.diff(ft)>0).all()
                start,end=max(ft[0],ts[0]),min(ft[-1],ts[-1])
                grid=start+np.arange(int(np.floor((end-start)*10))+1)/10
                ix=np.clip(np.searchsorted(ft,grid),1,len(ft)-1)
                ix=np.where(abs(ft[ix]-grid)<abs(ft[ix-1]-grid),ix,ix-1)
                sr=np.clip(np.searchsorted(ts,grid),1,len(ts)-1)
                assert (abs(ft[ix]-grid)<=.06).all(),'Camera alignment gap'
                assert ((ts[sr]-ts[sr-1])<=.15).all(),'CAN gap'
                paths=[data/frames[i]['filename'] for i in ix]
                assert all(p.exists() for p in paths),'Missing camera sweeps'
                v=np.interp(grid,ts,vel);w=np.interp(grid,ts,yaw)
                # Independent steering-sign check when enough nontrivial turns exist.
                steer_files=[n for n in z.namelist() if n.endswith(scene['name']+'_steeranglefeedback.json')]
                agreement=None
                if steer_files:
                    sm=json.loads(z.read(steer_files[0]));st=np.array([s['utime'] for s in sm])/1e6
                    sv=np.interp(grid,st,[s['value'] for s in sm])
                    turning=(v>1)&(abs(w)>.03)&(abs(sv)>.2)
                    if turning.sum()>=10:
                        agreement=float(np.mean(np.sign(sv[turning])==np.sign(w[turning])))
                        assert agreement>=.6,'Steering/yaw convention check failed'
                a,s,acc=labels(v,w,cfg)
                rows.append({'id':scene['name'],'log_token':scene['log_token'],'images':[str(p) for p in paths],
                             'accel':a.tolist(),'steer':s.tolist(),'camera_max_error_s':float(abs(ft[ix]-grid).max()),
                             'steer_yaw_sign_agreement':agreement,'frames':len(grid)})
            except (AssertionError,KeyError,ValueError) as e:
                excluded.append({'scene':scene['name'],'reason':str(e)})
    report={'split':split,'alignment':rows,'excluded':excluded,'metrics':{},'label_type':'sensor thresholds, NOT contest truth'}
    (run/'external_alignment.json').write_text(json.dumps(report,indent=2))
    assert rows,'No aligned external scenes; do not fabricate metrics'
    code=Path(__file__).parent
    checkpoint=root/'stage3_vjepa_runs/kitti_50gb_v1/vjepa21_vitb_official.pt'
    enc=load_encoder(code/'vendor/vjepa2',checkpoint,torch.device('cuda'))
    features={}
    for row in rows:
        def load(i):
            with Image.open(row['images'][i]) as im:return im.convert('RGB')
        features[row['id']]=extract_sequence(load,row['frames'],enc,cfg,torch.device('cuda'))
        print('EXTERNAL_FEATURES',row['id'],row['frames'],flush=True)
    del enc;torch.cuda.empty_cache()
    for kind in ['tcn','fact','ltcontext']:
        saved=torch.load(run/kind/'head.pt',map_location='cpu',weights_only=True)
        m=make_head(kind,saved['input_dim']).cuda().eval();m.load_state_dict(saved['model'],strict=True)
        at=[];ap=[];st=[];sp=[]
        for r in rows:
            a,s=decode_labels(predict_sequence(m,kind,features[r['id']],'cuda'))
            truth=np.array(r['accel']);moving=truth!=3
            at.extend(truth);ap.extend(a.tolist());st.extend(np.array(r['steer'])[moving]);sp.extend(s.numpy()[moving])
        report['metrics'][kind]={'accel':macro_f1(at,ap,4),'moving_steer':macro_f1(st,sp,3)}
        print('EXTERNAL_METRICS',kind,report['metrics'][kind],flush=True)
        del m;torch.cuda.empty_cache()
    report['limitations']='Small mini sample; sensor-derived labels; absent classes count as zero in fixed-class Macro-F1. No external-score model selection.'
    (run/'external_evaluation.json').write_text(json.dumps(report,indent=2))
    return report
