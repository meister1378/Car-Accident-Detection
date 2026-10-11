"""V3 sensor audit, immutable old holdouts, route/date-disjoint additions."""
import io,json,zipfile,tarfile,hashlib,shutil
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import cv2
from PIL import Image,ImageOps
from diverse_v2_data import write_json,extract,aligned_records

ROOT=Path('/content/MyDrive/MyDrive')
OUT=ROOT/'stage3_vjepa_runs/diverse_v3_20260917'
SRC=ROOT/'stage3_data/diverse_v3_20260917'
WORK=Path('/content/diverse_v3_work')

def clean(t,v):
    t=np.asarray(t,dtype=float).reshape(-1);v=np.asarray(v,dtype=float)
    assert len(t)==len(v)>16 and np.isfinite(t).all() and np.isfinite(v).all(),'invalid sensor array'
    assert (np.diff(t)>=0).all(),'sensor clock reversed'
    dup=np.diff(t)==0
    assert not dup.any() or np.allclose(v[1:][dup],v[:-1][dup],atol=1e-7),'conflicting duplicate timestamps'
    keep=np.r_[True,np.diff(t)>0]
    return t[keep],v[keep],int((~keep).sum())

def gapmask(t,g,maxgap=.15):
    j=np.clip(np.searchsorted(t,g),1,len(t)-1)
    return (g>=t[0])&(g<=t[-1])&((t[j]-t[j-1])<=maxgap)

def stable_split(groups,seed):
    groups=sorted(groups,key=lambda g:hashlib.sha256((seed+g).encode()).hexdigest())
    assert len(groups)>=5,'not enough independent groups'
    nt=max(1,len(groups)//10);nv=max(1,len(groups)//5)
    return {g:('test' if i<nt else 'val' if i<nt+nv else 'train') for i,g in enumerate(groups)}

def source_path(name):
    p=Path('/content/diverse_v3')/name
    return p if p.exists() else SRC/name

def prepare_comma(cfg,contract):
    """Official FRD IMU => left yaw=-gyro_z, no score-based lag fitting."""
    results=[];audit=[];candidates=[];support=defaultdict(lambda:[0,0])
    for chunk in [1,3]:
        with zipfile.ZipFile(source_path(f'comma2k19_Chunk_{chunk}.zip')) as z:
            prefixes=sorted(n.rsplit('/',1)[0] for n in z.namelist() if n.endswith('/video.hevc'))
            for prefix in prefixes:
                entry={'source':prefix,'vehicle':'RAV4' if chunk==1 else 'Civic'}
                try:
                    def arr(tail):return np.load(io.BytesIO(z.read(prefix+'/'+tail)),allow_pickle=False)
                    ft=arr('global_pose/frame_times').reshape(-1)
                    assert len(ft)>32 and np.isfinite(ft).all() and (np.diff(ft)>0).all()
                    st,s,sd=clean(arr('processed_log/CAN/speed/t'),arr('processed_log/CAN/speed/value').reshape(-1))
                    gt,g,gd=clean(arr('processed_log/IMU/gyro/t'),arr('processed_log/IMU/gyro/value'))
                    at,a,ad=clean(arr('processed_log/CAN/steering_angle/t'),arr('processed_log/CAN/steering_angle/value').reshape(-1))
                    assert g.shape[1]==3 and np.quantile(s,.99)<70 and s.min()>=-.1,'units/range invalid'
                    w=-g[:,2]
                    turn=(np.interp(at,st,s)>2)&(abs(np.interp(at,gt,w))>.03)&(abs(a)>1)&gapmask(gt,at)&gapmask(st,at)
                    agree=np.sign(np.interp(at,gt,w)[turn])==np.sign(a[turn])
                    route=prefix.split('/')[1];group=route.split('|')[0]+'|'+route.split('|')[1][:10]
                    support[group][0]+=int(agree.sum());support[group][1]+=int(turn.sum())
                    entry.update(group=group,duplicates_removed=sd+gd+ad,frames=len(ft),turn_support=int(turn.sum()),turn_agree=int(agree.sum()))
                    candidates.append((chunk,prefix,ft,st,s,gt,w,group,entry))
                except Exception as e:entry.update(excluded=str(e))
                audit.append(entry)
    # Split by vehicle+recording DATE, more conservative than adjacent route clips.
    groups_by_vehicle=defaultdict(set)
    for x in candidates:groups_by_vehicle[x[0]].add(x[7])
    proposed={}
    for vehicle,groups in groups_by_vehicle.items():proposed.update(stable_split(groups,'v3-comma42'))
    if 'comma_date_split' in contract:assert proposed==contract['comma_date_split']
    else:contract['comma_date_split']=proposed
    contract['comma_group_rule']='vehicle plus recording date, all routes on date stay together'
    write_json(OUT/'split_contract.json',contract)
    write_json(OUT/'comma_sensor_audit.json',{'segments':audit,'group_sign_support':dict(support),'coordinate_rule':'official forward/right/down -> left yaw=-gyro_z; CAN steering sign agreement gate >=0.7 with >=50 samples per date'})
    for chunk,prefix,ft,st,s,gt,w,group,entry in candidates:
        good,total=support[group]
        if total<50 or good/total<.7:
            entry['excluded']='date insufficient steering sign support or disagreement';continue
        token=hashlib.sha256(prefix.encode()).hexdigest()[:16]
        image_dir=WORK/'comma'/token;image_dir.mkdir(parents=True,exist_ok=True)
        paths=[image_dir/f'{i:06d}.jpg' for i in range(len(ft))]
        valid=lambda t:gapmask(gt,t)&(abs(np.interp(t,gt,w))<1.5)
        rr=aligned_records('comma_'+token,'comma2k19',group,paths,ft,st,s,np.interp(st,gt,w),cfg,proposed[group],valid)
        if not rr:entry['excluded']='no valid contiguous aligned interval';continue
        wanted={int(Path(p).stem) for r in rr for p in r['images']}
        marker=image_dir/'decoded.json'
        if not marker.exists():
            video=image_dir/'video.hevc'
            with zipfile.ZipFile(source_path(f'comma2k19_Chunk_{chunk}.zip')) as z, z.open(prefix+'/video.hevc') as f,video.open('wb') as out:shutil.copyfileobj(f,out)
            cap=cv2.VideoCapture(str(video));n=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                if n in wanted:
                    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    ImageOps.pad(im,(384,384),color=(124,116,104)).save(paths[n],quality=92)
                n+=1
            cap.release()
            if n!=len(ft):
                entry['excluded']=f'video timestamp count mismatch {n}/{len(ft)}';continue
            write_json(marker,{'decoded_frames':n,'selected_frames':len(wanted)})
        assert all(paths[i].exists() for i in wanted)
        for r in rr:r.update(vehicle=entry['vehicle'],source=prefix,alignment_validated=True)
        results.extend(rr);entry['accepted_frames']=sum(len(r['images']) for r in rr)
        print('V3_COMMA_ALIGNED',prefix,entry['accepted_frames'],flush=True)
    write_json(OUT/'comma_sensor_audit.json',{'segments':audit,'group_sign_support':dict(support),'units':'boot seconds,m/s,rad/s; yaw=-gyro_z','time_shift_s':0})
    assert results and {r['split'] for r in results}=={'train','val','test'}
    return results

def prepare_nusc(cfg,contract,old_ids):
    data=WORK/'nuscenes';src=ROOT/'stage3_data/diverse_driving_20260916'
    extract(src/'v1.0-trainval_meta.tgz',data,lambda n:Path(n).name in {'scene.json','sample.json','sample_data.json','log.json'})
    for n in ['03','06','08','10']:extract(SRC/f'v1.0-trainval{n}_front_sensor_candidates.tar',data,lambda n:'/CAM_FRONT/' in n,resize=True)
    meta=next(data.rglob('scene.json')).parent
    table=lambda n:json.loads((meta/(n+'.json')).read_text())
    samples={s['token']:s['scene_token'] for s in table('sample')};frames=defaultdict(list)
    for f in table('sample_data'):
        if '/CAM_FRONT/' in f['filename'] and (data/f['filename']).exists():frames[samples[f['sample_token']]].append(f)
    banned=set(contract['excluded_mini_logs']);oldsplit=contract['nuScenes_log_split']
    scenes=[s for s in table('scene') if len(frames[s['token']])>=32 and s['log_token'] not in banned]
    groups={s['log_token'] for s in scenes if s['log_token'] not in oldsplit}
    # Keep old log assignments verbatim, freeze all new log assignments before labels/model results.
    added=stable_split(groups,'v3-nusc42') if groups else {}
    if 'new_nuScenes_log_split' in contract:assert contract['new_nuScenes_log_split']==added
    else:contract['new_nuScenes_log_split']=added
    split={**oldsplit,**added};write_json(OUT/'split_contract.json',contract)
    inventory={r['scene']:r for r in json.loads((SRC/'diversity_inventory.json').read_text())['nuscenes_scene_inventory']}
    rows=[];audit=[]
    with zipfile.ZipFile(src/'can_bus.zip') as z:
        names=z.namelist()
        for scene in scenes:
            e={'scene':scene['name'],'log':scene['log_token'],'split':split[scene['log_token']]}
            try:
                # If a scene occurred in v2, exclude the entire scene even when v3 has more frames.
                assert not any(k.startswith('nusc_'+scene['name']+'_') for k in old_ids),'already in V2'
                info=inventory[scene['name']]
                assert not(info['protected_from_training'] and e['split']=='train'),'protected log in train'
                p=json.loads(z.read(next(n for n in names if n.endswith(scene['name']+'_pose.json'))))
                ts,v,d=clean(np.array([q['utime'] for q in p])/1e6,np.array([q['vel'][0] for q in p]))
                wt,w,_=clean(np.array([q['utime'] for q in p])/1e6,np.array([q['rotation_rate'][2] for q in p]))
                assert v.max()<70 and np.quantile(abs(w),.99)<1.5,'units/range invalid'
                sf=[n for n in names if n.endswith(scene['name']+'_steeranglefeedback.json')]
                assert sf,'missing independent steering sensor'
                sp=json.loads(z.read(sf[0]));at,a,_=clean(np.array([q['utime'] for q in sp])/1e6,np.array([q['value'] for q in sp]))
                yy=np.interp(at,wt,w);mask=(np.interp(at,ts,v)>1)&(abs(yy)>.03)&(abs(a)>.2)&gapmask(wt,at)
                agreement=float(np.mean(np.sign(yy[mask])==np.sign(a[mask]))) if mask.any() else None
                if mask.sum()>=20:assert agreement>=.6,'steering sign disagreement'
                f=sorted(frames[scene['token']],key=lambda s:s['timestamp']);ft=np.array([s['timestamp'] for s in f])/1e6
                rr=aligned_records('nusc_'+scene['name'],'nuscenes',scene['log_token'],[data/s['filename'] for s in f],ft,ts,v,np.interp(ts,wt,w),cfg,e['split'],lambda t:gapmask(wt,t))
                assert rr,'no valid aligned interval'
                for r in rr:r.update(alignment_validated=True,location=info['location'],night=info['night'],rain_mentioned=info['rain_mentioned'])
                rows.extend(rr);e.update(accepted_frames=sum(len(r['images']) for r in rr),turn_support=int(mask.sum()),sign_agreement=agreement)
            except (AssertionError,ValueError,KeyError,StopIteration) as err:e['excluded']=str(err)
            audit.append(e)
    write_json(OUT/'nuscenes_sensor_audit.json',{'scenes':audit,'units':'seconds,m/s,rad/s; vehicle FLU yaw positive left','time_shift_s':0})
    assert rows;return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    old=ROOT/'stage3_vjepa_runs/diverse_v2_20260917'
    cfg=json.loads((old/'config.json').read_text())
    contract=json.loads((OUT/'split_contract.json' if (OUT/'split_contract.json').exists() else old/'split_contract.json').read_text())
    base=json.loads((old/'manifest.json').read_text())['records']
    for r in base:r['cache_origin']='v2'
    allrows=list(base)
    for name,fn in [('new_nuscenes',lambda:prepare_nusc(cfg,contract,{r['id'] for r in base})),('comma',lambda:prepare_comma(cfg,contract))]:
        p=OUT/(name+'_rows.json')
        rr=json.loads(p.read_text()) if p.exists() else []
        if not rr or not all(Path(r['images'][0]).exists() and Path(r['images'][-1]).exists() for r in rr):rr=fn()
        write_json(p,rr);allrows+=rr
    assert len({r['id'] for r in allrows})==len(allrows),'duplicate record'
    sets={s:{r['domain']+':'+r['group'] for r in allrows if r['split']==s} for s in ['train','val','test']}
    assert not sets['train']&(sets['val']|sets['test']) and not sets['val']&sets['test']
    report={}
    for split in sets:
        for domain in sorted({r['domain'] for r in allrows}):
            rr=[r for r in allrows if r['split']==split and r['domain']==domain]
            if not rr:continue
            a=np.concatenate([r['accel'] for r in rr]);s=np.concatenate([r['steer'] for r in rr])
            report[split+'_'+domain]={'records':len(rr),'frames':len(a),'groups':len({r['group'] for r in rr}),'accel_counts':np.bincount(a,minlength=4).tolist(),'moving_steer_counts':np.bincount(s[a!=3],minlength=3).tolist()}
    write_json(OUT/'manifest.json',{'config':cfg,'records':allrows});write_json(OUT/'label_report.json',report)
    print('V3_ALIGNMENT_COMPLETE',json.dumps(report),flush=True)
    return allrows,cfg

if __name__=='__main__':main()
