"""Versioned, log-disjoint, sensor-derived data expansion. No contest labels."""
import json, tarfile, zipfile, shutil, hashlib, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps
from prepare import labels

def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x));q.replace(p)

def extract(archive,dest,choose,resize=False):
    marker=dest/(archive.name+'.complete.json')
    if marker.exists():return
    dest.mkdir(parents=True,exist_ok=True);n=0
    with tarfile.open(archive) as t:
        for m in t:
            if not m.isfile() or not choose(m.name):continue
            p=(dest/m.name).resolve();assert p.is_relative_to(dest.resolve())
            p.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(m) as f:
                if resize and p.suffix.lower() in ['.png','.jpg']:
                    with Image.open(f) as im:
                        ImageOps.pad(im.convert('RGB'),(384,384),color=(124,116,104)).save(p)
                else:
                    with p.open('wb') as out:shutil.copyfileobj(f,out)
            n+=1
    write_json(marker,{'files':n,'source':str(archive),'resize':resize})
    print('EXTRACT_COMPLETE',archive.name,n,flush=True)

def aligned_records(name,domain,group,paths,ft,ts,v,w,cfg,split,extra_valid=None):
    ft,ts=np.asarray(ft),np.asarray(ts)
    assert len(ft)>16 and len(ts)>16 and (np.diff(ft)>0).all() and (np.diff(ts)>0).all()
    lo,hi=max(ft[0],ts[0]),min(ft[-1],ts[-1])
    grid=lo+np.arange(max(0,int(np.floor((hi-lo)*10))+1))/10
    ix=np.clip(np.searchsorted(ft,grid),1,len(ft)-1)
    ix=np.where(abs(ft[ix]-grid)<abs(ft[ix-1]-grid),ix,ix-1)
    si=np.clip(np.searchsorted(ts,grid),1,len(ts)-1)
    speed=np.interp(grid,ts,v);yaw=np.interp(grid,ts,w)
    valid=(abs(ft[ix]-grid)<=.06)&((ts[si]-ts[si-1])<=.15)&(speed>=-.1)
    if extra_valid is not None:valid &= extra_valid(grid)
    valid &= np.isfinite(speed)&np.isfinite(yaw)
    # Gaps/reverse intervals divide clips; never concatenate across a gap.
    edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
    records=[]
    for j,(a,b) in enumerate(zip(edges[::2],edges[1::2])):
        if b-a<32:continue
        aa,ss,acc=labels(speed[a:b],yaw[a:b],cfg)
        records.append({'id':name+f'_seg{j:03d}','domain':domain,'group':group,'split':split,
            'images':[str(paths[k]) for k in ix[a:b]],'accel':aa.tolist(),'steer':ss.tolist(),
            'speed_mps':speed[a:b].tolist(),'acceleration_mps2':acc.tolist(),'yaw_left_radps':yaw[a:b].tolist(),
            'camera_max_error_s':float(abs(ft[ix[a:b]]-grid[a:b]).max()),'label_source':domain+'_sensor_thresholds'})
    return records

def prepare_nuscenes(root,out,work,cfg):
    src=root/'stage3_data/diverse_driving_20260916';new=root/'stage3_data/diverse_v2_20260917'
    data=work/'nuscenes'
    needed={'scene.json','sample.json','sample_data.json','calibrated_sensor.json','sensor.json','log.json'}
    extract(src/'v1.0-trainval_meta.tgz',data,lambda n:Path(n).name in needed)
    for folder,name in [(src,'v1.0-trainval01_blobs_camera.tgz'),(new,'v1.0-trainval02_blobs_camera.tgz')]:
        local=Path('/content/diverse_driving')/name if folder==src else Path('/content/diverse_v2')/name
        extract(local if local.exists() else folder/name,data,lambda n:'/CAM_FRONT/' in n,resize=True)
    meta=next(data.rglob('scene.json')).parent
    def table(n):return json.loads((meta/(n+'.json')).read_text())
    scenes=table('scene');samples={s['token']:s['scene_token'] for s in table('sample')}
    sensors={s['token']:s['channel'] for s in table('sensor')}
    calib={s['token']:sensors[s['sensor_token']] for s in table('calibrated_sensor')}
    frames=defaultdict(list)
    for f in table('sample_data'):
        if calib[f['calibrated_sensor_token']]=='CAM_FRONT' and (data/f['filename']).exists():
            frames[samples[f['sample_token']]].append(f)
    old=json.loads((root/'stage3_vjepa_runs/three_heads_20260917/external_split.json').read_text())
    banned=set(old['log_tokens'])
    available=[s for s in scenes if s['log_token'] not in banned and len(frames[s['token']])>=32]
    logs=sorted({s['log_token'] for s in available},key=lambda x:hashlib.sha256(('v2-fixed42'+x).encode()).hexdigest())
    assert len(logs)>=8,('too few disjoint logs',len(logs))
    nt=max(1,len(logs)//10);nv=max(2,len(logs)//5)
    split={g:('test' if i<nt else 'val' if i<nt+nv else 'train') for i,g in enumerate(logs)}
    contract={'nuScenes_log_split':split,'excluded_mini_logs':sorted(banned),'seed':42,
        'kitti_val_date':'2011_10_03','a2d2':{'20180810_150607':'train','20190401_121727':'val','20190401_145936':'test'},
        'selection':'mean of per-domain val accel/moving-steer macro F1','test_used_for_selection':False,
        'mini_status':'previously reported reference; not a fresh independent test',
        'label_limitation':'Same sensor threshold recipe is shared; these F1 values do not measure contest truth.'}
    cp=out/'split_contract.json'
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:write_json(cp,contract)
    rows=[];excluded=[]
    with zipfile.ZipFile(src/'can_bus.zip') as z:
        names=z.namelist()
        for scene in available:
            try:
                files=[n for n in names if n.endswith(scene['name']+'_pose.json')]
                assert len(files)==1,'CAN pose missing'
                p=json.loads(z.read(files[0]));ts=np.array([s['utime'] for s in p])/1e6
                v=np.array([s['vel'][0] for s in p]);w=np.array([s['rotation_rate'][2] for s in p])
                f=sorted(frames[scene['token']],key=lambda s:s['timestamp']);ft=np.array([s['timestamp'] for s in f])/1e6
                # Require independent steering sign consistency where turning support exists.
                sf=[n for n in names if n.endswith(scene['name']+'_steeranglefeedback.json')]
                if sf:
                    st=json.loads(z.read(sf[0]));a=np.interp(ts,np.array([s['utime'] for s in st])/1e6,[s['value'] for s in st])
                    mask=(v>1)&(abs(w)>.03)&(abs(a)>.2)
                    if mask.sum()>=20:assert np.mean(np.sign(w[mask])==np.sign(a[mask]))>=.6,'steering sign disagreement'
                rr=aligned_records('nusc_'+scene['name'],'nuscenes',scene['log_token'],[data/s['filename'] for s in f],ft,ts,v,w,cfg,split[scene['log_token']])
                assert rr,'No contiguous aligned segment';rows.extend(rr)
            except (AssertionError,ValueError,KeyError) as e:excluded.append({'scene':scene['name'],'reason':str(e)})
    write_json(out/'nuscenes_exclusions.json',excluded)
    assert {r['split'] for r in rows}=={'train','val','test'}
    return rows

def prepare_a2d2_route(root,out,work,cfg,stamp,split):
    src=root/'stage3_data/a2d2_pilot_20260916';new=root/'stage3_data/diverse_v2_20260917';data=work/'a2d2'
    name=f'camera_lidar-{stamp}_camera_frontcenter.tar';local=Path('/content/diverse_v2')/name
    def choose(n):
        if '/cam_front_center/' not in n:return False
        if Path(n).suffix not in ['.png','.json']:return False
        return int(Path(n).stem.split('_')[-1])%3==0
    if stamp=='20180810150607':extract(local if local.exists() else new/name,data,choose,resize=True)
    else:
        # Streamed dataset preserves full archive MD5 but saves selected camera frames only.
        report=json.loads((new/(stamp+'_stream_report.json')).read_text())
        assert report['verified'] and report['full_source_md5_verified']
        selected=Path('/content/diverse_v2')/(stamp+'_selected.tar')
        if not selected.exists():selected=new/selected.name
        extract(selected,data,lambda n:True,resize=False)
    with tarfile.open(src/f'camera_lidar-{stamp}_bus_signals.tar') as t:
        m=next(m for m in t if m.name.endswith('.json'));bus=json.load(t.extractfile(m))
    def signal(k,unit):
        assert bus[k]['unit']==unit,(k,bus[k]['unit']);a=np.array(bus[k]['values'],float)
        assert np.isfinite(a).all() and (np.diff(a[:,0])>0).all();return a[:,0]/1e6,a[:,1]
    ts,v=signal('vehicle_speed','Unit_KiloMeterPerHour');v=v/3.6
    wt,w=signal('angular_velocity_omega_z','Unit_DegreOfArcPerSecon');w=np.deg2rad(w)
    st,sv=signal('steering_angle_calculated','Unit_DegreOfArc')
    sg=np.array(bus['steering_angle_calculated_sign']['values'])
    sign=np.interp(st,sg[:,0]/1e6,sg[:,1]);signed=sv*np.where(sign<.5,1,-1)
    yaw=np.interp(st,wt,w);moving=np.interp(st,ts,v)>1
    mask=moving&(abs(yaw)>.03)&(abs(signed)>10)
    agreement=float(np.mean(np.sign(yaw[mask])==np.sign(signed[mask]))) if mask.any() else 0
    assert mask.sum()>100 and agreement>.7,('A2D2 sign audit',int(mask.sum()),agreement)
    pairs=[]
    for p in data.rglob('*.json'):
        if '/cam_front_center/' not in str(p) or not p.name.startswith(stamp):continue
        d=json.loads(p.read_text());image=p.with_suffix('.png')
        if image.exists():pairs.append((d['cam_tstamp']/1e6,image))
    pairs.sort();ft=np.array([p[0] for p in pairs]);paths=[p[1] for p in pairs]
    # cam_tstamp is camera master timestamp; delay sensitivity recorded, no score-based shift search.
    def yaw_valid(grid):
        ix=np.clip(np.searchsorted(wt,grid),1,len(wt)-1)
        return (grid>=wt[0])&(grid<=wt[-1])&((wt[ix]-wt[ix-1])<=.15)
    rows=aligned_records('a2d2_'+stamp,'a2d2',stamp,paths,ft,ts,v,np.interp(ts,wt,w),cfg,split,yaw_valid)
    # Split long recording into bounded feature jobs; all remain in same route split.
    result=[]
    for r in rows:
        n=len(r['images'])
        for start in range(0,n,1800):
            end=min(n,start+1800)
            if end-start<32:continue
            rr={k:(value[start:end] if isinstance(value,list) and len(value)==n else value) for k,value in r.items()}
            rr['id']=r['id']+f'_part{start:06d}';result.append(rr)
    write_json(out/(stamp+'_alignment.json'),{'split':split,'steer_yaw_agreement':agreement,'sign_support':int(mask.sum()),'records':len(result),'frames':sum(len(r['images']) for r in result),'source_units':['km/h','degrees/s'],'camera_timestamp':'cam_tstamp, no tuned offset'})
    assert result,'A2D2 no aligned data';return result

def prepare_all(root,out,work,cfg):
    old=root/'stage3_vjepa_runs/three_heads_20260917'
    kitti=json.loads((old/'manifest.json').read_text())['records']
    for r in kitti:r['domain']='kitti'
    pre=out/'nuscenes_rows.json'
    if pre.exists():nusc=json.loads(pre.read_text())
    else:
        nusc=prepare_nuscenes(root,out,work,cfg);write_json(pre,nusc)
    rows=kitti+nusc
    for stamp,split in [('20180810150607','train'),('20190401121727','val'),('20190401145936','test')]:
        rows+=prepare_a2d2_route(root,out,work,cfg,stamp,split)
    sets={s:{r['domain']+':'+r['group'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    report={}
    for split in sets:
        for domain in sorted({r['domain'] for r in rows}):
            rr=[r for r in rows if r['split']==split and r['domain']==domain]
            if not rr:continue
            a=np.concatenate([r['accel'] for r in rr]);s=np.concatenate([r['steer'] for r in rr])
            report[split+'_'+domain]={'records':len(rr),'frames':len(a),'accel_counts':np.bincount(a,minlength=4).tolist(),'moving_steer_counts':np.bincount(s[a!=3],minlength=3).tolist()}
    write_json(out/'manifest.json',{'config':cfg,'records':rows});write_json(out/'label_report.json',report)
    print('DIVERSE_LABEL_REPORT',json.dumps(report),flush=True);gc.collect();return rows
