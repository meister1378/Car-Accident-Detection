"""Select 450 training and 50 session-disjoint validation episodes from L2D R2."""
from pathlib import Path
import json,hashlib,time,sys
from collections import Counter
import numpy as np
import pandas as pd
WORK=Path('/content/l2d500_20260927')
OUT=Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
SOURCE=OUT.parent/'diverse_v3_20260917'
sys.path.insert(0,str(SOURCE/'code'))
from prepare import labels
def save(name,obj):
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def smooth(x):return np.convolve(np.pad(x,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
def main():
    cfg=json.loads((SOURCE/'manifest.json').read_text())['config']
    meta=pd.read_parquet(WORK/'meta/episodes/chunk-000/file-000.parquet').set_index('episode_index')
    cols=['observation.state.vehicle','observation.state.timestamp','action.continuous','action.discrete','timestamp','frame_index','episode_index','observation.state.road','observation.state.conditions','observation.state.lighting','observation.state.surface','task.policy']
    candidates=[];reject=Counter();ratios=[];angles=[];corr=[]
    for p in sorted((WORK/'data/chunk-000').glob('*.parquet')):
        df=pd.read_parquet(p,columns=cols)
        for eid,g in df.groupby('episode_index',sort=True):
            n=len(g);m=meta.loc[eid]
            if n!=int(m['length']) or n<64 or n>600:reject['length']+=1;continue
            v=np.stack(g['observation.state.vehicle']).astype(float);act=np.stack(g['action.continuous']);dis=np.stack(g['action.discrete']);t=g['timestamp'].to_numpy(dtype=float)
            if not np.isfinite(v).all() or not np.allclose(np.diff(t),.1,atol=.002) or not np.array_equal(g['frame_index'],np.arange(n)):reject['invalid_time_or_signal']+=1;continue
            if (v[:,0]<0).any():reject['negative_speed']+=1;continue
            # Raw speed is validated below against one-second GNSS displacement.
            speed=v[:,0]/3.6
            k=10;lat=np.deg2rad(v[:,3]);lon=np.deg2rad(v[:,4]);north=np.diff(lat,n=1)
            north=(lat[k:]-lat[:-k])*6371000;east=(lon[k:]-lon[:-k])*6371000*np.cos(lat[:-k]);gps=np.hypot(north,east)/(t[k:]-t[:-k]);raw=(v[k:,0]+v[:-k,0])/2
            moving=(gps>3)&(raw>5)&(gps<60)
            if moving.sum()>10:
                ratios.append(float(np.median(raw[moving]/gps[moving])))
                bearing=np.rad2deg(np.arctan2(east,north));delta=(bearing-v[:-k,1]+180)%360-180;angles.append(float(np.median(abs(delta[moving]))))
                if abs(np.median(raw[moving]/gps[moving])-3.6)>.8:reject['speed_gps_disagreement']+=1;continue
            heading=smooth(np.unwrap(np.deg2rad(v[:,1])))
            yaw=-np.gradient(heading,.1)
            if np.max(speed)>60 or ((speed>1).any() and np.quantile(abs(yaw[speed>1]),.99)>1.5):reject['implausible_dynamics']+=1;continue
            aa,ss,acc=labels(speed,yaw,cfg)
            if np.max(abs(acc))>10:reject['acceleration_spike']+=1;continue
            ac=np.bincount(aa,minlength=4);sc=np.bincount(ss[aa!=3],minlength=3)
            if ac[2]/n>.70:reject['constant_over70pct']+=1;continue
            if speed.std()>.3 and v[:,6].std()>.05:corr.append(float(np.corrcoef(acc,v[:,6])[0,1]))
            session=str(m['session_id']);held=int(hashlib.sha256(session.encode()).hexdigest()[:8],16)%10==0
            cam='observation.images.front_left';prefix='videos/'+cam+'/'
            candidates.append({'id':f'l2d_ep{eid:05d}','domain':'l2d','group':session,'split':'val' if held else 'train','episode_index':int(eid),'canonical_name':str(m['canonical_name']),
                'frames':n,'accel':aa.tolist(),'steer':ss.tolist(),'speed_mps':speed.tolist(),'acceleration_mps2':acc.tolist(),'yaw_left_radps':yaw.tolist(),
                'accel_counts':ac.tolist(),'steer_counts':sc.tolist(),'speed_bin':int(np.digitize(np.median(speed),[5,15,25])),
                'weak_motion_frames':int(((aa<2)&(abs(acc)<=1)).sum()),'strong_motion_frames':int(((aa<2)&(abs(acc)>1)).sum()),
                'road':str(g['observation.state.road'].mode().iloc[0]),'weather':str(g['observation.state.conditions'].mode().iloc[0]),'lighting':str(g['observation.state.lighting'].mode().iloc[0]),'surface':str(g['observation.state.surface'].mode().iloc[0]),'policy':str(g['task.policy'].mode().iloc[0]),
                'start_ns':int(g['observation.state.timestamp'].iloc[0]),'end_ns':int(g['observation.state.timestamp'].iloc[-1]),
                'video_path':f"videos/{cam}/chunk-{int(m[prefix+'chunk_index']):03d}/file-{int(m[prefix+'file_index']):03d}.mp4",'video_start':float(m[prefix+'from_timestamp']),'video_end':float(m[prefix+'to_timestamp']),
                'label_source':'speed_kmh/3.6 -> original5point derivative; negative compass-heading derivative -> positive-left yaw; sensor-derived weak labels'})
        print('SCANNED',p.name,len(candidates),flush=True)
    assert 3.3<np.median(ratios)<3.9,(np.quantile(ratios,[.1,.5,.9]))
    assert np.median(angles)<15,'Heading not aligned to GNSS compass bearing'
    selected=[];rng=np.random.default_rng(927500)
    for split,count in [('train',450),('val',50)]:
        pool=[r for r in candidates if r['split']==split];rng.shuffle(pool)
        sessions=Counter();cats={key:Counter() for key in ['road','weather','lighting','surface','speed_bin']};total=np.zeros(4);picked=[]
        cap=max(3,int(np.ceil(count/max(len(set(r['group'] for r in pool)),1)))+2)
        while len(picked)<count:
            best=None;bestscore=-1e20
            for r in pool:
                if sessions[r['group']]>=cap:continue
                if any(q['group']==r['group'] and not (r['start_ns']>q['end_ns']+30e9 or q['start_ns']>r['end_ns']+30e9) for q in picked):continue
                ac=np.asarray(r['accel_counts']);prop=(total+ac)/(total.sum()+ac.sum())
                if prop[2]>.40:continue
                quality=-12*float(((prop-np.array([.30,.30,.25,.15]))**2).sum())
                diversity=sum(1/(1+cats[k][r[k]]) for k in cats)+3/(1+sessions[r['group']])
                weak=min(r['weak_motion_frames']/r['frames'],.3)
                value=quality+diversity+weak
                if value>bestscore:best,bestscore=r,value
            if best is None:raise RuntimeError(f'Cannot fill {split}: {len(picked)}/{count}; do not silently relax diversity/constant limits')
            picked.append(best);pool.remove(best);total+=best['accel_counts'];sessions[best['group']]+=1
            for key in cats:cats[key][best[key]]+=1
        selected+=picked
    assert len(selected)==500 and len({r['episode_index'] for r in selected})==500
    assert not {r['group'] for r in selected if r['split']=='train'}&{r['group'] for r in selected if r['split']=='val'}
    summary={}
    for split in ['train','val']:
        rs=[r for r in selected if r['split']==split];ac=np.sum([r['accel_counts'] for r in rs],0)
        summary[split]={'episodes':len(rs),'frames':int(ac.sum()),'accel_counts':ac.tolist(),'accel_fraction':(ac/ac.sum()).tolist(),'sessions':len({r['group'] for r in rs}),'max_episodes_per_session':max(Counter(r['group'] for r in rs).values()),'distributions':{k:dict(Counter(str(r[k]) for r in rs)) for k in ['road','weather','lighting','surface','speed_bin','policy']}}
    save('selection_audit.json',{'summary':summary,'candidate_count':len(candidates),'rejected':dict(reject),'speed_raw_over_gps_quantiles':np.quantile(ratios,[.1,.5,.9]).tolist(),'median_absolute_heading_bearing_error_deg':float(np.median(angles)),'median_acceleration_x_correlation':float(np.nanmedian(corr)),'recipe':'450 train/50 val; session hash split before selection; 30s same-session buffer; <=40% CONSTANT per selected split; diverse sessions,roads,weather,lighting,speeds; labels fixed to existing thresholds','limitations':'GNSS and vehicle sensors corroborate units; not human ground truth. Existing validation repeatedly used for model selection.'})
    save('selected_episodes.json',{'config':cfg,'records':selected})
    print('SELECTION_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
