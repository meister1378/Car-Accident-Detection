"""Build Stage123 submissions using original V2 LTC package as immutable base."""
import json,shutil,zipfile,subprocess,sys,hashlib,csv
from pathlib import Path
import cv2
from diverse_v3_data import ROOT,OUT
from diverse_v2_data import write_json
from package_experiments import digest

def main():
    code=Path(__file__).parent;dest=ROOT/'dacon_stage1/submissions/diverse_v3_20260917';dest.mkdir(parents=True,exist_ok=True)
    work=Path('/content/diverse_v3_submissions');work.mkdir(exist_ok=True)
    base=ROOT/'dacon_stage1/submissions/diverse_v2_20260917/submit_stage123_v2_decoupled_ltc_20260917.zip'
    assert digest(base)=='278c402b4bf6f274642b5954152900ebbfe84d182d73a7a29300318396484a2d'
    fixtures=work/'fixtures'
    for rel in ['s1/videos','s2/images/000001','s3/videos','s3_alone/videos']:(fixtures/rel).mkdir(parents=True,exist_ok=True)
    records=json.loads((OUT/'manifest.json').read_text())['records']
    images=next(r['images'] for r in records if r.get('cache_origin')!='v2' and len(r['images'])>=80 and Path(r['images'][0]).exists())
    for name,n in [('000001',64),('000002',17)]:
        writer=cv2.VideoWriter(str(fixtures/f's3/videos/{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,360));assert writer.isOpened()
        for i in range(n):
            im=cv2.imread(images[i+(10 if name=='000002' else 0)]);assert im is not None
            im=cv2.resize(im,(640,360));writer.write(im)
            if name=='000001':cv2.imwrite(str(fixtures/f's2/images/000001/{i:06d}.jpg'),im)
        writer.release()
    shutil.copyfile(fixtures/'s3/videos/000001.mp4',fixtures/'s1/videos/000001.mp4')
    shutil.copyfile(fixtures/'s3/videos/000002.mp4',fixtures/'s3_alone/videos/000002.mp4')
    metrics=json.loads((OUT/'final_metrics.json').read_text());deliveries=[]
    for kind in ['ltc_scratch','ltc_finetune','ltc_ensemble']:
        staged=work/('submit_stage123_v3_'+kind+'_20260917');staged.mkdir(exist_ok=True)
        with zipfile.ZipFile(base) as z:
            for member in z.infolist():
                if member.is_dir():continue
                p=(staged/member.filename).resolve();assert p.is_relative_to(staged.resolve());p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as a,p.open('wb') as b:shutil.copyfileobj(a,b)
            shutil.copyfile(OUT/'export'/(kind+'.pt'),staged/'model/stage3/head.pt')
            for n in z.namelist():
                if (n=='inference.py' or n.startswith(('model/stage1/','model/stage2/'))) and not n.endswith('/'):
                    assert digest(staged/n)==hashlib.sha256(z.read(n)).hexdigest()
        write_json(staged/'experiment.json',{'version':3,'kind':kind,'metrics':metrics[kind],'datasets':['KITTI','A2D2','nuScenes','comma2k19'],'label_limitation':'sensor-derived weak-label F1, not contest F1'})
        report=work/(kind+'_smoke_report.json')
        subprocess.run([sys.executable,'-u',str(code/'smoke_three_heads.py'),'--out',str(staged),'--fixtures',str(fixtures),'--report',str(report),'--kind',kind],check=True)
        rr=json.loads(report.read_text());assert all(rr['tests'][str(s)]['passed'] for s in [1,2,3]) and rr['tests']['3']['per_video_reset_passed']
        assert rr['checkpoint_sha256']==digest(staged/'model/stage3/head.pt')
        target=work/(staged.name+'.zip');partial=target.with_suffix('.partial.zip')
        with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for p in sorted(staged.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(staged))
        with zipfile.ZipFile(partial) as z:assert z.testzip() is None and {'inference.py','requirements.txt','model/stage3/head.pt'}<=set(z.namelist())
        partial.replace(target);h=digest(target);shutil.copyfile(target,dest/target.name);assert digest(dest/target.name)==h
        shutil.copyfile(report,dest/report.name);deliveries.append({'kind':kind,'file':target.name,'bytes':target.stat().st_size,'sha256':h})
        print('V3_ZIP_VERIFIED',json.dumps(deliveries[-1]),flush=True)
    for name in ['final_metrics.json','baseline_metrics.json','split_contract.json','label_report.json','training_contract.json','comma_sensor_audit.json','nuscenes_sensor_audit.json']:shutil.copyfile(OUT/name,dest/name)
    rows=[]
    for kind,splits in metrics.items():
        for split,domains in splits.items():
            for d,m in domains.items():rows.append({'model':kind,'split':split,'dataset':d,'accel_F1':m['accel']['macro_f1'],'moving_steer_F1':m['moving_steer']['macro_f1']})
    with (dest/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (dest/'README_ko.txt').write_text('세 파일 모두 Stage1/2/3 포함. ZIP을 그대로 제출하세요. Stage1/2는 기존 v2 LTC 제출과 동일합니다.\nltc_scratch: 신규 초기화. ltc_finetune: v2 LTC에서 추가 학습. ltc_ensemble: 두 모델 동일 가중치 결합.\nKITTI/A2D2/nuScenes/comma2k19를 사용하며 기존 평가 분할은 유지하고 추가 데이터는 로그/차량+날짜별로 분리했습니다.\n내부 점수는 센서 임계값 라벨 기준으로 대회 점수가 아닙니다. 기존 v2 test는 재사용된 평가이고 새 로그 test와 구분해야 합니다.\nSTOPPED 정답 프레임은 조향 손실/평가에서 제외하고 출력에는 모두 포함합니다.\n오프라인 짧은 Stage1/2/3 검사 및 ZIP CRC를 통과했습니다. 전체 비공개 평가 시간은 미확인입니다.\n',encoding='utf-8')
    write_json(dest/'deliveries.json',deliveries);write_json(OUT/'status.json',{'phase':'THREE_ZIPS_PENDING_REMOTE_FLUSH','files':deliveries})
    print('V3_THREE_ZIPS_PENDING_REMOTE_FLUSH',flush=True)

if __name__=='__main__':main()
