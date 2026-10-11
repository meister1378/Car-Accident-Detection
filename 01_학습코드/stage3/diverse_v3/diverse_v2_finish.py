"""Three complete stage123 ZIPs; smoke, CRC and remote hashes gate completion."""
import argparse,json,shutil,subprocess,sys,zipfile,time,csv
from pathlib import Path
from package_experiments import digest
from diverse_v2_data import write_json

def main(root):
    root=Path(root);code=Path(__file__).parent;run=root/'stage3_vjepa_runs/diverse_v2_20260917'
    metrics=json.loads((run/'final_metrics.json').read_text())
    dest=root/'dacon_stage1/submissions/diverse_v2_20260917';dest.mkdir(parents=True,exist_ok=True)
    work=Path('/content/diverse_v2_submissions');work.mkdir(exist_ok=True)
    baseline=root/'dacon_stage1/submissions/three_heads_20260917/submit_stage123_tcn_20260917.zip'
    fixtures=Path('/content/three_head_smoke');assert (fixtures/'s3/videos/000001.mp4').exists()
    deliveries=[];rows=[]
    for kind in ['balanced_tcn','decoupled_ltc','ensemble']:
        out=work/f'submit_stage123_v2_{kind}_20260917';out.mkdir(exist_ok=True)
        with zipfile.ZipFile(baseline) as z:
            for i in z.infolist():
                if i.is_dir():continue
                p=(out/i.filename).resolve();assert p.is_relative_to(out.resolve())
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(i) as f,p.open('wb') as g:shutil.copyfileobj(f,g)
            s3=out/'model/stage3'
            shutil.copyfile(run/'export'/(kind+'.pt'),s3/'head.pt')
            shutil.copyfile(code/'diverse_v2_adapter.py',s3/'adapter.py')
            shutil.copyfile(code/'diverse_v2_models.py',s3/'diverse_v2_models.py')
            for n in z.namelist():
                if (n=='inference.py' or n.startswith(('model/stage1/','model/stage2/'))) and not n.endswith('/'):
                    import hashlib
                    assert digest(out/n)==hashlib.sha256(z.read(n)).hexdigest()
        write_json(out/'experiment.json',{'kind':kind,'metrics':metrics[kind],'data':['KITTI','nuScenes','A2D2'],
            'limitations':'sensor-derived labels; domain-held-out test; not contest score; short offline smoke only'})
        report=work/(kind+'_smoke_report.json')
        subprocess.run([sys.executable,'-u',str(code/'smoke_three_heads.py'),'--out',str(out),'--fixtures',str(fixtures),'--report',str(report),'--kind',kind],check=True)
        rr=json.loads(report.read_text());assert all(rr['tests'][str(s)]['passed'] for s in [1,2,3])
        assert rr['tests']['3']['per_video_reset_passed'] and rr['checkpoint_sha256']==digest(s3/'head.pt')
        target=work/(out.name+'.zip');partial=target.with_suffix('.partial.zip')
        with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(out))
        with zipfile.ZipFile(partial) as z:
            assert z.testzip() is None and {'inference.py','requirements.txt','model/stage3/head.pt'}<=set(z.namelist())
        partial.replace(target);h=digest(target);shutil.copyfile(target,dest/target.name);assert digest(dest/target.name)==h
        shutil.copyfile(report,dest/report.name)
        deliveries.append({'kind':kind,'file':target.name,'bytes':target.stat().st_size,'sha256':h})
        for split,domains in metrics[kind].items():
            for domain,mm in domains.items():rows.append({'model':kind,'split':split,'dataset':domain,'accel_F1':mm['accel']['macro_f1'],'moving_steer_F1':mm['moving_steer']['macro_f1']})
        print('V2_COMPLETE_ZIP_VERIFIED',json.dumps(deliveries[-1]),flush=True)
    for name in ['final_metrics.json','split_contract.json','label_report.json','training_contract.json']:
        shutil.copyfile(run/name,dest/name)
    with (dest/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (dest/'README_ko.txt').write_text('세 ZIP 모두 Stage1/2/3를 포함합니다. 압축을 다시 만들지 말고 그대로 제출하세요.\n학습 및 모델 선택용 검증에 KITTI, nuScenes, A2D2를 모두 사용하며 주행 기록을 분리했습니다.\n최종 시험은 별도 nuScenes 로그와 A2D2 주행입니다. 최종 시험 점수는 모델 선택에 사용하지 않았습니다.\n센서 임계값으로 만든 라벨 기준 F1이며 대회 성능을 보장하지 않습니다.\nSTOPPED 정답 프레임은 조향 손실과 평가에서 제외하되 모든 프레임 steer_label을 출력합니다.\n오프라인 짧은 실행 검사를 통과했으며 전체 비공개 평가 시간 및 PyTorch 2.8 환경은 별도 검증하지 못했습니다.\n',encoding='utf-8')
    write_json(dest/'deliveries.json',deliveries);write_json(run/'status.json',{'phase':'THREE_NEW_ZIPS_COPIED_PENDING_FLUSH','time':time.time(),'files':deliveries})
    print('V2_THREE_ZIPS_COPIED_PENDING_FLUSH',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
