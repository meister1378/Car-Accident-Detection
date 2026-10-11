"""After all training: held-out evaluation, offline smoke, three verified ZIPs."""
import argparse, json, shutil, subprocess, sys, time, traceback
from pathlib import Path
import cv2
from package_experiments import prepare, finalize, digest
from external_nuscenes import evaluate


def main(root):
    root=Path(root);code=Path(__file__).resolve().parent
    run=root/'stage3_vjepa_runs/three_heads_20260917'
    assert set(json.loads((run/'comparison.json').read_text()))=={'tcn','fact','ltcontext'}
    def status(phase):
        (run/'delivery_status.json').write_text(json.dumps({'phase':phase,'time':time.time()}))
    status('external_evaluation')
    if not (run/'external_evaluation.json').exists():evaluate(root,run)
    status('staging')
    folder=root/'dacon_stage1/submissions/stage123_vjepa_20260916'
    base=folder/'submit_stage123_vjepa_20260916.zip'
    assert base.exists(),f'Need verified baseline ZIP: {base}'
    output=Path('/content/three_head_submissions');output.mkdir(exist_ok=True)
    prepare(base,run,output)
    fixture=Path('/content/three_head_smoke');fixture.mkdir(exist_ok=True)
    manifest=json.loads((run/'manifest.json').read_text())
    images=next(r['images'] for r in manifest['records'] if len(r['images'])>=64)
    for rel in ['s1/videos','s2/images/000001','s3/videos','s3_alone/videos']:(fixture/rel).mkdir(parents=True,exist_ok=True)
    for name,n in [('000001',64),('000002',17)]:
        dst=fixture/f's3/videos/{name}.mp4'
        writer=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,360))
        assert writer.isOpened()
        for i in range(n):
            im=cv2.imread(images[i+(10 if name=='000002' else 0)]);assert im is not None
            im=cv2.resize(im,(640,360));writer.write(im)
            if name=='000001':cv2.imwrite(str(fixture/f's2/images/000001/{i:06d}.jpg'),im)
        writer.release()
    shutil.copyfile(fixture/'s3/videos/000001.mp4',fixture/'s1/videos/000001.mp4')
    shutil.copyfile(fixture/'s3/videos/000002.mp4',fixture/'s3_alone/videos/000002.mp4')
    destination=root/'dacon_stage1/submissions/three_heads_20260917';destination.mkdir(parents=True,exist_ok=True)
    deliveries=[]
    for kind in ['tcn','fact','ltcontext']:
        status('offline_smoke_'+kind)
        staged=output/f'submit_stage123_{kind}_20260917'
        report=output/f'{kind}_smoke_report.json'
        subprocess.run([sys.executable,'-u',str(code/'smoke_three_heads.py'),'--out',str(staged),
                        '--fixtures',str(fixture),'--report',str(report),'--kind',kind],check=True)
        status('packaging_'+kind)
        archive=finalize(output,kind)
        dest=destination/archive.name
        shutil.copyfile(archive,dest);assert digest(dest)==digest(archive)
        shutil.copyfile(report,destination/report.name)
        deliveries.append({'kind':kind,'filename':dest.name,'bytes':dest.stat().st_size,'sha256':digest(dest)})
        print('SUBMISSION_ZIP_COPIED_VERIFIED',deliveries[-1],flush=True)
    for name in ['comparison.json','external_evaluation.json','external_split.json','split_contract.json','provenance.json']:
        shutil.copyfile(run/name,destination/name)
    (destination/'deliveries.json').write_text(json.dumps(deliveries,indent=2))
    status('all_zips_copied_pending_flush')
    from google.colab import drive
    drive.flush_and_unmount()
    print('THREE_SUBMISSION_ZIPS_ALL_FLUSHED',json.dumps(deliveries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    try:main(a.root)
    except Exception:
        err=traceback.format_exc();print(err,flush=True)
        dest=Path(a.root)/'stage3_vjepa_runs/three_heads_20260917/delivery_failure.json'
        dest.write_text(json.dumps({'error':err,'time':time.time()},indent=2))
        raise
