"""Package the selected width256 head with the previously verified source Stage1/2."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
ROOT=Path('/content/MyDrive/MyDrive')
RUN=ROOT/'stage3_vjepa_runs/l2d500_w256_b48_20260927'
BASE=ROOT/'dacon_stage1/submissions/v3_w256_B_15ep_20260926/submit_stage123_w256_B_15ep_stage12_source_20260926.zip'
BASE_SHA='aa3f685e231b24653fd5609b5d5b9c129b63e95d1d4a70d5644522fcb15ea1c6'
OUT=RUN/'submission'
WORK=Path('/content/package_l2d500_20260927')
SMOKE=Path('/content/package_stage12_swap_20260926/smoke.py')
FIXTURES=Path('/content/package_wide192_20260926/fixtures')
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def extract(z,root):
    for i in z.infolist():assert (root/i.filename).resolve().is_relative_to(root.resolve())
    z.extractall(root)
def smoke(root,tag,expected_head_sha):
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('STAGE1_','STAGE2_','STAGE3_','L2D_')):env.pop(k)
    subprocess.run([sys.executable,str(SMOKE),'--out',str(root),'--fixtures',str(FIXTURES),'--report',str(OUT/(tag+'.json')),'--kind','B_width256_15ep'],check=True,cwd=root,env=env)
    report=json.loads((OUT/(tag+'.json')).read_text())
    assert report['checkpoint_sha256']==expected_head_sha
    assert all(report['tests'][str(i)]['passed'] for i in [1,2,3])
def main():
    if sys.flags.optimize:raise RuntimeError('Verification requires Python assertions enabled')
    assert (RUN/'training_complete.json').exists()
    history=json.loads((RUN/'trained/history.json').read_text())
    assert [r['epoch'] for r in history]==list(range(1,16))
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True)
    assert not (OUT/'complete.json').exists(),'Completed delivery preserved'
    assert digest(BASE)==BASE_SHA
    result=json.loads((RUN/'training_complete.json').read_text())
    saved={'selected':result['selected']}
    best=max(history,key=lambda r:r['selection_score'])
    assert saved['selected']['epoch']==best['epoch'] and saved['selected']['selection_score']==best['selection_score']
    head=RUN/'trained/head.pt';head_sha=digest(head);assert head_sha==result['head_sha256']
    with tempfile.TemporaryDirectory(prefix='stage_',dir=WORK) as tmp:
        stage=Path(tmp)
        with zipfile.ZipFile(BASE) as z:
            names=[i.filename for i in z.infolist() if not i.is_dir()]
            assert len(names)==len(set(names));extract(z,stage)
        preserved={n:digest(stage/n) for n in names if n!='model/stage3/head.pt'}
        shutil.copy2(head,stage/'model/stage3/head.pt')
        smoke(stage,'staged_smoke',head_sha)
        assert all(digest(stage/n)==h for n,h in preserved.items())
        local=WORK/'submit_stage123_l2d500_w256_b48_20260927.zip'
        with zipfile.ZipFile(local,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for n in sorted(names):z.write(stage/n,n)
    with zipfile.ZipFile(local) as z:
        assert set(z.namelist())==set(names) and len(z.namelist())==len(names)
        assert z.testzip() is None
        for n,h in (preserved|{'model/stage3/head.pt':head_sha}).items():
            hh=hashlib.sha256()
            with z.open(n) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):hh.update(b)
            assert hh.hexdigest()==h,n
        with tempfile.TemporaryDirectory(prefix='clean_',dir=WORK) as tmp:
            clean=Path(tmp);extract(z,clean);smoke(clean,'clean_archive_smoke',head_sha)
    destination=OUT/local.name;sha=digest(local);partial=destination.with_suffix('.zip.partial')
    shutil.copy2(local,partial);assert digest(partial)==sha;os.replace(partial,destination)
    report={'phase':'W256_SUBMISSION_VERIFIED','file':str(destination),'bytes':destination.stat().st_size,'sha256':sha,'head_sha256':head_sha,'selected_epoch':saved['selected']['epoch'],'validation_score':saved['selected']['selection_score'],'base_submission_sha256':BASE_SHA,'unchanged_files_sha256':preserved,'limitations':'Short synthetic Stage1/2/3 inference and archive checks; no hidden-test accuracy or full runtime claim.'}
    marker=OUT/'complete.json.tmp';marker.write_text(json.dumps(report,indent=2));os.replace(marker,OUT/'complete.json')
    print('W256_SUBMISSION_VERIFIED',json.dumps({k:v for k,v in report.items() if k!='unchanged_files_sha256'}),flush=True)
if __name__=='__main__':main()
