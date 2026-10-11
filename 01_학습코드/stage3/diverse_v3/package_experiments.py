"""Stage three complete submissions; final ZIP requires per-kind smoke reports.

Run on Colab after training. Original baseline archive stays untouched.
prepare(base_zip, run_dir, output_dir) -> directories to smoke-test.
finalize(output_dir, kind) -> ZIP only after an externally written passing report.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(base_zip, run_dir, output_dir):
    import torch
    from experiment_heads import make_head
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    comparison = json.loads((run_dir/'comparison.json').read_text())
    assert set(comparison)=={'tcn','fact','ltcontext'}, 'All experiments must finish'
    with zipfile.ZipFile(base_zip) as z:
        assert z.testzip() is None
        assert {'inference.py','requirements.txt'} <= set(z.namelist())
        for kind in ['tcn','fact','ltcontext']:
            checkpoint = run_dir/kind/'head.pt'
            saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert saved['kind']==kind and saved['joint_labels']
            assert saved['metrics']['epoch']>=1
            model = make_head(kind,saved['input_dim'])
            model.load_state_dict(saved['model'],strict=True)
            del model
            out = output_dir/f'submit_stage123_{kind}_20260917'
            out.mkdir(exist_ok=True)
            for info in z.infolist():
                if info.is_dir():continue
                target=(out/info.filename).resolve()
                assert target.is_relative_to(out.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
            s3=out/'model/stage3'
            shutil.copyfile(checkpoint,s3/'head.pt')
            shutil.copyfile(BASE/'adapter_three_heads.py',s3/'adapter.py')
            shutil.copyfile(BASE/'experiment_heads.py',s3/'experiment_heads.py')
            for name in ['FACT','LTContext']:
                shutil.copytree(BASE/'vendor'/name,s3/'vendor'/name,dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('.git','__pycache__'))
            # Stage1/2 require original transformer version. Other dependencies
            # must be recorded with actual versions after Colab smoke testing.
            req=(out/'requirements.txt').read_text()
            assert 'transformers==5.16.1' in req
            (out/'requirements.txt').write_text(req+'\neinops\nyacs\nscipy\nsimplejson\n')
            # Verify stage1/2 and root entry point byte-for-byte against baseline.
            for name in z.namelist():
                if name=='inference.py' or name.startswith(('model/stage1/','model/stage2/')):
                    if not name.endswith('/'):
                        assert digest(out/name)==hashlib.sha256(z.read(name)).hexdigest()
            (out/'experiment.json').write_text(json.dumps({
                'kind':kind,'checkpoint_sha256':digest(checkpoint),
                'metrics':saved['metrics'],'label_type':'sensor-derived proxy',
                'state':'STAGED_NOT_YET_SMOKE_VERIFIED'},indent=2))
            print('STAGED_FOR_SMOKE',out,flush=True)


def finalize(output_dir, kind):
    assert kind in ['tcn','fact','ltcontext']
    output_dir=Path(output_dir)
    out=output_dir/f'submit_stage123_{kind}_20260917'
    report_path=output_dir/f'{kind}_smoke_report.json'
    report=json.loads(report_path.read_text())
    assert report['kind']==kind
    assert report['checkpoint_sha256']==digest(out/'model/stage3/head.pt')
    assert report['offline_network_blocked'] is True
    assert all(report['tests'][str(s)]['passed'] for s in [1,2,3])
    assert report['tests']['3']['per_video_reset_passed']
    assert report['tests']['3']['schema_and_frame_counts_passed']
    target=output_dir/(out.name+'.zip')
    partial=target.with_suffix('.partial.zip')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(out))
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        assert 'inference.py' in z.namelist()
        assert 'model/stage3/head.pt' in z.namelist()
    partial.replace(target)
    (target.with_suffix('.sha256')).write_text(digest(target)+'  '+target.name+'\n')
    return target
