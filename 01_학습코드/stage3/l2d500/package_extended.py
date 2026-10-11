"""Prepare/check the new LTC loader and deliver one validation-selected ZIP.

--prepare-only tests the completed neck checkpoint on CPU while the paired
normalization training runs. --wait waits for both paired runs, chooses among
the preserved frozen baseline, neck, joint9 and taskwise runs by the existing
original-four-domain mean8 development MacroF1, then packages only the winner.
If the preserved frozen model wins, its already verified ZIP is reused without
modification. No claim about hidden-test performance follows from selection.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

DATA = Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
BASE = DATA / 'submission/submit_stage123_l2d500_w256_b48_20260927.zip'
BASE_SHA = 'f29403c5052c881d340dbd302b37432d85893f0cccb71d0b9352b974b3eac1fe'
FACTORY_SHA = 'd9582c0fe96a05b4e68a6ecc2214df5899a4c320d1aad917919230a45c5baa16'
NORMALIZER_SHA = '69ada4a0f3b1474c3cec388d53071a795e76e9e27f542897297598e404ad73bb'
OUT = DATA / 'extended_submission'
WORK = Path('/content/package_extended_20260927')
NEW_FILES = ['model/stage3/extended_ltc_runtime.py', 'model/stage3/ltc_task_normalization.py']
CHANGED_FILES = ['model/stage3/diverse_v2_models.py', 'model/stage3/head.pt']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(p)


def status(phase, **kw):
    value = dict(phase=phase, time=time.time(), **kw)
    save('package_status.json', value)
    print(json.dumps(value), flush=True)


def active_training():
    paths = [DATA / 'neck4_w512_15ep/train_neck.py', DATA / 'train_norm_ab.py',
             DATA / 'continue_norm_ab.py']
    found = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if any(str(x).encode() in args for x in paths):
                if (p / 'stat').read_text().split(') ')[1].split()[0] != 'Z':
                    found.append(int(p.name))
        except (OSError, IndexError):
            pass
    return found


def extract(archive, root):
    infos = archive.infolist()
    assert len(infos) == len({i.filename for i in infos}), 'Duplicate archive member'
    for i in infos:
        assert (root / i.filename).resolve().is_relative_to(root.resolve())
        assert ((i.external_attr >> 16) & 0o170000) != 0o120000, 'Symlink member'
    archive.extractall(root)


def stage(root, head):
    assert digest(BASE) == BASE_SHA
    with zipfile.ZipFile(BASE) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        extract(archive, root)
    assert not set(NEW_FILES) & set(names)
    preserved = {n: digest(root/n) for n in names if n not in CHANGED_FILES}
    factory = root / CHANGED_FILES[0]
    assert digest(factory) == FACTORY_SHA
    old = """def load_members(saved,device):
    members=[]
    for item in saved['members']:
        m=make_model(item['kind'],saved['input_dim'],saved.get('model_dim',96)).to(device).eval()
        m.load_state_dict(item['model'],strict=True);members.append((item['kind'],m))
    return members
"""
    new = """def load_members(saved,device):
    from extended_ltc_runtime import load_extended_member
    members=[]
    for item in saved['members']:
        if item['kind'] in ('decoupled_ltc','taskwise_ltc_keep_aux'):
            members.append(load_extended_member(saved,item,make_model,device))
        else:
            if 'neck_config' in saved or saved.get('intermediate_normalization','joint9')!='joint9':
                raise ValueError('Extended metadata cannot be applied to this member')
            m=make_model(item['kind'],saved['input_dim'],saved.get('model_dim',96)).to(device).eval()
            m.load_state_dict(item['model'],strict=True);members.append((item['kind'],m))
    return members
"""
    text = factory.read_text()
    assert text.count(old) == 1
    factory.write_text(text.replace(old, new))
    assert digest(DATA/'ltc_task_normalization.py') == NORMALIZER_SHA
    for name in NEW_FILES:
        shutil.copy2(DATA/Path(name).name, root/name)
    shutil.copy2(head, root/'model/stage3/head.pt')
    return sorted(names + NEW_FILES), preserved


def worker(script, args, root, tag, env=None):
    with (OUT/(tag+'.log')).open('a') as log:
        subprocess.run([sys.executable, '-B', '-u', str(script), *args],
                       cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       env=env, check=True)


def equivalence(root, tag):
    report = OUT/(tag+'.json')
    worker(DATA/'verify_extended_ltc.py', ['--root', str(root), '--data', str(DATA),
           '--report', str(report), '--device', 'cpu'], root, tag)
    value = json.loads(report.read_text())
    assert value['passed'] and value['head_sha256'] == digest(root/'model/stage3/head.pt')


def fixtures():
    import cv2
    import numpy as np
    directory = WORK/'fixtures'
    for name in ['s1/videos', 's2/images/000001', 's3/videos', 's3_alone/videos']:
        (directory/name).mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:384, :640]
    for video, length in [('000001', 64), ('000002', 17)]:
        writer = cv2.VideoWriter(str(directory/'s3/videos'/f'{video}.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'), 10., (640, 384))
        assert writer.isOpened()
        for i in range(length):
            frame = np.stack([(xx+i*5)%256, (yy+i*3)%256, ((xx+yy)//2+i*7)%256], -1).astype(np.uint8)
            cv2.rectangle(frame, (20+i*3, 180), (130+i*3, 260), (20, 240, 60), -1)
            writer.write(frame)
            if video == '000001':
                assert cv2.imwrite(str(directory/'s2/images/000001'/f'{i:06d}.jpg'), frame)
        writer.release()
    shutil.copy2(directory/'s3/videos/000001.mp4', directory/'s1/videos/000001.mp4')
    shutil.copy2(directory/'s3/videos/000002.mp4', directory/'s3_alone/videos/000002.mp4')
    return directory


def smoke(root, fixtures_path, tag):
    dependency = Path('/content/extended_submission_timm1015')
    if not (dependency/'timm').exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--no-deps', '--target', str(dependency), 'timm==1.0.15'], check=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('STAGE1_', 'STAGE2_', 'STAGE3_', 'L2D_')):
            env.pop(key)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '4'
    env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
    env['SUBMISSION_TIMM_DIR'] = str(dependency)
    report = OUT/(tag+'.json')
    worker(DATA/'smoke_extended_submission.py', ['--out', str(root), '--fixtures', str(fixtures_path),
           '--report', str(report), '--kind', 'selected_extended_ltc'], root, tag, env)
    value = json.loads(report.read_text())
    assert value['checkpoint_sha256'] == digest(root/'model/stage3/head.pt')
    assert all(value['tests'][str(i)]['passed'] for i in [1, 2, 3])


def select_model():
    paths = {'frozen': DATA, 'neck4': DATA/'neck4_w512_15ep',
             'joint9': DATA/'norm_ab_w256_15ep/joint9',
             'taskwise_keep_aux': DATA/'norm_ab_w256_15ep/taskwise_keep_aux'}
    rows = []
    for name, path in paths.items():
        complete = json.loads((path/'training_complete.json').read_text())
        history = json.loads((path/'trained/history.json').read_text())
        assert [r['epoch'] for r in history] == list(range(1, 16))
        row = max(history, key=lambda r: r['selection_score'])
        assert row == complete['selected']
        head = path/'trained/head.pt'
        assert digest(head) == complete['head_sha256']
        rows.append({'variant': name, 'head': str(head), 'head_sha256': digest(head),
                     'epoch': row['epoch'], 'score': row['selection_score'],
                     'validation': row['validation']})
    chosen = max(rows, key=lambda r: r['score'])
    save('selection.json', {'selected': chosen, 'candidates': rows,
                           'rule': 'Highest original4 mean8 development MacroF1; exact ties prefer preserved baseline.',
                           'limitations': 'Reused development validation, single seeds and sensor-derived labels. The neck and earlier frozen runs do not form a matched normalization A/B. No hidden-test improvement claim.'})
    return chosen


def main(args):
    if sys.flags.optimize:
        raise RuntimeError('Assertions must remain enabled')
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    lock = (WORK/'package.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.prepare_only:
        status('CPU_LOADER_PREFLIGHT')
        head = DATA/'neck4_w512_15ep/trained/head.pt'
        assert (head.parent.parent/'training_complete.json').exists()
        with tempfile.TemporaryDirectory(prefix='preflight_', dir=WORK) as tmp:
            root = Path(tmp)
            names, preserved = stage(root, head)
            equivalence(root, 'neck_loader_preflight')
            assert all(digest(root/n) == h for n, h in preserved.items())
        status('CPU_LOADER_PREFLIGHT_PASSED')
        return
    if (OUT/'complete.json').exists():
        result = json.loads((OUT/'complete.json').read_text())
        assert digest(result['file']) == result['sha256']
        return
    while not (DATA/'norm_ab_w256_15ep/complete.json').exists() or active_training():
        for failure in [DATA/'norm_ab_w256_15ep/queue_failure.json',
                        DATA/'neck4_w512_15ep/training_failure.json']:
            if failure.exists():
                raise RuntimeError(f'Upstream failure: {failure}')
        pids = active_training()
        if not pids and not (DATA/'norm_ab_w256_15ep/complete.json').exists():
            raise RuntimeError('Training is incomplete but no queue/trainer process remains')
        status('WAITING_FOR_PAIRED_TRAINING', pids=pids)
        time.sleep(30)
    assert json.loads((DATA/'norm_ab_w256_15ep/comparison.json').read_text())['paired_sampling_checks_passed']
    chosen = select_model()
    if chosen['variant'] == 'frozen':
        assert digest(BASE) == BASE_SHA
        original = json.loads((DATA/'submission/complete.json').read_text())
        assert original['sha256'] == BASE_SHA and original['head_sha256'] == chosen['head_sha256']
        verified = json.loads((DATA/'submission/clean_archive_smoke.json').read_text())
        assert verified['checkpoint_sha256'] == chosen['head_sha256']
        assert all(verified['tests'][str(i)]['passed'] for i in [1, 2, 3])
        save('complete.json', {'phase': 'PRESERVED_FROZEN_ZIP_SELECTED', 'file': str(BASE),
                              'sha256': BASE_SHA, 'selected': chosen,
                              'reused_smoke': str(DATA/'submission/clean_archive_smoke.json'),
                              'note': 'No new archive or new inference claim; reused unchanged previously verified submission.'})
        status('PRESERVED_FROZEN_ZIP_SELECTED')
        return
    status('PACKAGING', selected=chosen['variant'], epoch=chosen['epoch'])
    fixture_path = fixtures()
    destination = OUT/f"submit_stage123_{chosen['variant']}_selected_20260927.zip"
    assert not destination.exists(), 'Existing delivery must not be overwritten'
    temporary = WORK/(destination.name+'.partial')
    with tempfile.TemporaryDirectory(prefix='staged_', dir=WORK) as tmp:
        root = Path(tmp)
        names, preserved = stage(root, Path(chosen['head']))
        assert digest(root/'model/stage3/head.pt') == chosen['head_sha256']
        equivalence(root, 'staged_loader_equivalence')
        smoke(root, fixture_path, 'staged_smoke')
        assert all(digest(root/n) == h for n, h in preserved.items())
        expected = {n: digest(root/n) for n in names}
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for name in names:
                archive.write(root/name, name)
    with zipfile.ZipFile(temporary) as archive:
        assert sorted(archive.namelist()) == names
        assert archive.testzip() is None
        for name, sha in expected.items():
            h = hashlib.sha256()
            with archive.open(name) as handle:
                for block in iter(lambda: handle.read(8*1024*1024), b''):
                    h.update(block)
            assert h.hexdigest() == sha, name
        with tempfile.TemporaryDirectory(prefix='fresh_', dir=WORK) as tmp:
            root = Path(tmp)
            extract(archive, root)
            equivalence(root, 'fresh_loader_equivalence')
            smoke(root, fixture_path, 'fresh_archive_smoke')
            assert all(digest(root/n) == sha for n, sha in expected.items())
    sha = digest(temporary)
    transfer = destination.with_suffix('.zip.partial')
    shutil.copy2(temporary, transfer)
    assert digest(transfer) == sha
    transfer.replace(destination)
    save('complete.json', {'phase': 'SELECTED_SUBMISSION_VERIFIED', 'file': str(destination),
                          'sha256': sha, 'bytes': destination.stat().st_size, 'selected': chosen,
                          'base_sha256': BASE_SHA, 'unchanged_files_sha256': preserved,
                          'archive_files_sha256': expected,
                          'limitations': 'Synthetic structural video checks and numerical loader equivalence; no hidden-test accuracy or full runtime claim.'})
    status('SELECTED_SUBMISSION_VERIFIED', file=str(destination), sha256=sha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        save('package_failure.json', {'time': time.time(), 'traceback': traceback.format_exc()})
        raise
