"""Run the authorized paired normalization experiment after the active neck run."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

DATA = Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
OUT = DATA / 'norm_ab_w256_15ep'
NECK = DATA / 'neck4_w512_15ep'
MODES = ('joint9', 'taskwise_keep_aux')


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(OUT / name)


def status(phase, **values):
    value = dict(phase=phase, time=time.time(), **values)
    save('queue_status.json', value)
    print(json.dumps(value), flush=True)


def processes_for(script):
    found = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            state = (p / 'stat').read_text().split(') ')[1].split()[0]
            if str(script).encode() in args and state != 'Z':
                found.append(int(p.name))
        except (OSError, IndexError):
            pass
    return found


def run_script(script, args, log):
    assert not processes_for(script), f'Already running: {script}'
    with log.open('a') as handle:
        child = subprocess.Popen([sys.executable, '-u', str(script), *args],
                                 cwd=DATA, stdout=handle, stderr=subprocess.STDOUT)
        status('RUNNING_CHILD', script=script.name, args=args, child_pid=child.pid)
        code = child.wait()
    if code:
        raise RuntimeError(f'{script.name} {args} exited {code}; inspect {log}')


def compare():
    histories = {m: json.loads((OUT / m / 'trained/history.json').read_text()) for m in MODES}
    contracts = {m: json.loads((OUT / m / 'training_contract.json').read_text()) for m in MODES}
    a, b = [contracts[m] for m in MODES]
    assert {k: v for k, v in a.items() if k != 'intermediate_normalization'} == {
        k: v for k, v in b.items() if k != 'intermediate_normalization'}
    assert all([r['epoch'] for r in histories[m]] == list(range(1, 16)) for m in MODES)
    pairs = []
    for control, treated in zip(*(histories[m] for m in MODES)):
        for key in ['schedule_sha256', 'first_augmented_batch_sha256',
                    'cuda_rng_start_sha256', 'cuda_rng_end_sha256', 'actual_sampling']:
            assert control[key] == treated[key], (control['epoch'], key)
        by_domain = {}
        for domain, val in control['validation'].items():
            by_domain[domain] = {}
            for task in ['accel', 'moving_steer']:
                other = treated['validation'][domain][task]
                by_domain[domain][task] = {
                    'control': val[task]['macro_f1'],
                    'taskwise': other['macro_f1'],
                    'control_per_class': val[task]['per_class_f1'],
                    'taskwise_per_class': other['per_class_f1']}
                item = by_domain[domain][task]
                item['delta'] = item['taskwise'] - item['control']
        pairs.append({'epoch': control['epoch'], 'control': control['selection_score'],
                      'taskwise': treated['selection_score'],
                      'delta': treated['selection_score'] - control['selection_score'],
                      'domains': by_domain})
    baseline = {m: json.loads((OUT / m / 'baseline_validation.json').read_text()) for m in MODES}
    best = {m: max(histories[m], key=lambda r: r['selection_score']) for m in MODES}
    result = {
        'phase': 'PAIRED_TRAINING_COMPLETE', 'paired_epochs': pairs, 'selected': best,
        'initial_validation': baseline, 'paired_sampling_checks_passed': True,
        'randomness_checks': 'All window identities/views match; each epoch first augmented batch and CUDA RNG endpoints match. Not a hash of every augmented tensor.',
        'metric': 'Unweighted mean of acceleration4 and moving-steering3 class MacroF1 across original4 domains; L2D separately.',
        'limitations': 'One seed; reused development validation and sensor-derived weak labels; no independent hidden-test claim. Taskwise normalization also changes group amplitude; the two reserved channels retain joint normalization.'}
    save('comparison.json', result)
    save('complete.json', {'phase': 'PAIRED_TRAINING_COMPLETE', 'time': time.time(),
                          'best': {m: {'epoch': r['epoch'], 'score': r['selection_score']}
                                   for m, r in best.items()}})


def main():
    lock = open('/content/normalization_ab_20260927.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'complete.json').exists():
        return
    while not (NECK / 'training_complete.json').exists():
        if (NECK / 'training_failure.json').exists():
            raise RuntimeError('Neck training failed; inspect before continuing the queue')
        pids = processes_for(NECK / 'train_neck.py')
        if not pids:
            raise RuntimeError('Neck is incomplete but its training process is absent')
        status('WAITING_FOR_NECK', neck_pids=pids)
        time.sleep(30)
    while processes_for(NECK / 'train_neck.py'):
        time.sleep(2)
    report = OUT / 'preflight.json'
    module_sha = hashlib.sha256((DATA / 'ltc_task_normalization.py').read_bytes()).hexdigest()
    if not report.exists() or json.loads(report.read_text()).get('normalization_code_sha256') != module_sha:
        run_script(DATA / 'check_norm_ab.py', [], OUT / 'preflight.log')
    assert json.loads(report.read_text())['passed']
    for mode in MODES:
        folder = OUT / mode
        folder.mkdir(exist_ok=True)
        if (folder / 'training_complete.json').exists():
            continue
        run_script(DATA / 'train_norm_ab.py', [mode], folder / 'training.log')
        assert (folder / 'training_complete.json').exists()
    compare()
    status('PAIRED_TRAINING_COMPLETE')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('queue_failure.json', {'time': time.time(), 'traceback': traceback.format_exc()})
        raise
