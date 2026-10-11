"""Recover the omitted epoch-zero candidate; never train or alter old artifacts.

Repeat full development validation using a serialized normalization-only head.
Compare it to the recorded initial evaluation and all completed candidates.
Package it separately with the existing real-video/fresh-unpack checks.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import gc
import importlib.util
import json
import os
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import package_extended as package
from ltc_task_normalization import set_intermediate_normalization

DATA = package.DATA
OUT = DATA / 'initial_norm_submission'
package.OUT = OUT
package.WORK = Path('/content/package_initial_norm_20260927')
WARM_SHA = '11069af0aa0e2941d57d74b9445d1940e3b72de5830588177a92c0859463113f'


def run():
    OUT.mkdir(exist_ok=True)
    package.WORK.mkdir(exist_ok=True)
    lock = (package.WORK / 'initial_verification.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'complete.json').exists():
        done = json.loads((OUT / 'complete.json').read_text())
        assert package.digest(done['file']) == done['sha256']
        print('ALREADY_VERIFIED', flush=True)
        return
    assert not package.active_training(), 'Training must be finished'
    comparison = json.loads((DATA / 'norm_ab_w256_15ep/comparison.json').read_text())
    assert comparison['paired_sampling_checks_passed'] and len(comparison['paired_epochs']) == 15
    package.status('RECHECKING_INITIAL_NORMALIZATION')
    source = DATA.parent / 'diverse_v3_20260917'
    sys.path.insert(0, str(source / 'code'))
    from diverse_v2_models import probabilities
    prior_path = DATA.parent / 'v3_w256_B_15ep_20260926/train_three_experiments.py'
    spec = importlib.util.spec_from_file_location('initial_norm_prior', prior_path)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    warm_path = DATA / 'trained/head.pt'
    assert package.digest(warm_path) == WARM_SHA
    warm = torch.load(warm_path, map_location='cpu', weights_only=True)
    assert len(warm['members']) == 1 and 'neck_config' not in warm
    initial = copy.deepcopy(warm)
    initial['intermediate_normalization'] = 'taskwise_keep_aux'
    initial['members'][0]['kind'] = 'taskwise_ltc_keep_aux'
    initial['metrics'] = {'epoch': 0, 'optimizer_updates': 0,
                          'note': 'Preserved frozen head weights; only intermediate normalization changed.'}
    head = OUT / 'head_initial.pt'
    tmp = head.with_suffix('.tmp.pt')
    torch.save(initial, tmp)
    tmp.replace(head)
    loaded = torch.load(head, map_location='cpu', weights_only=True)
    original_state, candidate_state = warm['members'][0]['model'], loaded['members'][0]['model']
    assert original_state.keys() == candidate_state.keys()
    assert all(torch.equal(original_state[k], candidate_state[k]) for k in original_state)
    model = prior.make_wide().cuda()
    model.load_state_dict(candidate_state, strict=True)
    set_intermediate_normalization(model, loaded['intermediate_normalization'])
    model.eval()
    old = json.loads((source / 'manifest.json').read_text())
    new = json.loads((DATA / 'selected_episodes.json').read_text())
    assert old['config'] == new['config'] == loaded['config']
    records = [r for r in old['records'] if r['split'] in ['train', 'val']] + new['records']
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'val']
    assert not {(r['domain'], r['group']) for r in train} & {(r['domain'], r['group']) for r in val}
    assert len(val) == 209 and sum(len(r['accel']) for r in val) == 78223
    expected = json.loads((DATA / 'norm_ab_w256_15ep/taskwise_keep_aux/baseline_validation.json').read_text())
    results, counts = {}, {}
    started = time.time()
    with torch.inference_mode():
        for d in sorted({r['domain'] for r in val}):
            ats, sts, aps, sps = [], [], [], []
            for r in val:
                if r['domain'] != d:
                    continue
                path = DATA/'features'/(r['id']+'_v0.npz') if d == 'l2d' else prior.cache(r, 0)
                with np.load(path) as z:
                    features, accel, steer = z['features'], z['accel'], z['steer']
                    assert features.shape == (len(r['accel']), 3072) and np.isfinite(features).all()
                    assert np.array_equal(accel, r['accel']) and np.array_equal(steer, r['steer'])
                    a, s = probabilities(model, 'decoupled_ltc', features, 'cuda')
                    ats.append(accel.copy()); sts.append(steer.copy())
                    aps.append(a.numpy().argmax(1)); sps.append(s.numpy().argmax(1))
            a, s, ap, sp = map(np.concatenate, [ats, sts, aps, sps])
            moving = a != 3
            results[d] = {'accel': prior.metric(a, ap, 4),
                          'moving_steer': prior.metric(s[moving], sp[moving], 3)}
            counts[d] = {'records': len(ats), 'frames': len(a), 'moving_frames': int(moving.sum())}
            for task in ['accel', 'moving_steer']:
                assert results[d][task] == expected['validation'][d][task], (d, task)
            package.status('VALIDATED_DOMAIN', domain=d, **counts[d])
    score = float(np.mean([results[d][k]['macro_f1'] for d in results if d != 'l2d'
                           for k in ['accel', 'moving_steer']]))
    assert score == expected['selection_score']
    assert package.digest(warm_path) == WARM_SHA
    report = {'passed': True, 'selection_score': score, 'validation': results,
              'counts': counts, 'seconds': time.time()-started, 'optimizer_updates': 0,
              'head_sha256': package.digest(head), 'warm_head_sha256': WARM_SHA,
              'all_weights_bitwise_unchanged': True, 'all_recorded_full_validation_metrics_match': True,
              'manifest_sha256': package.digest(source/'manifest.json'),
              'l2d_selection_sha256': package.digest(DATA/'selected_episodes.json'),
              'limitations': 'Repeat of the same development validation with sensor-derived labels; not independent generalization evidence.'}
    package.save('initial_full_validation_recheck.json', report)
    candidates = json.loads((DATA/'extended_submission/selection.json').read_text())['candidates']
    candidate = {'variant': 'taskwise_keep_aux_initial0', 'epoch': 0, 'optimizer_updates': 0,
                 'score': score, 'validation': results, 'head': str(head), 'head_sha256': package.digest(head)}
    candidates.append(candidate)
    selected = max(candidates, key=lambda r: r['score'])
    assert selected == candidate, 'The initial candidate must actually win before separate packaging'
    package.save('selection.json', {'selected': selected, 'candidates': candidates,
                 'rule': 'Highest original4 mean8 development score including the previously omitted initial normalization-only candidate.',
                 'limitations': report['limitations'],
                 'previous_trained_delivery_preserved': str(DATA/'extended_submission/complete.json')})
    del model, warm, loaded, initial, original_state, candidate_state
    gc.collect(); torch.cuda.empty_cache()
    package.select_model = lambda: selected
    package.main(argparse.Namespace(prepare_only=False, wait=True))


if __name__ == '__main__':
    try:
        run()
    except Exception:
        package.save('package_failure.json', {'time': time.time(), 'traceback': traceback.format_exc()})
        raise
