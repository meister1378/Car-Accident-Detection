"""Compare a packaged inference loader with the actual training definitions.

Uses cached L2D validation features only as numerical inputs, not for accuracy
claims. Short/padded inputs and a multi-window sequence exercise inference
shape, mask and context handling. The video smoke is a separate test.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main(args):
    data = Path(args.data)
    root = Path(args.root).resolve()
    stage3 = root / 'model/stage3'
    device = args.device
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(stage3 / 'head.pt', map_location='cpu', weights_only=True)
    assert len(saved['members']) == 1
    source = data.parent / 'diverse_v3_20260917/code'
    sys.path.insert(0, str(source))
    prior_path = data.parent / 'v3_w256_B_15ep_20260926/train_three_experiments.py'
    prior = module('verification_prior', prior_path)
    training_models = module('verification_training_models', source / 'diverse_v2_models.py')
    reference_sources = {str(prior_path): digest(prior_path)}
    if 'neck_config' in saved:
        script = data / 'neck4_w512_15ep/train_neck.py'
        tree = ast.parse(script.read_text())
        classes = [x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name in ('Neck', 'NeckModel')]
        assert {x.name for x in classes} == {'Neck', 'NeckModel'}
        scope = {'nn': nn, 'torch': torch, 'prior': prior}
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(script), 'exec'), scope)
        reference = scope['NeckModel']()
        reference_sources[str(script)] = digest(script)
    else:
        reference = prior.make_wide()
        mode = saved.get('intermediate_normalization', 'joint9')
        if mode != 'joint9':
            script = data / 'ltc_task_normalization.py'
            normalizer = module('verification_training_normalizer', script)
            normalizer.set_intermediate_normalization(reference, mode)
            reference_sources[str(script)] = digest(script)
    reference.load_state_dict(saved['members'][0]['model'], strict=True)
    reference.to(device).eval()
    sys.path.insert(0, str(stage3))
    for name in ['extended_ltc_runtime', 'ltc_task_normalization']:
        sys.modules.pop(name, None)
    packaged = module('verification_packaged_models', stage3 / 'diverse_v2_models.py')
    members = packaged.load_members(saved, device)
    kind, restored = members[0]
    assert kind == 'decoupled_ltc'
    assert reference.state_dict().keys() == restored.state_dict().keys()
    for key, value in reference.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    if 'extended_ltc_runtime' in sys.modules:
        assert Path(sys.modules['extended_ltc_runtime'].__file__).parent == stage3
    if saved.get('intermediate_normalization') == 'taskwise_keep_aux':
        assert Path(sys.modules['ltc_task_normalization'].__file__).parent == stage3
        for bad in [dict(saved, intermediate_normalization='joint9'),
                    {k: v for k, v in saved.items() if k != 'intermediate_normalization'}]:
            try:
                packaged.load_members(bad, device)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid taskwise metadata was silently accepted')
    selection = json.loads((data / 'selected_episodes.json').read_text())
    record = max((r for r in selection['records'] if r['split'] == 'val'),
                 key=lambda r: len(r['accel']))
    feature_file = data / 'features' / (record['id'] + '_v0.npz')
    with np.load(feature_file) as archive:
        base_features = archive['features'].astype(np.float32)
    # Tiling, if needed, is solely for exercising a 513-frame inference boundary.
    features = np.tile(base_features, (1 + 513 // len(base_features), 1))[:513]
    tests = []
    with torch.inference_mode():
        for length in [1, 17, 65, 256, 448, 513]:
            x = torch.as_tensor(features[:length].T[None].copy(), device=device)
            mask = torch.ones(1, 1, length, dtype=torch.bool, device=device)
            a, b = reference(x, mask), restored(x, mask)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
            tests.append({'frames': length, 'max_absolute_logit_difference': float((a-b).abs().max())})
        x = torch.as_tensor(np.stack([features[:65].T] * 2), device=device)
        mask = torch.ones(2, 1, 65, dtype=torch.bool, device=device)
        mask[1, :, 17:] = False
        x = x * mask
        a, b = reference(x, mask), restored(x, mask)
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        tests.append({'padded_lengths': [65, 17], 'max_absolute_logit_difference': float((a-b).abs().max())})
        a = training_models.probabilities(reference, 'decoupled_ltc', features, device)
        b = packaged.probabilities(restored, kind, features, device)
        for old, new in zip(a, b):
            assert torch.isfinite(new).all()
            torch.testing.assert_close(old, new, rtol=1e-6, atol=1e-6)
            assert torch.equal(old.argmax(-1), new.argmax(-1))
    report = {'passed': True, 'head_sha256': digest(stage3 / 'head.pt'),
              'reference_sources_sha256': reference_sources,
              'feature_source_sha256': digest(feature_file), 'tests': tests,
              'context_window_probabilities_and_labels_match': True,
              'device': device, 'limitations': 'Numerical loader equivalence, not independent accuracy evidence.'}
    Path(args.report).write_text(json.dumps(report, indent=2))
    print('PACKAGED_LOADER_EQUIVALENCE_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    main(p.parse_args())
