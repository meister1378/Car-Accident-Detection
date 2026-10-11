"""Verify the intervention and unchanged original path on the actual LTC."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from ltc_task_normalization import intermediate_probabilities, set_intermediate_normalization

DATA = Path('/content/MyDrive/MyDrive/stage3_vjepa_runs/l2d500_w256_b48_20260927')
OUT = DATA / 'norm_ab_w256_15ep'


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(OUT / name)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location(
        'prior_train', DATA.parent / 'v3_w256_B_15ep_20260926/train_three_experiments.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    from diverse_v2_models import load_members
    try:
        load_members({'input_dim': 3072, 'members': [
            {'kind': 'taskwise_ltc_keep_aux', 'model': {}}]}, 'cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('Legacy loader accepted the treatment checkpoint kind')
    ck = torch.load(DATA / 'trained/head.pt', map_location='cpu', weights_only=True)
    model = prior.make_wide().cuda().eval()
    model.load_state_dict(ck['members'][0]['model'], strict=True)
    state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    x = torch.randn(2, 3072, 65, device='cuda')
    mask = torch.ones(2, 1, 65, dtype=torch.bool, device='cuda')
    mask[1, :, 49:] = False
    x *= mask
    with torch.no_grad():
        legacy = model(x, mask)
        set_intermediate_normalization(model, 'joint9')
        control = model(x, mask)
        assert torch.equal(legacy, control), 'Control changed the original LTC'
        set_intermediate_normalization(model, 'taskwise_keep_aux')
        treated = model(x, mask)
        assert torch.equal(legacy[0], treated[0]), 'First stage changed'
        assert not torch.allclose(legacy[-1], treated[-1]), 'Intervention had no effect'
        assert torch.isfinite(treated).all()
    assert list(state_before) == list(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state_before.items())

    # Independent mathematical oracle: conditioning the legacy distribution
    # within a task must equal softmax on that task's logits.
    z = torch.randn(2, 9, 13, device='cuda', requires_grad=True)
    p = z.softmax(1)
    q = intermediate_probabilities(z, 'taskwise_keep_aux')
    assert torch.allclose(q[:, :4], p[:, :4] / p[:, :4].sum(1, keepdim=True), atol=2e-7)
    assert torch.allclose(q[:, 4:7], p[:, 4:7] / p[:, 4:7].sum(1, keepdim=True), atol=2e-7)
    assert torch.equal(q[:, 7:], p[:, 7:]), 'Auxiliary channels changed'
    ga = torch.autograd.grad(q[:, 0].sum(), z, retain_graph=True)[0]
    gs = torch.autograd.grad(q[:, 4].sum(), z)[0]
    assert torch.count_nonzero(ga[:, 4:]) == 0
    assert torch.count_nonzero(gs[:, :4]) == 0 and torch.count_nonzero(gs[:, 7:]) == 0

    a = torch.arange(65, device='cuda').repeat(2, 1) % 4
    s = torch.arange(65, device='cuda').repeat(2, 1) % 3
    loss = prior.batched_loss(model(x, mask), a, s, mask,
                              torch.ones(4, device='cuda'), torch.ones(3, device='cuda'))
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    # Checkpoint reload must preserve the alternative forward, not silently use
    # the legacy implementation despite identical state_dict keys.
    replica = prior.make_wide().cuda().eval()
    set_intermediate_normalization(replica, 'taskwise_keep_aux')
    replica.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(replica(x, mask), treated)
    save('preflight.json', {
        'passed': True, 'legacy_control_bitwise_equal': True,
        'first_stage_bitwise_equal': True, 'later_stage_changed': True,
        'state_dict_and_parameters_unchanged': True, 'auxiliary_values_unchanged': True,
        'auxiliary_equality_scope': 'same input logits only',
        'legacy_loader_rejects_treatment_kind': True,
        'task_probability_cross_gradients_zero': True,
        'finite_nonzero_training_gradients': True, 'checkpoint_reload_equal': True,
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'warm_start_sha256': prior.sha(DATA / 'trained/head.pt'),
        'normalization_code_sha256': prior.sha(Path(__file__).with_name('ltc_task_normalization.py')),
        'loss': float(loss.detach()),
        'scope': 'Architecture and backward smoke on synthetic feature tensors; not an accuracy result.'})
    print('NORMALIZATION_PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('preflight_failure.json', {'traceback': traceback.format_exc()})
        raise
