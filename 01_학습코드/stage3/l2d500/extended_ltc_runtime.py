"""Inference support for the authorized neck and taskwise LTContext experiments.

The factory passes B x 3072 x T features and B x 1 x T boolean masks. The
wrapper returns the unchanged stage x B x 9 x T logits. Channel groups are
acceleration0:4, steering4:7, reserved7:9. Normalization mode is checkpoint
metadata, never inferred from a weight tensor or silently defaulted for the
taskwise kind. This file adds no online adaptation or persistent video state.
"""
import torch
from torch import nn
from ltc_task_normalization import set_intermediate_normalization


class TemporalNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        expected = {'width': 512, 'depth': 4, 'kernel': 3,
                    'dilations': [1, 2, 4, 8]}
        if config != expected:
            raise ValueError(f'Unsupported neck architecture: {config}')
        self.norm = nn.LayerNorm(3072)
        self.down = nn.Conv1d(3072, 512, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(512, 512, 3, padding=d, dilation=d)
            for d in config['dilations']])
        self.up = nn.Conv1d(512, 3072, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, mask):
        z = self.down(self.norm(x.transpose(1, 2)).transpose(1, 2)) * mask
        for block in self.blocks:
            z = (z + torch.nn.functional.gelu(block(z))) * mask
        return (x + self.up(z)) * mask


class NeckLTC(nn.Module):
    def __init__(self, head, config):
        super().__init__()
        self.neck = TemporalNeck(config)
        self.head = head

    def forward(self, x, mask):
        return self.head(self.neck(x, mask), mask)


def load_extended_member(saved, item, make_model, device):
    """Return a strict-loaded LTC member and the existing masked-forward kind."""
    kind = item['kind']
    if kind not in ('decoupled_ltc', 'taskwise_ltc_keep_aux'):
        raise ValueError(kind)
    mode = saved.get('intermediate_normalization', 'joint9')
    expected = 'taskwise_keep_aux' if kind == 'taskwise_ltc_keep_aux' else 'joint9'
    if mode != expected:
        raise ValueError(f'Checkpoint kind/mode mismatch: {kind}/{mode}')
    if kind == 'taskwise_ltc_keep_aux' and 'intermediate_normalization' not in saved:
        raise ValueError('Taskwise checkpoint requires explicit normalization metadata')
    head = make_model('decoupled_ltc', saved['input_dim'], saved.get('model_dim', 96))
    if 'neck_config' in saved:
        if saved['input_dim'] != 3072 or saved.get('model_dim') != 256 or mode != 'joint9':
            raise ValueError('Unsupported combined neck/mode/dimension configuration')
        model = NeckLTC(head, saved['neck_config'])
    else:
        model = head
        if mode == 'taskwise_keep_aux':
            set_intermediate_normalization(model, mode)
    model.load_state_dict(item['model'], strict=True)
    return 'decoupled_ltc', model.to(device).eval()
