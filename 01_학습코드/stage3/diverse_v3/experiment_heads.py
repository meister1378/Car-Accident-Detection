"""Official FACT/LTContext adapters. Joint labels omit steering for STOPPED.

0..8 = accel(0..2)*3 + steer(0..2); 9 = STOPPED.
Compact configs are adaptations, not reproductions of published benchmark scores.
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'vendor'))
sys.path.insert(0, str(BASE/'vendor/LTContext'))


def make_head(kind, dim):
    if kind == 'tcn':
        return JointTCN(dim)
    if kind == 'fact':
        from FACT.configs.default import get_cfg_defaults
        from FACT.models.blocks import FACT
        from FACT.models.loss import MatchCriterion
        c = get_cfg_defaults()
        c.merge_from_file(str(BASE/'vendor/FACT/configs/breakfast.yaml'))
        c.FACT.block = 'iuu'
        c.FACT.ntoken = 20
        c.FACT.trans = False
        c.FACT.cmr = .1
        c.Bi.hid_dim = c.Bi.f_dim = c.Bi.a_dim = 128
        c.Bi.a_ffdim = 256
        c.Bi.a_layers = 2
        c.Bi.a_nhead = 4
        c.Bi.f_layers = 6
        c.Bi.dropout = .1
        c.Bu.f_layers = 6
        c.Bu.a_nhead = 4
        c.Loss.match = 'o2m'
        c.Loss.nullw = .5
        c.Loss.sw = .05
        c.TM.use = False
        m = FACT(c, dim, 10)
        m.mcriterion = MatchCriterion(c, 10, bg_ids=[])
        return m
    if kind == 'ltcontext':
        from ltc.model.ltcontext import LTC
        c = NS(INPUT_DIM=dim, NUM_CLASSES=10,
               ATTENTION=NS(NUM_ATTN_HEADS=1, DROPOUT=.1),
               LTC=NS(NUM_LAYERS=6, MODEL_DIM=64, NUM_STAGES=3,
                      DIM_REDUCTION=2, CONV_DILATION_FACTOR=2,
                      WINDOWED_ATTN_W=32, LONG_TERM_ATTN_G=32,
                      USE_INSTANCE_NORM=True, DROPOUT_PROB=.1,
                      CHANNEL_MASKING_PROB=.1))
        return LTC(c)
    raise ValueError(kind)


class JointTCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Conv1d(dim, 128, 1)
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=d, dilation=d),
            nn.GELU(), nn.Dropout(.1)) for d in [1, 2, 4, 8]])
        self.output = nn.Conv1d(128, 10, 1)

    def forward(self, x):
        x = self.project(self.norm(x).T[None])
        for block in self.layers:
            x = x + block(x)
        return self.output(x)[0].T


def loss_and_predict(model, kind, x, y=None):
    """One independent sequence, T,D. Inference never receives ground truth."""
    if kind == 'fact':
        model._forward_one_video(x[:, None], transcript=None)
        pred = model.block_list[-1].eval(None)
        loss = model._loss_one_video(y) if y is not None else None
    else:
        if kind == 'ltcontext':
            out = model(x.T[None], torch.ones(1, 1, len(x), device=x.device, dtype=torch.bool))
            logits = [stage[0].T for stage in out]
        else:
            logits = [model(x)]
        pred = logits[-1].argmax(-1)
        loss = sum(F.cross_entropy(z, y) for z in logits)/len(logits) if y is not None else None
    return loss, pred


def encode_labels(a, s):
    return torch.where(a == 3, 9, a*3+s)


def decode_labels(j):
    return torch.where(j == 9, 3, j//3), torch.where(j == 9, 1, j%3)


def predict_sequence(model, kind, features, device, core=256, context=64):
    pieces = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), core):
            end = min(start+core, len(features))
            left, right = max(0, start-context), min(len(features), end+context)
            x = torch.as_tensor(features[left:right], dtype=torch.float32, device=device)
            _, pred = loss_and_predict(model, kind, x)
            pieces.append(pred[start-left:end-left].cpu())
    return torch.cat(pieces)
