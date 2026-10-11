"""추가 시계열 헤드 (학습·제출 런타임 공용). 입력 (T, 3072) -> 출력 리스트[(T, 9)] (0:4 가감속, 4:7 조향, 7 가속도 회귀, 8 속도 회귀).
bigru: 양방향 GRU / tfm: 합성곱 위치부호 + Transformer 인코더 / mstcn: 다단계 팽창 TCN (MS-TCN)"""
import torch
from torch import nn
from torch.nn import functional as F

EXTRA_KINDS = ('bigru', 'tfm', 'mstcn', 'dgru', 'mspool', 'vn_tcn')


class BiGRUHead(nn.Module):
    def __init__(self, dim=3072, hid=128):
        super().__init__(); self.norm = nn.LayerNorm(dim); self.proj = nn.Linear(dim, 256); self.drop = nn.Dropout(.2)
        self.gru = nn.GRU(256, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=.2); self.out = nn.Linear(2 * hid, 9)

    def forward(self, x):
        h = self.drop(F.gelu(self.proj(self.norm(x)))); h, _ = self.gru(h[None]); return [self.out(self.drop(h[0]))]


class TfmHead(nn.Module):
    def __init__(self, dim=3072, d=256, layers=4, heads=4):
        super().__init__(); self.norm = nn.LayerNorm(dim); self.proj = nn.Linear(dim, d); self.drop = nn.Dropout(.2)
        self.pos = nn.Conv1d(d, d, 15, padding=7, groups=d)
        el = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=.2, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers, enable_nested_tensor=False); self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 9))

    def forward(self, x):
        h = self.drop(self.proj(self.norm(x)))[None]
        h = h + self.pos(h.transpose(1, 2)).transpose(1, 2)
        return [self.out(self.enc(h))[0]]


class _Stage(nn.Module):
    def __init__(self, cin, c=64, layers=10):
        super().__init__(); self.inp = nn.Conv1d(cin, c, 1)
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv1d(c, c, 3, padding=2 ** i, dilation=2 ** i), nn.ReLU(), nn.Conv1d(c, c, 1), nn.Dropout(.2)) for i in range(layers)])
        self.out = nn.Conv1d(c, 9, 1)

    def forward(self, x):
        h = self.inp(x)
        for l in self.layers: h = h + l(h)
        return self.out(h)


class MSTCNHead(nn.Module):
    def __init__(self, dim=3072, stages=3):
        super().__init__(); self.norm = nn.LayerNorm(dim); self.proj = nn.Linear(dim, 256); self.drop = nn.Dropout(.3)
        self.stages = nn.ModuleList([_Stage(256)] + [_Stage(7) for _ in range(stages - 1)])

    def forward(self, x):
        h = self.drop(F.gelu(self.proj(self.norm(x)))).T[None]; outs = []
        z = self.stages[0](h); outs.append(z[0].T)
        for s in self.stages[1:]:
            p = torch.cat([z[:, :4].softmax(1), z[:, 4:7].softmax(1)], 1)
            z = s(p); outs.append(z[0].T)
        return outs


class DiffGRUHead(nn.Module):
    """특징의 시간 차분(x_t - x_{t-3}, 움직임 신호)과 원특징을 따로 투영해 합친 BiGRU (오류 상관을 낮추려는 다른 입력 표현)"""
    def __init__(self, dim=3072, hid=128, lag=3):
        super().__init__(); self.lag = lag; self.norm = nn.LayerNorm(dim); self.normd = nn.LayerNorm(dim)
        self.px = nn.Linear(dim, 160); self.pd = nn.Linear(dim, 160); self.drop = nn.Dropout(.25)
        self.gru = nn.GRU(320, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=.25); self.out = nn.Linear(2 * hid, 9)

    def forward(self, x):
        d = x - torch.cat([x[:1].expand(self.lag, -1), x[:-self.lag]], 0) if len(x) > self.lag else torch.zeros_like(x)
        h = self.drop(F.gelu(torch.cat([self.px(self.norm(x)), self.pd(self.normd(d))], -1)))
        h, _ = self.gru(h[None]); return [self.out(self.drop(h[0]))]


class MSPoolHead(nn.Module):
    """순환·팽창 없이 다중 창 평균풀링(1/5/15/45프레임) + MLP. 대칭 국소 문맥만 사용"""
    def __init__(self, dim=3072, d=192, wins=(1, 5, 15, 45)):
        super().__init__(); self.wins = wins; self.norm = nn.LayerNorm(dim); self.proj = nn.Linear(dim, d); self.drop = nn.Dropout(.3)
        self.mlp = nn.Sequential(nn.LayerNorm(d * len(wins)), nn.Linear(d * len(wins), 256), nn.GELU(), nn.Dropout(.3), nn.Linear(256, 9))

    def forward(self, x):
        h = self.drop(F.gelu(self.proj(self.norm(x)))).T[None]; L = []
        for w in self.wins:
            L.append(h if w == 1 else F.avg_pool1d(F.pad(h, (w // 2, w // 2), mode='replicate'), w, 1))
        return [self.mlp(torch.cat(L, 1)[0].T)]


class VNormTCN(nn.Module):
    """영상(구간)별 특징 정규화 + motion TCN: 입력에서 해당 구간의 시간 평균을 빼고 학습셋 전체 평균(gmean)을 더함.
    카메라·해상도·차량처럼 영상 전체에 깔린 정적 차이를 입력 단계에서 제거 (OOD 대응)."""
    def __init__(self, dim=3072):
        super().__init__()
        import diverse_v2_models as _dv
        self.core = _dv.make_model('motion_tcn', dim); self.register_buffer('gmean', torch.zeros(dim))

    def forward(self, x):
        return self.core(x - x.mean(0, keepdim=True) + self.gmean)


def make_extra(kind, dim=3072):
    return {'bigru': BiGRUHead, 'tfm': TfmHead, 'mstcn': MSTCNHead, 'dgru': DiffGRUHead, 'mspool': MSPoolHead, 'vn_tcn': VNormTCN}[kind](dim)
