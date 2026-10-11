"""
Stage2Net (temporal dilated 1D-CNN) 학습 스크립트.

레퍼런스 하이퍼파라미터:
  in_dim=2382, hidden=256, dilations=[1,2,4,8,16,32,64]
  epochs=26, batch=16, lr=0.0003, weight_decay=0.01
  sigma=0.15(실라벨), sigma_derived=0.3(파생라벨)
  real_frac=0.5, aihub_col_weight=0.35
  p_stride2=0.15, min_seq_sec=12, max_seq_sec=45

사용법:
  # best.pt (전체 학습)
  python scripts/train_stage2net.py --nexar_eval_fold -1 --out models/stage2/s2_best.pt

  # fold 0
  python scripts/train_stage2net.py --nexar_eval_fold 0 --out models/stage2/s2_fold0.pt
"""
import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Stage2Net (레퍼런스 stage2_model.py 에서 복사)
# ─────────────────────────────────────────────────────────────────────────────
SIDE_LABELS    = ("LEFT", "RIGHT")
EVASION_LABELS = ("0", "1")
NEG_INF        = -1e4


class _ResBlock(nn.Module):
    def __init__(self, ch, dilation, kernel=3, groups=8, dropout=0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(groups, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(groups, ch), ch)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(self.norm2(self.conv2(h)))
        return F.gelu(x + h)


class Stage2Net(nn.Module):
    def __init__(self, in_dim, hidden=256,
                 dilations=(1,2,4,8,16,32),
                 dropout=0.1, proj=512,
                 feature_meta=None):
        super().__init__()
        self.feature_meta = dict(feature_meta or {})
        self.wants_ft = bool(self.feature_meta.get("backbone"))
        self.in_dim   = int(in_dim)
        self.hidden   = int(hidden)
        self.dilations= list(dilations)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std",  torch.ones(in_dim))

        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, proj, 1),
            nn.GroupNorm(min(8, proj), proj), nn.GELU(),
            nn.Conv1d(proj, hidden, 1),
            nn.GroupNorm(min(8, hidden), hidden), nn.GELU(),
        )
        self.blocks = nn.ModuleList([_ResBlock(hidden, d, dropout=dropout) for d in self.dilations])
        self.head_collision = nn.Conv1d(hidden, 1, 1)
        self.head_entry     = nn.Conv1d(hidden, 1, 1)
        self.att_query      = nn.Conv1d(hidden, 1, 1)
        pooled = hidden * 5
        self.clip_mlp  = nn.Sequential(nn.Linear(pooled, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head_side    = nn.Linear(hidden, len(SIDE_LABELS))
        self.head_evasion = nn.Linear(hidden, len(EVASION_LABELS))

    @torch.no_grad()
    def fit_normalization(self, feats):
        x = torch.as_tensor(np.asarray(feats)).reshape(-1, self.in_dim).float()
        m = torch.nan_to_num(x.mean(0))
        s = torch.nan_to_num(x.std(0), nan=1.0).clamp_min(1e-3)
        self.feat_mean.copy_(m)
        self.feat_std.copy_(s)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        x = torch.nan_to_num(x.float())
        x = (x - self.feat_mean) / self.feat_std
        x = x.clamp(-8., 8.).transpose(1, 2)
        x = x * mask.unsqueeze(1).float()
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        h = h * mask.unsqueeze(1).float()

        col = self.head_collision(h).squeeze(1)
        ent = self.head_entry(h).squeeze(1)
        col = col.masked_fill(~mask, NEG_INF)
        ent = ent.masked_fill(~mask, NEG_INF)

        att = self.att_query(h).squeeze(1).masked_fill(~mask, NEG_INF)
        n   = mask.float().sum(1, keepdim=True).clamp_min(1.)
        mean = h.sum(2) / n
        mx   = h.masked_fill(~mask.unsqueeze(1), NEG_INF).max(dim=2).values
        pools = [mean, mx]
        for logits in (att, col, ent):
            w = torch.softmax(logits, dim=1).unsqueeze(1)
            pools.append((h * w).sum(2))
        z = self.clip_mlp(torch.cat(pools, dim=1))
        return {"collision": col, "entry": ent,
                "side": self.head_side(z), "evasion": self.head_evasion(z)}


def gaussian_target(t_sec, centre_sec, sigma_sec=0.15):
    d = (np.asarray(t_sec, np.float64) - float(centre_sec)) / max(1e-6, sigma_sec)
    g = np.exp(-0.5 * d * d)
    s = g.sum()
    if s <= 1e-9:
        g = np.zeros_like(g)
        g[int(np.argmin(np.abs(np.asarray(t_sec) - centre_sec)))] = 1.0
        return g.astype(np.float32)
    return (g / s).astype(np.float32)


def soft_time_ce(logits, target, mask, weight):
    logp = torch.log_softmax(logits.masked_fill(~mask, NEG_INF), dim=1)
    per  = -(target * logp).sum(dim=1)
    w    = weight.clamp_min(0.)
    return (per * w).sum() / w.sum().clamp_min(1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# .npz 캐시 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_npz(cache_dir, corpus, clip_id):
    p = Path(cache_dir) / corpus / f"{clip_id}.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p)
        return {k: z[k] for k in ("t", "idx", "cnn", "cnn_flip", "flow", "fd")}
    except Exception:
        return None


def assemble(d, flip=False, cnn_gain=1.0, feat_noise=0.0):
    cnn  = (d["cnn_flip"] if flip else d["cnn"]).astype(np.float32)
    flow = d["flow"].astype(np.float32)
    fd   = d["fd"].astype(np.float32)
    feat = np.concatenate([cnn * cnn_gain, flow, fd], axis=1)
    if feat_noise > 0:
        feat += np.random.randn(*feat.shape).astype(np.float32) * feat_noise
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 클립 레코드
# ─────────────────────────────────────────────────────────────────────────────
def build_clip_records(cache_dir, aihub_labels_csv, nexar_dir,
                       nexar_eval_fold, hz=10.0):
    """
    returns: (train_records, eval_records)
    record = {clip_id, corpus, collision_sec, entry_sec, entry_side,
              evasion_space, sigma, has_entry, t_sec array, cache dict}
    """
    train_recs, eval_recs = [], []

    # ── AI Hub ───────────────────────────────────────────────────────────
    df = pd.read_csv(aihub_labels_csv)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="AI Hub 캐시 로드"):
        cid = row["clip_id"]
        d   = load_npz(cache_dir, "aihub", cid)
        if d is None:
            continue
        rec = {
            "clip_id":       cid,
            "corpus":        "aihub",
            "collision_sec": float(row["collision_sec"]),
            "entry_sec":     float(row["entry_sec"]),
            "entry_side":    0 if row["entry_side"] == "LEFT" else 1,
            "evasion_space": int(row["evasion_space"]),
            "sigma":         0.3,      # 파생 라벨 → 넓은 가우시안
            "has_entry":     True,
            "d":             d,
            "t_sec":         d["t"].astype(np.float64),
        }
        train_recs.append(rec)

    # ── Nexar ────────────────────────────────────────────────────────────
    nexar_dir = Path(nexar_dir)
    event_map = {}
    meta_path = nexar_dir / "train" / "positive" / "metadata.csv"
    if meta_path.exists():
        try:
            tmp = pd.read_csv(meta_path)
            for _, r in tmp.dropna(subset=["time_of_event"]).iterrows():
                cid = Path(str(r["file_name"])).stem
                event_map[cid] = float(r["time_of_event"])
        except Exception:
            pass

    nexar_clips = sorted((Path(cache_dir) / "nexar").glob("*.npz"))
    n_nexar = len(nexar_clips)
    fold_size = max(1, n_nexar // 2)

    for i, npz_path in enumerate(tqdm(nexar_clips, desc="Nexar 캐시 로드")):
        cid = npz_path.stem
        d   = load_npz(cache_dir, "nexar", cid)
        if d is None:
            continue
        col_sec = event_map.get(cid)
        if col_sec is None:
            continue
        rec = {
            "clip_id":       cid,
            "corpus":        "nexar",
            "collision_sec": col_sec,
            "entry_sec":     None,
            "entry_side":    -1,
            "evasion_space": -1,
            "sigma":         0.15,   # 실라벨
            "has_entry":     False,
            "d":             d,
            "t_sec":         d["t"].astype(np.float64),
        }
        fold = 0 if i < fold_size else 1
        if nexar_eval_fold == fold:
            eval_recs.append(rec)
        else:
            train_recs.append(rec)

    # ── CCD (Car Crash Dataset) ─────────────────────────────────────────────
    # DACON 공식 Stage2 baseline(000001~5)의 t_collision이 CCD Crash_Table.csv의
    # 프레임별 사고 플래그(첫 1이 뜨는 프레임)와 0프레임 오차로 정확히 일치함을 확인함
    # -> collision_frame에 대한 진짜 라벨. baseline 5개는 data/ccd_labels.csv 생성 시
    # 이미 제외되어 있음 (build_feature_cache_ccd.py EXCLUDE_CLIPS).
    ccd_labels_csv = Path("data/ccd_labels.csv")
    if ccd_labels_csv.exists():
        ccd_df = pd.read_csv(ccd_labels_csv, dtype={"clip_id": str})
        n_ccd = len(ccd_df)
        ccd_fold_size = max(1, n_ccd // 2)
        for i, row in enumerate(tqdm(ccd_df.itertuples(), total=n_ccd, desc="CCD 캐시 로드")):
            cid = row.clip_id
            d = load_npz(cache_dir, "ccd", cid)
            if d is None:
                continue
            rec = {
                "clip_id":       cid,
                "corpus":        "ccd",
                "collision_sec": float(row.collision_sec),
                "entry_sec":     None,
                "entry_side":    -1,
                "evasion_space": -1,
                "sigma":         0.15,   # 실라벨 (CCD Crash_Table.csv 원본)
                "has_entry":     False,
                "d":             d,
                "t_sec":         d["t"].astype(np.float64),
            }
            fold = 0 if i < ccd_fold_size else 1
            if nexar_eval_fold == fold:
                eval_recs.append(rec)
            else:
                train_recs.append(rec)

    return train_recs, eval_recs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ClipDataset(Dataset):
    def __init__(self, records, min_sec=12.0, max_sec=45.0, p_long=0.8,
                 p_stride2=0.15, cnn_gain=0.08, feat_noise=0.05):
        self.records   = records
        self.min_sec   = min_sec
        self.max_sec   = max_sec
        self.p_long    = p_long
        self.p_stride2 = p_stride2
        self.cnn_gain  = cnn_gain
        self.feat_noise= feat_noise

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec    = self.records[idx]
        d      = rec["d"]
        t_all  = rec["t_sec"]
        T_all  = len(t_all)

        # 윈도 샘플링 (p_long: 긴 쪽 절반에서 뽑을 확률 - 레퍼런스 체크포인트 args에서
        # 복구한 값(0.8)이나 정확한 구현은 원본 코드가 없어 추정. 평소엔 넉넉한 문맥,
        # 가끔(1-p_long)은 짧은 문맥으로 학습시켜 5초 안팎의 짧은 클립(CCD/baseline)에도
        # 강건하게 만드는 의도로 재구성.)
        dur = t_all[-1] - t_all[0] if T_all > 1 else 0.0
        mid = (self.min_sec + self.max_sec) / 2.0
        if random.random() < self.p_long:
            win = random.uniform(mid, self.max_sec)
        else:
            win = random.uniform(self.min_sec, mid)
        if dur > win:
            lo = random.uniform(t_all[0], t_all[-1] - win)
            hi = lo + win
            mask_t = (t_all >= lo) & (t_all <= hi)
        else:
            mask_t = np.ones(T_all, bool)

        t_win = t_all[mask_t]
        flip  = random.random() < 0.5

        feat  = assemble(
            {k: v[mask_t] for k, v in d.items() if k not in ("t", "idx")},
            flip=flip,
            cnn_gain=self.cnn_gain,
            feat_noise=self.feat_noise,
        )

        # stride-2 다운샘플링
        if random.random() < self.p_stride2 and len(t_win) > 4:
            feat  = feat[::2]
            t_win = t_win[::2]

        col_sec = rec["collision_sec"]
        ent_sec = rec["entry_sec"] if rec["has_entry"] else col_sec
        sigma   = rec["sigma"]

        col_tgt = gaussian_target(t_win, col_sec, sigma).astype(np.float32)
        ent_tgt = gaussian_target(t_win, ent_sec, sigma).astype(np.float32) if rec["has_entry"] else col_tgt.copy()

        side_lbl = rec["entry_side"] if not flip else (1 - rec["entry_side"]) if rec["entry_side"] >= 0 else -1
        evas_lbl = rec["evasion_space"]
        has_entry= rec["has_entry"]

        return {
            "feat":      torch.from_numpy(feat).float(),
            "t_sec":     torch.from_numpy(t_win).float(),
            "col_tgt":   torch.from_numpy(col_tgt),
            "ent_tgt":   torch.from_numpy(ent_tgt),
            "side":      side_lbl,
            "evasion":   evas_lbl,
            "has_entry": has_entry,
            "corpus":    rec["corpus"],
        }


def pad_collate(batch):
    T_max = max(b["feat"].shape[0] for b in batch)
    D     = batch[0]["feat"].shape[1]
    feats  = torch.zeros(len(batch), T_max, D)
    masks  = torch.zeros(len(batch), T_max, dtype=torch.bool)
    col_tgts = torch.zeros(len(batch), T_max)
    ent_tgts = torch.zeros(len(batch), T_max)
    sides    = torch.full((len(batch),), -1, dtype=torch.long)
    evasions = torch.full((len(batch),), -1, dtype=torch.long)
    has_entry= torch.zeros(len(batch), dtype=torch.bool)
    aihub_mask = torch.zeros(len(batch), dtype=torch.bool)

    for i, b in enumerate(batch):
        T = b["feat"].shape[0]
        feats[i, :T]    = b["feat"]
        masks[i, :T]    = True
        col_tgts[i, :T] = b["col_tgt"]
        ent_tgts[i, :T] = b["ent_tgt"]
        if b["side"] >= 0:
            sides[i]    = b["side"]
        if b["evasion"] >= 0:
            evasions[i] = b["evasion"]
        has_entry[i]    = b["has_entry"]
        aihub_mask[i]   = (b["corpus"] == "aihub")

    return feats, masks, col_tgts, ent_tgts, sides, evasions, has_entry, aihub_mask


# ─────────────────────────────────────────────────────────────────────────────
def build_mixed_batch(train_recs, real_frac, aihub_col_weight):
    """에폭마다 real(Nexar+CCD):AI Hub = real_frac:(1-real_frac) 비율로 섞어 반환.
    소수 코퍼스는 복원 추출(oversample), 다수 코퍼스는 부분 추출."""
    real  = [r for r in train_recs if r["corpus"] in ("nexar", "ccd")]
    aihub = [r for r in train_recs if r["corpus"] == "aihub"]

    if not real:
        return list(aihub)
    if not aihub:
        return list(real)

    total  = len(real) + len(aihub)
    n_real = round(total * real_frac)
    n_aihub = total - n_real

    mixed = random.choices(real, k=n_real) + random.choices(aihub, k=n_aihub)
    random.shuffle(mixed)
    return mixed


def eval_pass(model, records, device, sigma=0.15, max_clips=200):
    model.eval()
    col_hits = ent_hits = side_hits = evas_hits = 0
    col_n = ent_n = side_n = evas_n = 0

    sample = random.sample(records, min(max_clips, len(records)))
    with torch.no_grad():
        for rec in sample:
            d      = rec["d"]
            t_sec  = rec["t_sec"]
            feat   = assemble(d, flip=False, cnn_gain=0.08).astype(np.float32)
            x      = torch.from_numpy(feat[None]).to(device)
            mask   = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
            out    = model(x, mask)

            col_logits = out["collision"][0].cpu().numpy()
            ent_logits = out["entry"][0].cpu().numpy()
            T = len(t_sec)

            # argmax → seconds
            col_pred_sec = t_sec[int(np.argmax(col_logits[:T]))]
            ent_pred_sec = t_sec[int(np.argmax(ent_logits[:T]))]

            col_true = rec["collision_sec"]
            if abs(col_pred_sec - col_true) <= 0.3:
                col_hits += 1
            col_n += 1

            if rec["has_entry"]:
                ent_true = rec["entry_sec"]
                if abs(ent_pred_sec - ent_true) <= 0.3:
                    ent_hits += 1
                ent_n += 1

            if rec["entry_side"] >= 0:
                side_pred = int(out["side"][0].argmax().item())
                side_hits += (side_pred == rec["entry_side"])
                side_n += 1

            if rec["evasion_space"] >= 0:
                evas_pred = int(out["evasion"][0].argmax().item())
                evas_hits += (evas_pred == rec["evasion_space"])
                evas_n += 1

    return {
        "collision": col_hits / max(col_n, 1),
        "entry":     ent_hits / max(ent_n, 1),
        "side":      side_hits / max(side_n, 1),
        "evasion":   evas_hits / max(evas_n, 1),
        "n":         col_n,
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir",      default="data/stage2_ftF")
    ap.add_argument("--labels_csv",     default="data/aihub_labels.csv")
    ap.add_argument("--nexar_dir",      default="C:/Users/kiot_/Desktop/competition/blackbox_new/Black_Box/DATASET/nexar_collision_prediction")
    ap.add_argument("--out",            default="models/stage2/s2_best.pt")
    ap.add_argument("--nexar_eval_fold",type=int,   default=-1)
    ap.add_argument("--epochs",         type=int,   default=26)
    ap.add_argument("--batch",          type=int,   default=16)
    ap.add_argument("--num_workers",    type=int,   default=0)
    ap.add_argument("--lr",             type=float, default=3e-4)
    ap.add_argument("--weight_decay",   type=float, default=0.01)
    ap.add_argument("--hidden",         type=int,   default=256)
    ap.add_argument("--dropout",        type=float, default=0.15)
    ap.add_argument("--dilations",      default="1,2,4,8,16,32,64")
    ap.add_argument("--sigma",          type=float, default=0.15)
    ap.add_argument("--sigma_derived",  type=float, default=0.3)
    ap.add_argument("--aihub_col_weight",type=float,default=0.35)
    ap.add_argument("--real_frac",      type=float, default=0.5)
    ap.add_argument("--p_long",         type=float, default=0.8)
    ap.add_argument("--p_stride2",      type=float, default=0.15)
    ap.add_argument("--min_seq_sec",    type=float, default=12.0)
    ap.add_argument("--max_seq_sec",    type=float, default=45.0)
    ap.add_argument("--cnn_gain",       type=float, default=0.08)
    ap.add_argument("--feat_noise",     type=float, default=0.05)
    ap.add_argument("--entry_lead",     type=float, default=0.3)
    ap.add_argument("--eval_every",     type=int,   default=3)
    ap.add_argument("--aihub_eval_cap", type=int,   default=400)
    ap.add_argument("--seed",           type=int,   default=20260828)
    ap.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone_meta",  default="models/stage2/bb_new.pt",
                    help="학습된 backbone 경로 (feature_meta 기록용)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dilations = [int(x) for x in args.dilations.split(",")]
    os.makedirs(Path(args.out).parent, exist_ok=True)

    # ── 클립 로드 ──────────────────────────────────────────────────────────
    print("캐시 로드 중...")
    train_recs, eval_recs = build_clip_records(
        args.cache_dir, args.labels_csv, args.nexar_dir, args.nexar_eval_fold)
    print(f"train={len(train_recs)}, eval={len(eval_recs)}")

    aihub_eval = [r for r in train_recs + eval_recs if r["corpus"] == "aihub"]
    aihub_eval = random.sample(aihub_eval, min(args.aihub_eval_cap, len(aihub_eval)))

    # ── 모델 ───────────────────────────────────────────────────────────────
    feature_meta = {"backbone": args.backbone_meta, "with_frozen": False, "cnn_dim": 2304}
    in_dim = 2382   # 2304 + 70 + 8

    model = Stage2Net(in_dim=in_dim, hidden=args.hidden, dilations=dilations,
                      dropout=args.dropout, feature_meta=feature_meta)
    model = model.to(device)

    # 정규화 피팅 (첫 500개 클립 특징)
    print("정규화 피팅...")
    fit_feats = []
    for rec in random.sample(train_recs, min(200, len(train_recs))):
        fit_feats.append(assemble(rec["d"], flip=False, cnn_gain=args.cnn_gain))
    model.fit_normalization(np.concatenate(fit_feats, axis=0))

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_recs) // args.batch)
    def lr_lambda(step):
        prog = step / max(1, total_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_score = -1.0
    best_path  = args.out

    for epoch in range(1, args.epochs + 1):
        # 에폭마다 Nexar:AI Hub = real_frac:(1-real_frac) 비율로 재샘플링
        epoch_recs = build_mixed_batch(train_recs, args.real_frac, args.aihub_col_weight)
        train_ds = ClipDataset(epoch_recs, min_sec=args.min_seq_sec, max_sec=args.max_seq_sec,
                               p_long=args.p_long, p_stride2=args.p_stride2,
                               cnn_gain=args.cnn_gain, feat_noise=args.feat_noise)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=pad_collate,
                                  drop_last=True, pin_memory=(device.type == "cuda"))

        model.train()
        ep_loss = ep_n = 0

        for feats, masks, col_tgts, ent_tgts, sides, evasions, has_entry, aihub_m in \
                tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):

            feats, masks = feats.to(device), masks.to(device)
            col_tgts = col_tgts.to(device)
            ent_tgts = ent_tgts.to(device)
            sides    = sides.to(device)
            evasions = evasions.to(device)
            aihub_m  = aihub_m.to(device)

            out  = model(feats, masks)

            # collision weight: aihub는 낮게 (파생 라벨)
            col_w = torch.where(aihub_m, torch.tensor(args.aihub_col_weight, device=device),
                                torch.tensor(1.0, device=device))
            ent_w = torch.where(has_entry.to(device),
                                torch.tensor(1.0, device=device),
                                torch.tensor(0.0, device=device))

            loss_col = soft_time_ce(out["collision"], col_tgts, masks, col_w)
            loss_ent = soft_time_ce(out["entry"],     ent_tgts, masks, ent_w)

            valid_side = sides >= 0
            loss_side = F.cross_entropy(out["side"][valid_side], sides[valid_side]) \
                if valid_side.any() else torch.tensor(0., device=device)

            valid_evas = evasions >= 0
            loss_evas = F.cross_entropy(out["evasion"][valid_evas], evasions[valid_evas]) \
                if valid_evas.any() else torch.tensor(0., device=device)

            loss = 0.35 * loss_col + 0.35 * loss_ent + 0.15 * loss_side + 0.15 * loss_evas

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item() * feats.shape[0]
            ep_n    += feats.shape[0]

        print(f"  train_loss={ep_loss / max(ep_n,1):.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_set = eval_recs + aihub_eval
            if eval_set:
                metrics = eval_pass(model, eval_set, device, max_clips=args.aihub_eval_cap)
                score = (0.35 * metrics["collision"] + 0.35 * metrics["entry"] +
                         0.15 * metrics["side"]      + 0.15 * metrics["evasion"])
                print(f"  eval: col={metrics['collision']:.3f} ent={metrics['entry']:.3f} "
                      f"side={metrics['side']:.3f} evas={metrics['evasion']:.3f} "
                      f"score={score:.4f} n={metrics['n']}")

                if score > best_score:
                    best_score = score
                    ck = {
                        "model":          model.state_dict(),
                        "cfg":            {"in_dim": in_dim, "hidden": args.hidden,
                                          "dilations": dilations, "dropout": args.dropout,
                                          "feature_meta": feature_meta},
                        "args":           vars(args),
                        "in_dim":         in_dim,
                        "hidden":         args.hidden,
                        "feature_layout": {"cnn": 2304, "flow": 70, "fd": 8,
                                           "hz": 10.0, "backbone": "convnext_tiny"},
                        "feature_meta":   feature_meta,
                        "honest":         metrics,
                        "honest_key":     score,
                        "epoch":          epoch,
                    }
                    torch.save(ck, best_path)
                    print(f"  → 저장: {best_path} (score={score:.4f})")

    print(f"완료. best_score={best_score:.4f}")


if __name__ == "__main__":
    main()
