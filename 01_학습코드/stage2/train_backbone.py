"""
ImpactNet (stage2_backbone.pt) 학습 스크립트.

레퍼런스 하이퍼파라미터:
  epochs=24, batch=64, lr=0.0001, head_lr_mult=10.0, weight_decay=0.05
  drop_path=0.1, w_col=0.55, w_ent=0.25, w_side=0.1, w_evas=0.1
  p_near=0.5, lag2_p=0.3

사용법:
  python scripts/train_backbone.py \
    --labels_csv data/aihub_labels.csv \
    --video_map  data/aihub_video_map.csv \
    --out        models/stage2/bb_new.pt
"""
import argparse
import math
import os
import random
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# ImpactNet (레퍼런스 stage2_backbone.py 에서 복사)
# ─────────────────────────────────────────────────────────────────────────────
N_BANDS    = 3
BACKBONE_CH = {"convnext_tiny": 768}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
BIN_EDGES = (-4.0, -2.5, -1.6, -1.0, -0.6, -0.3, 0.3, 0.6, 1.0, 1.6, 2.5, 4.0)
N_BINS = len(BIN_EDGES) + 1


def offset_bin(dt):
    return int(np.searchsorted(np.asarray(BIN_EDGES), float(dt), side="right"))


class ImpactNet(nn.Module):
    def __init__(self, name="convnext_tiny", n_frames=2, pretrained=True,
                 drop_path=0.1, head_hidden=512, pool="mean"):
        super().__init__()
        import timm
        self.name = name
        self.n_frames = int(n_frames)
        self.pool = pool
        ch = BACKBONE_CH.get(name, BACKBONE_CH.get(name.split(".")[0], 768))
        self.ch = ch
        # pretrained=True → ImageNet 가중치 자동 다운로드
        m = timm.create_model(name, pretrained=pretrained, features_only=True,
                              out_indices=(3,), drop_path_rate=drop_path)
        self.backbone = m
        self._inflate_stem()
        self.feat_dim = ch * N_BANDS
        self.norm  = nn.LayerNorm(self.feat_dim)
        self.trunk = nn.Sequential(nn.Linear(self.feat_dim, head_hidden), nn.GELU(), nn.Dropout(0.1))
        self.head_col  = nn.Linear(head_hidden, N_BINS)
        self.head_ent  = nn.Linear(head_hidden, N_BINS)
        self.head_side = nn.Linear(head_hidden, 2)
        self.head_evas = nn.Linear(head_hidden, 2)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).repeat(self.n_frames).view(1,-1,1,1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD ).repeat(self.n_frames).view(1,-1,1,1))

    def _inflate_stem(self):
        """6채널 스템 팽창: [W, 0] 초기화 — 기존 ImageNet 응답 보존 + 모션 경로는 0."""
        if self.n_frames == 1:
            return
        conv = None
        for mod in self.backbone.modules():
            if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
                conv = mod
                break
        if conv is None:
            return
        w = conv.weight.data
        new = torch.zeros(w.shape[0], 3 * self.n_frames, *w.shape[2:], dtype=w.dtype)
        new[:, :3] = w   # frame_t: ImageNet 가중치
        # new[:, 3:]: frame_{t-1}: 0 (모션 경로 처음엔 0)
        conv.weight    = nn.Parameter(new)
        conv.in_channels = 3 * self.n_frames

    def bands(self, x):
        f = self.backbone(x)[-1]
        a = F.adaptive_avg_pool2d(f, (1, N_BANDS)).flatten(1)
        return a

    def forward(self, x):
        z = self.trunk(self.norm(self.bands(x)))
        return {"col": self.head_col(z), "ent": self.head_ent(z),
                "side": self.head_side(z), "evasion": self.head_evas(z)}


# ─────────────────────────────────────────────────────────────────────────────
# 청크 단위 mp4 추출 (ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_one(args_tuple):
    clip_id, zip_path = args_tuple
    # 이미 통짜로 풀려있는 mp4 직접경로면 압축해제 건너뛰고 그대로 사용
    # (임시파일이 아니므로 cleanup()에서 지우지 않게 extract_chunk가 별도 표시함)
    if str(zip_path).lower().endswith(".mp4"):
        return clip_id, (zip_path if os.path.exists(zip_path) else None)
    try:
        with zipfile.ZipFile(zip_path) as z:
            mp4_name = f"{clip_id}.mp4"
            if mp4_name not in z.namelist():
                return clip_id, None
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(z.read(mp4_name))
            tmp.close()
            return clip_id, tmp.name
    except Exception:
        return clip_id, None


def extract_chunk(chunk_rows, n_threads=8):
    """chunk_rows: [(clip_id, zip_path), ...] → ({clip_id: tmp_path}, [tmp_files])
    zip_path가 이미 통짜 mp4 직접경로인 경우 path == 원본 zip_path 그대로 반환되므로
    (임시파일이 아니라 영구파일) tmp_files(삭제 대상)에 넣지 않음."""
    original_paths = {cid: zp for cid, zp in chunk_rows}
    vcache, tmp_files = {}, []
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(_extract_one, row): row for row in chunk_rows}
        for fut in as_completed(futures):
            cid, path = fut.result()
            if path:
                vcache[cid] = path
                if path != original_paths.get(cid):
                    tmp_files.append(path)
    return vcache, tmp_files


def cleanup(tmp_files):
    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — DataLoader(num_workers>0)에게 디코딩을 맡김: 워커는 별도 프로세스라
# 메인 프로세스(CUDA 실행)와 메모리/GIL을 안 나눠 씀 -> 스레드-CUDA 경합으로 인한
# ACCESS_VIOLATION이 구조적으로 불가능해짐. __init__은 메타데이터(plan)만 만들고,
# 실제 프레임 디코딩은 __getitem__에서 워커 프로세스 안에서 지연 실행함.
# ─────────────────────────────────────────────────────────────────────────────
_video_cap_cache = {}       # 워커 프로세스별 cv2.VideoCapture 캐시 (LRU, 재탐색 비용 줄임)
_video_cap_cache_order = []
_CAP_CACHE_MAX = 8
_ccd_zf_cache = {}          # 워커 프로세스별 CCD zip 핸들 캐시


def _get_cap(vpath):
    cap = _video_cap_cache.get(vpath)
    if cap is not None:
        return cap
    cap = cv2.VideoCapture(vpath)
    if len(_video_cap_cache_order) >= _CAP_CACHE_MAX:
        oldest = _video_cap_cache_order.pop(0)
        old = _video_cap_cache.pop(oldest, None)
        if old is not None:
            old.release()
    _video_cap_cache[vpath] = cap
    _video_cap_cache_order.append(vpath)
    return cap


def _get_ccd_zf(ccd_zip_path):
    zf = _ccd_zf_cache.get(ccd_zip_path)
    if zf is None:
        zf = zipfile.ZipFile(ccd_zip_path)
        _ccd_zf_cache[ccd_zip_path] = zf
    return zf


_frame_cache_mm = {}  # 워커 프로세스별 디코딩 캐시 memmap (경로 -> (mm_x, mm_filled)), lazy-open


def _get_cache_mm(x_path, f_path, x_shape):
    """x_path를 키로 memmap을 프로세스 안에서 딱 1번만 엶(그 뒤로는 캐시 재사용).
    ChunkDataset이 이 memmap 객체 자체를 들고 있다가 워커 프로세스로 pickle되면
    (Windows spawn) ndarray 기본 pickling 때문에 파일과 끊긴 사본이 되어버려서
    쓰기가 디스크에 반영이 안 됨(실측 확인됨) -> 그래서 경로만 넘기고 여기서 다시 엶."""
    mm = _frame_cache_mm.get(x_path)
    if mm is None:
        mm_x = np.memmap(x_path, dtype=np.uint8, mode="r+", shape=x_shape)
        mm_f = np.memmap(f_path, dtype=np.uint8, mode="r+", shape=(x_shape[0],))
        mm = (mm_x, mm_f)
        _frame_cache_mm[x_path] = mm
    return mm


# ── GPU(NVDEC) 디코딩 경로 ──────────────────────────────────────────────────
# 원본 AI-Hub mp4는 MPEG-4 Part2(구형 코덱)라 NVDEC이 아예 지원 안 함 + 기본 GOP가
# 클립 전체라 랜덤 접근이 극도로 느림 -> H.264로 재인코딩(-g 8 -bf 0, 짧은 GOP)한
# 사본(aihub_extracted_h264_shortgop/)을 대신 읽음. num_decoder_engines=1(단일
# 물리 엔진)이라 동시 세션 여러 개는 디코드 에러를 내므로, 이 경로는 반드시
# num_workers=0(메인 프로세스 단일 스레드)에서만 써야 함.
_GPU_DEC_CACHE = {}
_GPU_DEC_ORDER = []
_GPU_DEC_CACHE_MAX = 16
_GPU_SHORTGOP_DIR = "D:/workspace_mj/aihub_extracted_h264_shortgop"


def _gpu_path(vpath):
    if not vpath:
        return vpath
    return os.path.join(_GPU_SHORTGOP_DIR, os.path.basename(vpath))


def _get_gpu_decoder(vpath):
    dec = _GPU_DEC_CACHE.get(vpath)
    if dec is not None:
        return dec
    import PyNvVideoCodec as nvc
    dec = nvc.SimpleDecoder(vpath, output_color_type=nvc.OutputColorType.RGB)
    if len(_GPU_DEC_ORDER) >= _GPU_DEC_CACHE_MAX:
        oldest = _GPU_DEC_ORDER.pop(0)
        _GPU_DEC_CACHE.pop(oldest, None)
    _GPU_DEC_CACHE[vpath] = dec
    _GPU_DEC_ORDER.append(vpath)
    return dec


def _decode_pair_gpu(vpath, t, p, size):
    """t, p 두 프레임을 NVDEC(GPU)로 디코딩+리사이즈한 뒤 CPU 텐서로 반환.
    DataLoader 워커 프로세스(num_workers=1) 안에서 실행됨 - 워커->메인 프로세스로
    넘길 때 CUDA IPC(Windows에서 불안정) 대신 평범한 CPU 텐서 pickling을 쓰기 위해
    일부러 .cpu()로 내림(메인 프로세스의 기존 .to(device) 호출이 다시 GPU로 옮김).
    실패 시 None (호출부에서 cv2 경로로 폴백)."""
    gpath = _gpu_path(vpath)
    try:
        dec = _get_gpu_decoder(gpath)
        n_frames = len(dec)
        t = min(t, n_frames - 1); p = min(p, n_frames - 1)
        ft = dec[t]; fp = dec[p]
        xt = torch.as_tensor(ft.cuda()[0], device="cuda")  # (H,W,3) uint8
        xp = torch.as_tensor(fp.cuda()[0], device="cuda")
        x6 = torch.cat([xt, xp], dim=-1).permute(2, 0, 1).float().div_(255.0)  # (6,H,W)
        x6 = F.interpolate(x6.unsqueeze(0), size=(size, size), mode="bilinear",
                           align_corners=False).squeeze(0)
        return x6.cpu()
    except Exception:
        return None


def _decode_one_frame(corpus, vpath, cid, fidx, size, ccd_zip_path):
    blank = np.zeros((size, size, 3), np.uint8)
    if corpus == "ccd":
        if not ccd_zip_path:
            return blank
        try:
            zf = _get_ccd_zf(ccd_zip_path)
            name = f"CrashBest/C_{cid}_{fidx+1:02d}.jpg"
            with zf.open(name) as f:
                buf = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return blank
            return cv2.resize(img[:, :, ::-1], (size, size), interpolation=cv2.INTER_AREA)
        except (KeyError, Exception):
            return blank
    else:
        if not vpath:
            return blank
        try:
            cap = _get_cap(vpath)
            if not cap.isOpened():
                return blank
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                return blank
            return cv2.resize(frame[:, :, ::-1], (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            return blank


class ChunkDataset(Dataset):
    """clip_id마다 samples_per_clip개 (t, t-lag) 프레임쌍을 샘플링한 계획(plan)만
    미리 만들어두고, 실제 디코딩은 __getitem__에서(=DataLoader 워커 프로세스 안에서)
    지연 수행함.

    corpus=="ccd"인 행은 entry/side/evasion 정답이 없음(CCD Crash_Table.csv는
    collision만 real label) -> has_label=False로 표시해 그 손실만 마스킹."""
    def __init__(self, df, vcache, size=256, p_near=0.5, lag2_p=0.3,
                 fps=15.0, samples_per_clip=8, ccd_zip_path=None, gpu_decode=False,
                 cache_path=None):
        self.size = size
        self.ccd_zip_path = ccd_zip_path
        self.gpu_decode = gpu_decode
        self.items = []  # list of dict(cid, vpath, corpus, t, p, col_bin, ent_bin, side, evasion, has_label)
        for _, row in df.iterrows():
            cid      = row["clip_id"]
            corpus   = row.get("corpus", "aihub")
            vpath    = vcache.get(cid)
            fps_i    = float(row.get("fps", fps))
            has_label = bool(row.get("has_label", True))
            col_fidx = int(row["collision_frame_idx"])
            ent_fidx = int(row["entry_frame_idx"])
            side_lbl = 0 if row["entry_side"] == "LEFT" else 1
            evas_lbl = int(row["evasion_space"])
            n_frames = int(row["n_frames"])
            near_win = int(2.0 * fps_i)

            for _ in range(samples_per_clip):
                if random.random() < p_near:
                    lo = max(0, col_fidx - near_win)
                    hi = min(n_frames - 1, col_fidx + near_win)
                else:
                    lo, hi = 0, n_frames - 1
                t = random.randint(lo, hi)
                lag = 2 if random.random() < lag2_p else 1
                p = max(0, t - lag)
                col_offset = (t - col_fidx) / fps_i
                ent_offset = (t - ent_fidx) / fps_i
                self.items.append({
                    "cid": cid, "vpath": vpath, "corpus": corpus, "t": t, "p": p,
                    "col_bin": offset_bin(col_offset), "ent_bin": offset_bin(ent_offset),
                    "side": side_lbl, "evasion": evas_lbl, "has_label": has_label,
                })

        # ── 디코딩 결과 디스크 캐시 (memmap) ──────────────────────────────────
        # 위에서 샘플링한 (t,p) 플랜은 이 ds 객체가 살아있는 한(=여러 epoch 동안) 고정.
        # 같은 idx는 항상 같은 프레임쌍을 가리키므로, 한 번 디코딩한 결과를
        # uint8 그대로 디스크에 저장해두면 다음 epoch부터는 디코딩 없이 읽기만 하면 됨.
        #
        # 주의: memmap 인스턴스를 self에 직접 들고 있으면 DataLoader가 이 Dataset을
        # 워커 프로세스로 pickle할 때(Windows는 spawn만 지원) ndarray의 기본 pickling이
        # 적용되어 "파일과 연결된 memmap"이 아니라 "내용만 복사된 일반 배열"이 되어버림
        # (실측 확인됨: 워커가 쓴 값이 디스크에 전혀 반영 안 됨). 그래서 여기서는 경로
        # 문자열/shape만 저장해두고, 실제 memmap은 _get_cache_mm()에서 프로세스별로
        # (워커든 메인이든) 그때그때 새로 열게 함 -> 진짜로 파일에 쓰기가 반영됨.
        self.cache_x_path = None
        self.cache_f_path = None
        self.cache_shape = None
        if cache_path is not None:
            n_items = len(self.items)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            x_path = cache_path + "_x.dat"
            f_path = cache_path + "_filled.dat"
            x_shape = (n_items, 6, size, size)
            mode_x = "r+" if (os.path.exists(x_path) and os.path.getsize(x_path) == n_items * 6 * size * size) else "w+"
            mode_f = "r+" if (os.path.exists(f_path) and os.path.getsize(f_path) == n_items) else "w+"
            np.memmap(x_path, dtype=np.uint8, mode=mode_x, shape=x_shape)  # 파일 생성/사이즈 확보만
            np.memmap(f_path, dtype=np.uint8, mode=mode_f, shape=(n_items,))
            self.cache_x_path = x_path
            self.cache_f_path = f_path
            self.cache_shape = x_shape

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mm_x = mm_f = None
        if self.cache_x_path is not None:
            mm_x, mm_f = _get_cache_mm(self.cache_x_path, self.cache_f_path, self.cache_shape)
        x_u8 = None
        if mm_f is not None and mm_f[idx]:
            x_u8 = np.array(mm_x[idx])  # (6,H,W) uint8, memmap에서 복사
        if x_u8 is None:
            x = None
            if self.gpu_decode and it["corpus"] == "aihub":
                x = _decode_pair_gpu(it["vpath"], it["t"], it["p"], self.size)
            if x is not None:
                x_u8 = (x.clamp(0, 1) * 255.0).round().byte().numpy()
            else:  # CCD/Nexar 이거나, GPU 디코딩 실패 -> CPU 경로 (cv2)
                rgb_t = _decode_one_frame(it["corpus"], it["vpath"], it["cid"], it["t"], self.size, self.ccd_zip_path)
                rgb_p = _decode_one_frame(it["corpus"], it["vpath"], it["cid"], it["p"], self.size, self.ccd_zip_path)
                rgb6  = np.concatenate([rgb_t, rgb_p], axis=2)
                x_u8  = rgb6.transpose(2, 0, 1).copy()  # HWC -> CHW uint8
            if mm_x is not None:
                mm_x[idx] = x_u8
                mm_f[idx] = 1
        x = torch.from_numpy(np.ascontiguousarray(x_u8)).float().div_(255.0)
        return {"x": x, "col_bin": it["col_bin"], "ent_bin": it["ent_bin"],
                "side": it["side"], "evasion": it["evasion"], "has_label": it["has_label"]}


def collate_fn(batch):
    return (
        torch.stack([b["x"]       for b in batch]),
        torch.tensor([b["col_bin"] for b in batch], dtype=torch.long),
        torch.tensor([b["ent_bin"] for b in batch], dtype=torch.long),
        torch.tensor([b["side"]    for b in batch], dtype=torch.long),
        torch.tensor([b["evasion"] for b in batch], dtype=torch.long),
        torch.tensor([b["has_label"] for b in batch], dtype=torch.bool),
    )


def _prefetch_batches(ds, batch_size, shuffle, queue_maxsize=2):
    """GPU 디코딩 전용 배치 이터레이터: 백그라운드 스레드 1개가 다음 배치를 미리
    디코딩(NVDEC, 세션 1개만 씀 - 동시 세션 아님)해서 큐에 넣어두고, 메인 스레드는
    그동안 이전 배치로 GPU 연산(forward/backward)을 진행함 -> 디코딩과 연산이
    겹쳐서(오버랩) DataLoader(num_workers=0)의 완전 순차 실행보다 빨라짐."""
    import threading, queue as _queue
    idxs = list(range(len(ds)))
    if shuffle:
        random.shuffle(idxs)
    n_batches = math.ceil(len(idxs) / batch_size)
    q = _queue.Queue(maxsize=queue_maxsize)
    SENTINEL = object()

    def _producer():
        for bi in range(n_batches):
            batch_idxs = idxs[bi * batch_size:(bi + 1) * batch_size]
            items = [ds[i] for i in batch_idxs]
            q.put(collate_fn(items))
        q.put(SENTINEL)

    th = threading.Thread(target=_producer, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        yield item
    th.join()


# ─────────────────────────────────────────────────────────────────────────────
def _build_dataset(df, vcache, args, is_train, ccd_zip_path=None):
    cache_dir = getattr(args, "frame_cache_dir", None) or None
    cache_path = os.path.join(cache_dir, "train" if is_train else "val") if cache_dir else None
    return ChunkDataset(df, vcache, size=args.size,
                        p_near=args.p_near if is_train else 0.5,
                        lag2_p=args.lag2_p if is_train else 0.0,
                        samples_per_clip=8 if is_train else 4,
                        ccd_zip_path=ccd_zip_path,
                        gpu_decode=getattr(args, "gpu_decode", False),
                        cache_path=cache_path)


def _train_on_ds(model, ds, args, device, optimizer, scheduler, is_train, scaler=None, n_workers=0):
    if is_train:
        model.train()
    else:
        model.eval()
    gpu_decode = getattr(args, "gpu_decode", False)
    if gpu_decode:
        # 워커 프로세스 1개만 사용(NVDEC 단일 엔진 -> 세션 1개만 존재, 동시성 충돌 없음).
        # 스레드가 아니라 진짜 별도 프로세스라 GIL 영향을 안 받고, 워커가 CPU 텐서로
        # 내려서 반환하므로(CUDA IPC 불필요) PyTorch DataLoader의 표준 prefetch
        # 메커니즘을 그대로 씀 - 디코딩(워커 프로세스)과 GPU 연산(메인 프로세스)이 겹침.
        n_workers = 1
    loader = DataLoader(ds, batch_size=args.batch,
                        shuffle=is_train, num_workers=n_workers,
                        collate_fn=collate_fn, drop_last=False,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=False,
                        timeout=180 if n_workers > 0 else 0)
    tot_loss = col_acc = ent_acc = side_acc = evas_acc = n = n_lab = 0
    n_batches = math.ceil(len(ds) / args.batch)
    use_amp = device.type == "cuda"
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for step, (xs, cb, eb, sl, ev, hl) in enumerate(loader, 1):
            xs, cb, eb, sl, ev, hl = [t.to(device) for t in (xs, cb, eb, sl, ev, hl)]
            mn_d = model.mean.float(); st_d = model.std.float()
            xs_n = (xs.float() - mn_d) / st_d
            hl_f = hl.float()
            n_labeled = hl_f.sum().clamp_min(1.0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(xs_n)
                # collision은 aihub/ccd 전부 real/derived label 존재 -> 항상 전체 배치 사용.
                # entry/side/evasion은 CCD에 정답이 없어 has_label(hl)로 마스킹.
                loss_col  = F.cross_entropy(out["col"].float(), cb)
                loss_ent  = (F.cross_entropy(out["ent"].float(),     eb, reduction="none") * hl_f).sum() / n_labeled
                loss_side = (F.cross_entropy(out["side"].float(),    sl, reduction="none") * hl_f).sum() / n_labeled
                loss_evas = (F.cross_entropy(out["evasion"].float(), ev, reduction="none") * hl_f).sum() / n_labeled
                loss = (args.w_col * loss_col + args.w_ent * loss_ent +
                        args.w_side * loss_side + args.w_evas * loss_evas)
            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
            B = xs.shape[0]
            tot_loss += loss.item() * B
            col_acc  += (out["col"].argmax(1) == cb).sum().item()
            ent_acc  += ((out["ent"].argmax(1)     == eb) & hl).sum().item()
            side_acc += ((out["side"].argmax(1)    == sl) & hl).sum().item()
            evas_acc += ((out["evasion"].argmax(1) == ev) & hl).sum().item()
            n     += B
            n_lab += int(hl.sum().item())
            if is_train and (step % 100 == 0 or step == n_batches):
                print(f"    batch {step}/{n_batches}  loss={tot_loss/max(n,1):.4f}", flush=True)
    return tot_loss, col_acc, ent_acc, side_acc, evas_acc, n, n_lab


def train_one_epoch(model, ds, optimizer, scheduler, args, device, n_threads, scaler=None):
    tl, _, _, _, _, n, _ = _train_on_ds(model, ds, args, device, optimizer, scheduler,
                                        True, scaler, n_workers=n_threads)
    return tl / max(n, 1)


def eval_model(model, ds, args, device, n_threads):
    tot = _train_on_ds(model, ds, args, device, None, None, False, n_workers=n_threads)
    n, n_lab = tot[5], max(tot[6], 1)
    if n == 0: return {}
    return {"loss": tot[0]/n, "col_acc": tot[1]/n, "ent_acc": tot[2]/n_lab,
            "side_acc": tot[3]/n_lab, "evas_acc": tot[4]/n_lab}


# ─────────────────────────────────────────────────────────────────────────────
def _load_dataframes(args):
    """labels/video_map(+CCD+Nexar) 로드해서 (train_df, val_df, zip_map, ccd_zip_path, nexar_vcache) 리턴.
    main()과 standalone 캐시 채우기 스크립트가 똑같은 로직을 공유하기 위해 분리함
    (샘플링 플랜이 두 곳에서 달라지면 frame_cache_dir 캐시 인덱스가 어긋나버림)."""
    df_lab = pd.read_csv(args.labels_csv)
    df_map = pd.read_csv(args.video_map)
    df = df_lab.merge(df_map, on="clip_id", how="inner")
    zip_map = {row["clip_id"]: row["zip_path"] for _, row in df_map.iterrows()}
    if args.downloads_root:
        zip_map = {k: (v.replace("/downloads", args.downloads_root, 1) if v.startswith("/downloads") else v)
                  for k, v in zip_map.items()}
    before = len(df)
    df = df[df["n_frames"] <= 5000].reset_index(drop=True)  # 비정상 긴 클립 제외
    df["corpus"]    = "aihub"
    df["fps"]       = 15.0
    df["has_label"] = True
    print(f"전체 클립: {before}개 → n_frames≤5000 필터 후 {len(df)}개")

    n_val    = max(1, int(len(df) * args.val_frac))
    val_df   = df.sample(n_val, random_state=args.seed).reset_index(drop=True)
    train_df = df[~df["clip_id"].isin(val_df["clip_id"])].reset_index(drop=True)

    # ── CCD (Car Crash Dataset) — collision_frame real label, entry/side/evasion 없음 ──
    ccd_zip_path = None
    ccd_path = Path(args.ccd_labels_csv)
    if ccd_path.exists() and Path(args.ccd_zip).exists():
        ccd_df = pd.read_csv(ccd_path, dtype={"clip_id": str})
        ccd_df["collision_frame_idx"] = (ccd_df["collision_sec"] * 10.0).round().astype(int)
        ccd_df["entry_frame_idx"] = ccd_df["collision_frame_idx"]  # 더미(has_label=False로 손실 마스킹)
        ccd_df["entry_side"]      = "LEFT"                          # 더미
        ccd_df["evasion_space"]   = 0                                # 더미
        ccd_df["n_frames"] = 50
        ccd_df["corpus"]    = "ccd"
        ccd_df["fps"]       = 10.0
        ccd_df["has_label"] = False
        keep_cols = ["clip_id", "collision_frame_idx", "entry_frame_idx", "entry_side",
                    "evasion_space", "n_frames", "corpus", "fps", "has_label"]
        train_df = pd.concat([train_df, ccd_df[keep_cols]], ignore_index=True)
        # CCD clip_id는 zip_map에 넣지 않음 -> extract_chunk(mp4 압축해제)는 자동으로 건너뛰고,
        # ChunkDataset이 corpus=="ccd"를 보고 ccd_zip_path에서 JPG로 직접 읽음.
        ccd_zip_path = args.ccd_zip
        print(f"CCD 통합: +{len(ccd_df)}개 real collision label 클립 (entry/side/evasion은 손실에서 제외)")
    else:
        print(f"[skip] CCD 라벨/zip 없음 ({args.ccd_labels_csv} / {args.ccd_zip}) — AI-Hub만 사용")

    # ── Nexar — collision_frame real label(time_of_event), entry/side/evasion 없음 ──
    nexar_vcache = {}
    nexar_dir = Path(args.nexar_dir)
    nexar_meta = nexar_dir / "train" / "positive" / "metadata.csv"
    if nexar_meta.exists():
        meta = pd.read_csv(nexar_meta)
        rows = []
        for _, r in meta.dropna(subset=["time_of_event"]).iterrows():
            cid = Path(str(r["file_name"])).stem
            mp4_path = nexar_dir / "train" / "positive" / f"{cid}.mp4"
            if not mp4_path.exists():
                continue
            cap = cv2.VideoCapture(str(mp4_path))
            fps_i = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames_i = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps_i <= 0 or n_frames_i <= 0:
                continue
            col_fidx = int(round(float(r["time_of_event"]) * fps_i))
            rows.append({"clip_id": cid, "collision_frame_idx": col_fidx,
                        "entry_frame_idx": col_fidx, "entry_side": "LEFT", "evasion_space": 0,
                        "n_frames": n_frames_i, "corpus": "nexar", "fps": fps_i, "has_label": False})
            nexar_vcache[cid] = str(mp4_path)
        if rows:
            nexar_df = pd.DataFrame(rows)
            train_df = pd.concat([train_df, nexar_df], ignore_index=True)
            print(f"Nexar 통합: +{len(nexar_df)}개 real collision label 클립 (entry/side/evasion은 손실에서 제외)")
    else:
        print(f"[skip] Nexar metadata 없음 ({nexar_meta}) — 미사용")

    return train_df, val_df, zip_map, ccd_zip_path, nexar_vcache


def _build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv",   default="data/aihub_labels.csv")
    ap.add_argument("--video_map",    default="data/aihub_video_map.csv")
    ap.add_argument("--out",          default="models/stage2/bb_new.pt")
    ap.add_argument("--epochs",       type=int,   default=24)
    ap.add_argument("--batch",        type=int,   default=64)
    ap.add_argument("--warmup_steps", type=int,   default=500)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--head_lr_mult", type=float, default=10.0)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--drop_path",    type=float, default=0.1)
    ap.add_argument("--w_col",        type=float, default=0.55)
    ap.add_argument("--w_ent",        type=float, default=0.25)
    ap.add_argument("--w_side",       type=float, default=0.10)
    ap.add_argument("--w_evas",       type=float, default=0.10)
    ap.add_argument("--p_near",       type=float, default=0.5)
    ap.add_argument("--lag2_p",       type=float, default=0.3)
    ap.add_argument("--val_frac",     type=float, default=0.01,  help="val 비율")
    ap.add_argument("--val_every",    type=int,   default=6)
    ap.add_argument("--size",         type=int,   default=256)
    ap.add_argument("--backbone_name", default="convnext_tiny.dinov3_lvd1689m",
                    help="timm 백본 이름 (예: convnext_tiny, convnext_tiny.dinov3_lvd1689m)")
    ap.add_argument("--chunk_size",   type=int,   default=200,   help="(미사용, 하위호환용) DataLoader 방식으로 전환되어 더 이상 안 씀")
    ap.add_argument("--n_threads",    type=int,   default=8,     help="DataLoader num_workers (디코딩 워커 프로세스 수, gpu_decode 켜면 무시되고 0으로 강제됨)")
    ap.add_argument("--gpu_decode",   action="store_true",
                    help="AI-Hub(aihub) 클립을 NVDEC(PyNvVideoCodec)으로 디코딩 (H.264 짧은-GOP 재인코딩본 필요). "
                         "num_decoder_engines=1이라 num_workers는 자동으로 0이 됨.")
    ap.add_argument("--seed",         type=int,   default=20260828)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--downloads_root", default=None,
                    help="'/downloads' 접두사를 로컬 경로로 치환 (도커 밖에서 실행할 때)")
    ap.add_argument("--ccd_labels_csv", default="data/ccd_labels.csv",
                    help="CCD(Car Crash Dataset) real collision label CSV. 없으면 CCD 미사용")
    ap.add_argument("--ccd_zip",        default="C:/Users/Server/Downloads/ccd.zip")
    ap.add_argument("--nexar_dir",      default="D:/workspace_mj/nexar_collision_prediction",
                    help="Nexar real collision label (time_of_event). 없으면 미사용")
    ap.add_argument("--frame_cache_dir", default="D:/workspace_mj/backbone_frame_cache",
                    help="디코딩된 프레임쌍(uint8) memmap 캐시 디렉토리. 샘플링 플랜을 "
                         "학습 전체(모든 epoch) 동안 고정하고, 1회차에 디코딩한 결과를 "
                         "여기 저장해서 2회차부터는 재디코딩 없이 읽기만 함. 빈 문자열이면 캐시 끔.")
    ap.add_argument("--resume",         default=None,
                    help="이 경로의 체크포인트(모델+옵티마이저+스케줄러+epoch)에서 이어서 학습")
    return ap


def main():
    args = _build_argparser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    device = torch.device(args.device)
    train_df, val_df, zip_map, ccd_zip_path, nexar_vcache = _load_dataframes(args)

    print(f"train={len(train_df)}, val={len(val_df)}")
    if args.gpu_decode:
        print("gpu_decode=True -> num_workers=1 (NVDEC 단일 엔진, aihub는 GPU/H.264-shortgop, ccd/nexar는 CPU)")
    else:
        print(f"num_workers={args.n_threads} (DataLoader 워커 프로세스)")

    # ── 학습/검증 데이터셋: 프로세스 시작 시 딱 1번만 빌드(=샘플링 플랜 고정) ──
    # train_one_epoch를 매 epoch 새로 부르면서 매번 _build_dataset을 다시 하면
    # (t,p) 프레임쌍이 epoch마다 재추첨되어 frame_cache_dir 캐시가 의미 없어짐.
    # 여기서 한 번만 만들고 같은 ds 객체를 24 epoch 내내 재사용 -> idx당 (t,p) 고정
    # -> 캐시 히트가 그대로 재사용됨(1 epoch차만 디코딩, 이후 epoch는 디스크 읽기만).
    vcache = dict(zip_map)
    vcache.update(nexar_vcache)
    print("데이터셋(샘플링 플랜) 빌드 중...")
    train_ds = _build_dataset(train_df, vcache, args, True, ccd_zip_path)
    val_ds   = _build_dataset(val_df,   vcache, args, False, ccd_zip_path)
    if args.frame_cache_dir:
        print(f"frame_cache_dir={args.frame_cache_dir} (1 epoch차에 디코딩→캐시 채움, 이후 재사용)")

    # ── 모델 (사전학습 ConvNeXt-tiny → 6채널 팽창) ──────────────
    print(f"ImpactNet 초기화 (timm pretrained=True, backbone={args.backbone_name})...")
    model = ImpactNet(name=args.backbone_name, n_frames=2, pretrained=True,
                      drop_path=args.drop_path, head_hidden=512, pool="mean")
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    head_params = (list(model.norm.parameters()) + list(model.trunk.parameters()) +
                   list(model.head_col.parameters()) + list(model.head_ent.parameters()) +
                   list(model.head_side.parameters()) + list(model.head_evas.parameters()))
    back_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW([
        {"params": back_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=args.weight_decay)

    est_steps_per_epoch = len(train_ds) // args.batch
    total_steps = args.epochs * est_steps_per_epoch
    warmup      = min(args.warmup_steps, total_steps // 4)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    start_epoch = 1
    resume_path = args.resume or (str(Path(args.out).with_name(Path(args.out).stem + "_resume.pt"))
                                  if Path(args.out).with_name(Path(args.out).stem + "_resume.pt").exists() else None)
    if resume_path and Path(resume_path).exists():
        print(f"재개: {resume_path} 로드 중...")
        rck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model"])
        optimizer.load_state_dict(rck["optimizer"])
        scheduler.load_state_dict(rck["scheduler"])
        if rck.get("scaler") is not None and scaler is not None:
            scaler.load_state_dict(rck["scaler"])
        best_val    = rck.get("best_val", float("inf"))
        start_epoch = rck["epoch"] + 1
        print(f"  -> epoch {start_epoch}부터 재개 (best_val={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_ds, optimizer, scheduler,
                                     args, device, args.n_threads, scaler)
        print(f"  train_loss={train_loss:.4f}", flush=True)

        # 매 에폭 종료 시 무조건 재개용 체크포인트 저장 (모델+옵티마이저+스케줄러+epoch).
        # 크래시/환경 리셋으로 죽어도 --resume 없이 그냥 재실행하면 여기서 자동으로 이어짐
        # (out 경로 옆에 _resume.pt 있으면 자동 탐지).
        resume_ck = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val": best_val,
        }
        resume_save_path = str(Path(args.out).with_name(Path(args.out).stem + "_resume.pt"))
        torch.save(resume_ck, resume_save_path)
        print(f"  → 재개용 저장: {resume_save_path} (epoch {epoch})", flush=True)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics = eval_model(model, val_ds, args, device, args.n_threads)
            print(f"  val: {metrics}", flush=True)
            val_loss = metrics.get("loss", 9e9)
            ck = {
                "model": model.state_dict(),
                "cfg":   {"name": "convnext_tiny", "n_frames": 2,
                          "head_hidden": 512, "pool": "mean", "size": args.size,
                          "feat_dim": model.feat_dim},
                "args":  vars(args),
                "epoch": epoch,
                "val_metrics": metrics,
            }
            # "최신" 체크포인트는 매 검증 시점마다 무조건 저장 (나중에 다른 기준으로
            # 비교/선택할 수 있도록 — loss 최적값이 accuracy 최적값과 다를 수 있음)
            latest_path = str(Path(args.out).with_name(Path(args.out).stem + f"_epoch{epoch}.pt"))
            torch.save(ck, latest_path)
            print(f"  → 저장(latest): {latest_path}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ck, args.out)
                print(f"  → 저장(best): {args.out}")

    print(f"\n완료. best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
