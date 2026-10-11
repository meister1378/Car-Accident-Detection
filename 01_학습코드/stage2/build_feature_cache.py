"""
AI Hub + Nexar 영상 → Stage2 특징 캐시 (.npz) 빌드.

출력: cache_dir/{aihub,nexar}/{clip_id}.npz
      keys: t(float32), idx(int32), cnn(float16), cnn_flip(float16),
            flow(float16), fd(float16)

사용법:
  python scripts/build_feature_cache.py \
    --labels_csv data/aihub_labels.csv \
    --video_map  data/aihub_video_map.csv \
    --nexar_dir  .../nexar_collision_prediction \
    --backbone_pt C:/Users/kiot_/Downloads/stage2_backbone.pt \
    --cache_dir  data/stage2_ftF \
    --workers 4
"""
import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path
import concurrent.futures

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── 레퍼런스 stage2_features 상수 복제 ──────────────────────────────────────
HZ         = 10.0
CNN_SIZE   = (192, 192)
FLOW_SIZE  = (384, 288)
FLOW_POOL  = 2
GRAY_SIZE  = (192, 144)
N_BANDS    = 3
FD_NAMES   = ("fd_mean","fd_p90","fd_top","fd_bottom","fd_left","fd_centre","fd_right","fd_max")
FD_DIM     = len(FD_NAMES)


def clip_window(duration: float, event_sec: float | None,
                pre=11.0, post=5.0, max_full=45.0):
    if duration <= max_full or event_sec is None:
        return 0.0, max(0.0, duration - 1.0 / HZ)
    t0 = max(0.0, event_sec - pre)
    t1 = min(max(0.0, duration - 1.0 / HZ), event_sec + post)
    if t1 <= t0:
        return 0.0, max(0.0, duration - 1.0 / HZ)
    return t0, t1


def _fd_features(gray: np.ndarray) -> np.ndarray:
    T, H, W = gray.shape
    out = np.zeros((T, FD_DIM), np.float32)
    if T < 2:
        return out
    d = np.abs(gray[1:].astype(np.int16) - gray[:-1].astype(np.int16)).astype(np.float32)
    h2, w3 = H // 2, W // 3
    cols = [d[:, :, :w3], d[:, :, w3:2*w3], d[:, :, 2*w3:]]
    vals = np.stack([
        d.mean(axis=(1, 2)),
        np.percentile(d.reshape(T-1, -1), 90, axis=1),
        d[:, :h2].mean(axis=(1, 2)),
        d[:, h2:].mean(axis=(1, 2)),
        cols[0].mean(axis=(1, 2)),
        cols[1].mean(axis=(1, 2)),
        cols[2].mean(axis=(1, 2)),
        d.reshape(T-1, -1).max(axis=1),
    ], axis=1)
    out[1:] = np.log1p(vals)
    return out


# ── flow_features: 레퍼런스 s2flow.py 직접 import ──────────────────────────
_REF_STAGE2 = str(Path(__file__).resolve().parent.parent /
    "blackbox_reference/blackbox_dashcam_monitoring-3c1b1e3a40103d43fed4b0e935aee5608ad71f27/submission/model/stage2")
sys.path.insert(0, _REF_STAGE2)
import s2flow as _S2F
flow_features = _S2F.flow_features
flip_flow_features = _S2F.flip_features
FLOW_DIM = _S2F.FEATURE_DIM   # 70


# ── CPU worker: DIS flow + fd ─────────────────────────────────────────────────
def cpu_extract(video_path: str, times: np.ndarray, fps: float, n_frames: int):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    want = np.clip(np.round(np.asarray(times) * fps).astype(np.int64), 0, max(0, n_frames-1))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    lo, hi = int(want.min()), int(want.max())
    if lo > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    keep = {}
    need = set(want.tolist())
    i = lo
    while i <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if i in need:
            keep[i] = frame
        i += 1
    cap.release()
    if not keep:
        return None

    last = keep[min(keep)]
    T = len(want)
    bgr_list = []
    for w in want:
        f = keep.get(int(w))
        if f is None:
            f = last
        else:
            last = f
        bgr_list.append(f)

    rgb   = np.empty((T, CNN_SIZE[1],  CNN_SIZE[0],  3), np.uint8)
    rgb_s = np.empty((T, 256, 256, 3), np.uint8)          # for FTBandExtractor (size=256)
    gflow = np.empty((T, FLOW_SIZE[1], FLOW_SIZE[0]),  np.uint8)
    ggray = np.empty((T, GRAY_SIZE[1], GRAY_SIZE[0]), np.uint8)

    for i, f in enumerate(bgr_list):
        rgb[i]   = cv2.resize(f, CNN_SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1]
        rgb_s[i] = cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gflow[i] = cv2.resize(g, FLOW_SIZE, interpolation=cv2.INTER_AREA)
        ggray[i] = cv2.resize(g, GRAY_SIZE, interpolation=cv2.INTER_AREA)

    fh = FLOW_SIZE[1] // FLOW_POOL
    fw = FLOW_SIZE[0] // FLOW_POOL
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flows = np.zeros((T, 2, fh, fw), np.float32)
    prev = None
    for i in range(T):
        if prev is not None:
            fl = dis.calc(prev, gflow[i], None)
            fl = fl.reshape(fh, FLOW_POOL, fw, FLOW_POOL, 2).mean(axis=(1, 3))
            flows[i] = fl.transpose(2, 0, 1)
        prev = gflow[i]

    flow_feat = flow_features(torch.from_numpy(flows)).numpy().astype(np.float32)
    fd_feat   = _fd_features(ggray)

    return {"idx": want.astype(np.int32), "rgb": rgb, "rgb_s": rgb_s,
            "flow": flow_feat, "fd": fd_feat}


# ── GPU backbone extractor ────────────────────────────────────────────────────
class FTExtractor:
    def __init__(self, backbone_pt: str, device: str = "cuda", batch: int = 64):
        import torch
        ck = torch.load(backbone_pt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if _REF_STAGE2 not in sys.path:
            sys.path.insert(0, _REF_STAGE2)
        from stage2_backbone import FTBandExtractor
        self.ext = FTBandExtractor(backbone_pt, device=device, batch=batch, size=int(cfg.get("size", 256)))
        self.device = device
        self.batch = batch

    def __call__(self, rgb_s: np.ndarray, flip=False) -> np.ndarray:
        return self.ext(rgb_s, flip=flip)


# ── .npz 저장 ────────────────────────────────────────────────────────────────
def save_npz(cache_dir, corpus, clip_id, times, idx, cnn, cnn_flip, flow, fd):
    p = Path(cache_dir) / corpus / f"{clip_id}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, t=times.astype(np.float32), idx=idx.astype(np.int32),
             cnn=cnn.astype(np.float16), cnn_flip=cnn_flip.astype(np.float16),
             flow=flow.astype(np.float16), fd=fd.astype(np.float16))
    return p


# ── mp4 임시 추출 (zip → tempfile) ───────────────────────────────────────────
def extract_mp4_from_zip(zip_path: str, clip_id: str) -> str | None:
    try:
        with zipfile.ZipFile(zip_path) as z:
            mp4_name = f"{clip_id}.mp4"
            if mp4_name not in z.namelist():
                return None
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(z.read(mp4_name))
            tmp.close()
            return tmp.name
    except Exception:
        return None


# ── 클립 하나 처리 ────────────────────────────────────────────────────────────
def process_clip(clip_id, video_path, event_sec, corpus, cache_dir, extractor):
    p = Path(cache_dir) / corpus / f"{clip_id}.npz"
    if p.exists():
        return True  # 이미 있음

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    cap.release()

    t0, t1 = clip_window(duration, event_sec)
    n = max(1, int(round((t1 - t0) * HZ)) + 1)
    times = (t0 + np.arange(n) / HZ).astype(np.float64)

    data = cpu_extract(video_path, times, fps, n_frames)
    if data is None:
        return False

    cnn      = extractor(data["rgb_s"], flip=False)
    cnn_flip = extractor(data["rgb_s"], flip=True)

    save_npz(cache_dir, corpus, clip_id, times, data["idx"],
             cnn, cnn_flip, data["flow"], data["fd"])
    return True


# ── CPU/GPU 파이프라인 병렬화용: CPU 파트(디코딩+광류)만 워커 스레드에서 ────────
def prepare_clip_cpu(clip_id, video_path, event_sec, corpus, cache_dir, cleanup_path=None):
    """cv2 디코딩 + DIS 광류 계산(전부 GIL을 놓는 네이티브 호출)만 수행.
    ThreadPoolExecutor 워커에서 호출 -> 여러 클립을 동시에 디코딩해서
    (이후 GPU forward는 메인 스레드에서 순차 처리) 전체 처리량을 늘림."""
    try:
        p = Path(cache_dir) / corpus / f"{clip_id}.npz"
        if p.exists():
            return (clip_id, "skip", None, None)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_frames / fps
        cap.release()
        t0, t1 = clip_window(duration, event_sec)
        n = max(1, int(round((t1 - t0) * HZ)) + 1)
        times = (t0 + np.arange(n) / HZ).astype(np.float64)
        data = cpu_extract(video_path, times, fps, n_frames)
        if data is None:
            return (clip_id, "fail", None, None)
        data.pop("rgb", None)  # CNN_SIZE(192)판은 아래서 안 씀 -> 메모리만 차지, 버림
        return (clip_id, "ok", data, times)
    except Exception:
        return (clip_id, "fail", None, None)
    finally:
        if cleanup_path is not None:
            try:
                os.unlink(cleanup_path)
            except Exception:
                pass


def run_clips_parallel(rows, corpus, cache_dir, extractor, n_workers, desc):
    """rows: [(clip_id, video_path_or_None(=실패), event_sec, cleanup_path_or_None), ...]
    CPU 파트는 ThreadPoolExecutor로 병렬, GPU forward+저장은 메인 스레드에서 순차.

    주의: 전체 rows를 한 번에 submit하고 as_completed로만 돌리면, futures 리스트가
    이미 처리된 Future까지 계속 들고 있어서(각 Future가 내부에 큰 numpy 결과를 보관)
    처리량과 무관하게 메모리가 계속 쌓인다(실측: 16000여 클립 처리 중 84GB까지 증가).
    그래서 청크 단위로 submit -> 그 청크 결과를 전부 소비 -> futures/청크를 버려서
    GC가 회수하게 함(윈도우를 작게 유지)."""
    ok = fail = 0
    valid_rows = [r for r in rows if r[1] is not None]
    fail += len(rows) - len(valid_rows)
    CHUNK = max(n_workers * 4, 16)
    pbar = tqdm(total=len(valid_rows), desc=desc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        for ci in range(0, len(valid_rows), CHUNK):
            chunk = valid_rows[ci:ci + CHUNK]
            futures = [ex.submit(prepare_clip_cpu, clip_id, video_path, event_sec,
                                  corpus, cache_dir, cleanup_path)
                       for clip_id, video_path, event_sec, cleanup_path in chunk]
            for fut in concurrent.futures.as_completed(futures):
                clip_id, status, data, times = fut.result()
                if status == "skip":
                    ok += 1
                elif status == "fail":
                    fail += 1
                else:
                    try:
                        cnn      = extractor(data["rgb_s"], flip=False)
                        cnn_flip = extractor(data["rgb_s"], flip=True)
                        save_npz(cache_dir, corpus, clip_id, times, data["idx"],
                                 cnn, cnn_flip, data["flow"], data["fd"])
                        ok += 1
                    except Exception:
                        fail += 1
                pbar.update(1)
            futures.clear()
    pbar.close()
    return ok, fail


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv",   default="data/aihub_labels.csv")
    ap.add_argument("--video_map",    default="data/aihub_video_map.csv")
    ap.add_argument("--nexar_dir",    default="C:/Users/kiot_/Desktop/competition/blackbox_new/Black_Box/DATASET/nexar_collision_prediction")
    ap.add_argument("--backbone_pt",  default="C:/Users/kiot_/Downloads/stage2_backbone.pt")
    ap.add_argument("--cache_dir",    default="data/stage2_ftF")
    ap.add_argument("--workers",      type=int, default=1)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_clips",    type=int, default=0, help="0=전체, N=테스트용 상한")
    ap.add_argument("--row_start",    type=int, default=0,   help="aihub df 시작 행 (포함)")
    ap.add_argument("--row_end",      type=int, default=-1,  help="aihub df 끝 행 (미포함), -1=전체")
    ap.add_argument("--downloads_root", default=None,
                    help="'/downloads' 접두사를 로컬 경로로 치환 (도커 밖에서 실행할 때)")
    args = ap.parse_args()

    print(f"[build_feature_cache] device={args.device}, workers={args.workers}, row_start={args.row_start}, row_end={args.row_end}")
    extractor = FTExtractor(args.backbone_pt, device=args.device)

    # ── AI Hub ───────────────────────────────────────────────────────────────
    df_lab = pd.read_csv(args.labels_csv)
    df_map = pd.read_csv(args.video_map)
    df = df_lab.merge(df_map, on="clip_id", how="inner")
    if args.downloads_root:
        df["zip_path"] = df["zip_path"].apply(
            lambda v: v.replace("/downloads", args.downloads_root, 1) if str(v).startswith("/downloads") else v)

    # 범위 분할
    if args.row_end == -1:
        df = df.iloc[args.row_start:]
    else:
        df = df.iloc[args.row_start:args.row_end]
    if args.max_clips > 0:
        df = df.head(args.max_clips)
    print(f"[aihub] {len(df)}개 클립 처리 (행 {args.row_start}~{args.row_end})")
    rows = []
    for _, row in df.iterrows():
        is_flat_mp4 = str(row["zip_path"]).lower().endswith(".mp4") and os.path.exists(row["zip_path"])
        if is_flat_mp4:
            rows.append((row["clip_id"], row["zip_path"], row["collision_sec"], None))
        else:
            tmp_path = extract_mp4_from_zip(row["zip_path"], row["clip_id"])
            rows.append((row["clip_id"], tmp_path, row["collision_sec"], tmp_path))
    ok, fail = run_clips_parallel(rows, "aihub", args.cache_dir, extractor, args.workers, "AI Hub")

    print(f"[aihub] 완료: {ok}개 성공, {fail}개 실패")

    # ── Nexar ────────────────────────────────────────────────────────────────
    nexar_dir = Path(args.nexar_dir)
    pos_dir = nexar_dir / "train" / "positive"
    # train/positive/metadata.csv 에 time_of_event 있음
    event_map = {}
    meta_path = pos_dir / "metadata.csv"
    if meta_path.exists():
        try:
            tmp = pd.read_csv(meta_path)
            for _, r in tmp.dropna(subset=["time_of_event"]).iterrows():
                cid = Path(str(r["file_name"])).stem
                event_map[cid] = float(r["time_of_event"])
        except Exception:
            pass
    mp4s = sorted(pos_dir.glob("*.mp4"))
    print(f"[nexar] {len(mp4s)}개 클립, {len(event_map)}개 time_of_event 보유")

    if args.max_clips > 0:
        mp4s = mp4s[:args.max_clips]
    rows = [(mp4.stem, str(mp4), event_map.get(mp4.stem), None) for mp4 in mp4s]
    ok, fail = run_clips_parallel(rows, "nexar", args.cache_dir, extractor, args.workers, "Nexar")

    print(f"[nexar] 완료: {ok}개 성공, {fail}개 실패")
    print(f"\n캐시 디렉토리: {args.cache_dir}")


if __name__ == "__main__":
    main()
