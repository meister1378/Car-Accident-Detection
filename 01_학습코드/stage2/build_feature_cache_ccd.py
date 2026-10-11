"""
CCD(Car Crash Dataset) -> Stage2 특징 캐시(.npz) 빌드.

CCD는 mp4가 아니라 클립당 50장 JPG 시퀀스(10fps 가정, 5초)로 배포됨.
Crash_Table.csv의 frame_1..frame_50 (0/1)에서 첫 1이 뜨는 프레임이 실제 사고 시작
프레임(real label) - DACON 공식 baseline 5개 영상(000001~5)이 정확히 이 라벨과
0프레임 오차로 일치함을 확인함.

baseline 평가에 쓰는 000001~000005는 반드시 학습에서 제외.

출력: cache_dir/ccd/{clip_id}.npz  (build_feature_cache.py와 동일 스키마)
      + data/ccd_labels.csv (clip_id, collision_sec, egoinvolve)
"""
import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from build_feature_cache import (
    CNN_SIZE, FLOW_SIZE, FLOW_POOL, GRAY_SIZE, flow_features, _fd_features, FTExtractor,
    save_npz,
)

HZ = 10.0  # CCD 표준: 50프레임/5초 = 10fps
EXCLUDE_CLIPS = {"000001", "000002", "000003", "000004", "000005"}  # baseline 평가셋


def cpu_extract_ccd(frames_bgr: list) -> dict:
    cv2.setNumThreads(1)
    T = len(frames_bgr)
    idx = np.arange(T, dtype=np.int32)

    rgb   = np.empty((T, CNN_SIZE[1],  CNN_SIZE[0],  3), np.uint8)
    rgb_s = np.empty((T, 256, 256, 3), np.uint8)
    gflow = np.empty((T, FLOW_SIZE[1], FLOW_SIZE[0]),  np.uint8)
    ggray = np.empty((T, GRAY_SIZE[1], GRAY_SIZE[0]), np.uint8)

    for i, f in enumerate(frames_bgr):
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

    import torch
    flow_feat = flow_features(torch.from_numpy(flows)).numpy().astype(np.float32)
    fd_feat = _fd_features(ggray)

    return {"idx": idx, "rgb": rgb, "rgb_s": rgb_s, "flow": flow_feat, "fd": fd_feat}


def process_ccd_clip(zf: zipfile.ZipFile, vidname: str, cache_dir: str, extractor) -> bool:
    p = Path(cache_dir) / "ccd" / f"{vidname}.npz"
    if p.exists():
        return True

    frames = []
    for i in range(1, 51):
        name = f"CrashBest/C_{vidname}_{i:02d}.jpg"
        try:
            with zf.open(name) as f:
                buf = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return False
            frames.append(img)
        except KeyError:
            return False

    times = (np.arange(50) / HZ).astype(np.float64)
    data = cpu_extract_ccd(frames)

    cnn      = extractor(data["rgb_s"], flip=False)
    cnn_flip = extractor(data["rgb_s"], flip=True)

    save_npz(cache_dir, "ccd", vidname, times, data["idx"], cnn, cnn_flip, data["flow"], data["fd"])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccd_zip", default="C:/Users/Server/Downloads/ccd.zip")
    ap.add_argument("--backbone_pt", default="models/stage2/bb_new_geo.pt")
    ap.add_argument("--cache_dir", default="data/stage2_ftF")
    ap.add_argument("--out_labels", default="data/ccd_labels.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--row_start", type=int, default=0)
    ap.add_argument("--row_end", type=int, default=-1)
    args = ap.parse_args()

    extractor = FTExtractor(args.backbone_pt, device=args.device)

    zf = zipfile.ZipFile(args.ccd_zip)
    with zf.open("Crash_Table.csv") as f:
        data = f.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(data))
    rows = list(reader)
    print(f"[ccd] total rows in Crash_Table.csv: {len(rows)}")

    label_rows = []
    ok = fail = skipped = 0
    if args.row_end == -1:
        rows = rows[args.row_start:]
    else:
        rows = rows[args.row_start:args.row_end]
    if args.max_clips > 0:
        rows = rows[:args.max_clips]

    for r in tqdm(rows, desc="CCD"):
        vidname = r["vidname"]
        if vidname in EXCLUDE_CLIPS:
            skipped += 1
            continue
        first_one = None
        for i in range(1, 51):
            if r[f"frame_{i}"] == "1":
                first_one = i
                break
        if first_one is None:
            skipped += 1
            continue
        collision_sec = (first_one - 1) / HZ  # 0-indexed

        success = process_ccd_clip(zf, vidname, args.cache_dir, extractor)
        if success:
            ok += 1
            label_rows.append({
                "clip_id": vidname,
                "collision_sec": round(collision_sec, 4),
                "egoinvolve": r["egoinvolve"],
                "weather": r["weather"],
                "timing": r["timing"],
            })
        else:
            fail += 1

    df = pd.DataFrame(label_rows)
    df.to_csv(args.out_labels, index=False)
    print(f"[ccd] 완료: {ok}개 성공, {fail}개 실패, {skipped}개 스킵(baseline 5개 제외 등)")
    print(f"[ccd] 라벨 저장: {args.out_labels} ({len(df)} rows)")


if __name__ == "__main__":
    main()
