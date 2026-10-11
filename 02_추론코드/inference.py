"""
DACON BlackBox 대회 제출용 추론 엔트리포인트
predict_stage1/2/3 은 AST 검사를 위해 top-level FunctionDef로 정의됨.
"""
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.environ.setdefault("STAGE2_LANE_BACKEND", "ccd_style_multitask")
os.environ.setdefault("STAGE2_ENTRY_ALGO", "first_match")
os.environ.setdefault("STAGE2_EVASION_TTC", "1")
os.environ.setdefault("STAGE2_ENTRY_FRAME_DEBUG_OVERRIDE", "0")
os.environ.setdefault("STAGE2_ENTRY_DIRECTION_CORRECT", "1")
os.environ.setdefault("STAGE2_ENTRY_TRAFFIC_CLUSTER", "1")
os.environ.setdefault("STAGE2_CCD_CKPT_FILENAME", "unet_r18_unstructured_finetune.pt")
os.environ.setdefault("STAGE2_EVASION_OCC_WINDOW", "1")
os.environ.setdefault("STAGE2_USE_ADABN", "0")
os.environ.setdefault("STAGE1_USE_SEQUENCE_NUDGE", "1")
os.environ.setdefault("STAGE2_EVASION_EGO_NEAR", "0")
os.environ.setdefault("STAGE1_USE_MVIT_AIHUB", "1")
os.environ.setdefault("STAGE3_USE_FINAL_CANDIDATE", "0")
os.environ.setdefault("STAGE3_USE_COMMA_FULL_NOHFLIP", "0")
os.environ.setdefault("STAGE3_USE_COMMA80_HFLIP", "0")
os.environ.setdefault("STAGE3_USE_COMMA80_STEERFIX", "0")
os.environ.setdefault("STAGE3_USE_L2D", "1")
os.environ.setdefault("STAGE2_EVASION_BARRIER", "0")
os.environ.setdefault("STAGE2_EVASION_DRIVABLE", "0")
os.environ.setdefault("STAGE2_ENTRY_BIAS", "0.70")
os.environ.setdefault("L2D_BATCH", "64")
os.environ.setdefault("STAGE3_L2D_CKPT_FILENAME", "mvit_accel051_win05_minf3.pt")
os.environ.setdefault("STAGE3_USE_COMMA_FULL_THRESH_CALIB", "0")
os.environ.setdefault("STAGE3_USE_COMMA_ONLY_V2", "0")


def _real_cpu_quota() -> int:
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        if quota > 0:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read())
            return max(1, int(quota / period))
    except Exception:
        pass
    return os.cpu_count() or 1


_NUM_THREADS = str(_real_cpu_quota())
for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, _NUM_THREADS)

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch

cv2.setNumThreads(int(_NUM_THREADS))
torch.set_num_threads(int(_NUM_THREADS))

from train_stage1 import predict_stage1 as _predict_stage1

# ── Stage 2: 레퍼런스 구현 ──────────────────────────────────────────────────
_REF_S2_SUBMISSION = (
    Path(__file__).resolve().parent
    / "blackbox_reference"
    / "blackbox_dashcam_monitoring-3c1b1e3a40103d43fed4b0e935aee5608ad71f27"
    / "submission"
)
if str(_REF_S2_SUBMISSION) not in sys.path:
    sys.path.insert(0, str(_REF_S2_SUBMISSION))

from model.stage2.predict import predict as _ref_s2_predict  # noqa: E402


def predict_stage1(data_dir: str, model_dir: str) -> pd.DataFrame:
    return _predict_stage1(data_dir, model_dir)


def predict_stage2(data_dir: str, model_dir: str) -> pd.DataFrame:
    import shutil

    # fold 파일 복사 (앙상블 안정화)
    _stage2_code = _REF_S2_SUBMISSION / "model" / "stage2"
    for fname in ("v2_fold0.pt", "v2_fold1.pt"):
        src = Path(model_dir) / fname
        dst = _stage2_code / fname
        if src.is_file() and not dst.exists():
            try:
                shutil.copy2(str(src), str(dst))
            except OSError:
                pass  # 서버 디렉터리가 읽기 전용이면 무시 — cwd 기반 경로로 폴백

    return _ref_s2_predict(Path(data_dir), Path(model_dir))




def predict_stage3(data_dir, model_dir):
    from pathlib import Path
    import importlib.util, sys, torch
    directory = Path(model_dir).resolve()
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('_submission_stage3_adapter', directory / 'adapter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.predict_stage3(data_dir, directory)
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlackBox 3-Stage 추론")
    parser.add_argument("--data_dir",  required=True,
                        help="테스트 데이터 루트 (stage1/, stage2/, stage3/ 포함)")
    parser.add_argument("--model_dir", required=True,
                        help="모델 루트 (stage1/best.pt, stage2/best.pt, stage3/best.pt)")
    parser.add_argument("--out_dir",   default="./submission",
                        help="결과 CSV 저장 경로")
    parser.add_argument("--stage",     default="all",
                        choices=["all", "1", "2", "3"],
                        help="실행할 Stage (기본: all)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data_root  = args.data_dir
    model_root = args.model_dir

    if args.stage in ("all", "1"):
        print("=" * 50)
        print("[Stage 1] 재촬영 탐지")
        df1 = predict_stage1(
            os.path.join(data_root, "stage1"),
            os.path.join(model_root, "stage1"),
        )
        out1 = os.path.join(args.out_dir, "stage1_result.csv")
        df1.to_csv(out1, index=False)
        print(f"  저장: {out1} ({len(df1)}건)")
        print(df1.head())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if args.stage in ("all", "2"):
        print("=" * 50)
        print("[Stage 2] 사고 시점·상황 분석")
        df2 = predict_stage2(
            os.path.join(data_root, "stage2"),
            os.path.join(model_root, "stage2"),
        )
        out2 = os.path.join(args.out_dir, "stage2_result.csv")
        df2.to_csv(out2, index=False)
        print(f"  저장: {out2} ({len(df2)}건)")
        print(df2.head())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if args.stage in ("all", "3"):
        print("=" * 50)
        print("[Stage 3] 차량 거동 분류")
        df3 = predict_stage3(
            os.path.join(data_root, "stage3"),
            os.path.join(model_root, "stage3"),
        )
        out3 = os.path.join(args.out_dir, "stage3_result.csv")
        df3.to_csv(out3, index=False)
        print(f"  저장: {out3} ({len(df3)}건)")
        print(df3.head())

    print("\n추론 완료.")


