"""evasion_space를 실제 SSDLite 탐지기로 재계산 (stage2_side.py의 free_space_counts 재현).

레퍼런스 문서: "a lane counts as blocked if any frame of the second before impact
has a road user in it, which is the rule the AI Hub label was derived under."
AI Hub 자체 bbox 주석(isObjectOthers)은 너무 희소해 이 규칙을 재현 못 함 —
실제로는 SSDLite320 탐지기를 영상 프레임에 직접 돌려서 만든 라벨이었음
(pseudo-label 대비 80% 일치, 기존 AI-Hub-주석 기반 규칙 대비 65%보다 크게 개선).

중단 시 이어하기 지원.
"""
import argparse
import os
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models.detection as D

STAGE2_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "blackbox_reference",
    "blackbox_dashcam_monitoring-3c1b1e3a40103d43fed4b0e935aee5608ad71f27",
    "submission", "model", "stage2")
sys.path.insert(0, STAGE2_DIR)
import stage2_entry as S2E  # noqa: E402

COCO_ROAD_USER = frozenset({1, 2, 3, 4, 6, 8})
EVASION_SCORE = 0.30
EVASION_WINDOW = (-1.0, 0.0)
EVASION_FOE_WINDOW = (-2.5, 0.0)
EVASION_N_FRAMES = 5
EVASION_MAX_NEIGHBOUR, EVASION_MAX_SCENE = 0, 2
DEFAULT_FPS = 15.0


def neighbour_counts(boxes, labels, geom, W, H, margin=0.15):
    n_l = n_r = 0
    for b, lb in zip(boxes, labels):
        if int(lb) not in COCO_ROAD_USER:
            continue
        u = 0.5 * (float(b[0]) + float(b[2])) / W
        v = min(1.0, float(b[3]) / H)
        if v <= geom.v_vp:
            continue
        half = max(geom.half(v), 1e-6)
        if abs(u - (geom.u_vp - 2.0 * half)) <= half * (1.0 + margin):
            n_l += 1
        if abs(u - (geom.u_vp + 2.0 * half)) <= half * (1.0 + margin):
            n_r += 1
    return n_l, n_r


def free_space_counts(model, device, frames_bgr, collision_idx, fps):
    n = len(frames_bgr)
    numbers = np.arange(n, dtype=np.float64)

    def pick(window, count):
        offs = np.linspace(window[0], window[1], count)
        want = float(collision_idx) + offs * float(fps)
        return np.unique(np.clip(np.searchsorted(numbers, want), 0, n - 1))

    foe_idx = pick(EVASION_FOE_WINDOW, 18)
    grays = []
    for j in foe_idx:
        g = cv2.cvtColor(frames_bgr[int(j)], cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(g, (480, 270), interpolation=cv2.INTER_AREA))
    if len(grays) < 4:
        return None
    geom = S2E.estimate_foe(grays)

    rgb = []
    for j in pick(EVASION_WINDOW, EVASION_N_FRAMES):
        rgb.append(cv2.cvtColor(frames_bgr[int(j)], cv2.COLOR_BGR2RGB))
    if len(rgb) < 2:
        return None
    H, W = rgb[0].shape[:2]
    n_l = n_r = 0
    totals = []
    with torch.inference_mode():
        ten = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)
               .to(device).float().div_(255.0) for f in rgb]
        preds = model(ten)
        for pr in preds:
            k = pr["scores"] >= EVASION_SCORE
            lab = pr["labels"][k].cpu().numpy()
            a, b = neighbour_counts(pr["boxes"][k].cpu().numpy(), lab, geom, W, H)
            n_l, n_r = max(n_l, a), max(n_r, b)
            totals.append(sum(1 for x in lab if int(x) in COCO_ROAD_USER))
        del ten, preds
    return int(n_l), int(n_r), int(np.median(totals)) if totals else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default="data/aihub_labels.csv")
    ap.add_argument("--video_map", default="data/aihub_video_map.csv")
    ap.add_argument("--ssdlite_ckpt",
                    default="models/stage2_reference/ssdlite320_mobilenet_v3_large_coco-a79551df.pth")
    ap.add_argument("--out", default="data/aihub_evasion_ssdlite.csv")
    ap.add_argument("--downloads_root", default=None,
                    help="'/downloads' 접두사를 로컬 경로로 치환 (도커 밖에서 실행할 때)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = D.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    model.load_state_dict(torch.load(args.ssdlite_ckpt, map_location="cpu", weights_only=True))
    model = model.eval().to(device)
    print(f"detector loaded on {device}", flush=True)

    labels = pd.read_csv(args.labels_csv)
    vmap = pd.read_csv(args.video_map).set_index("clip_id")["zip_path"].to_dict()
    if args.downloads_root:
        vmap = {k: v.replace("/downloads", args.downloads_root, 1) if v.startswith("/downloads") else v
                for k, v in vmap.items()}
    df = labels[labels["clip_id"].isin(vmap.keys())].reset_index(drop=True)
    print(f"total clips to process: {len(df)}", flush=True)

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["clip_id"].tolist())
        print(f"resuming: {len(done)} already done", flush=True)

    header_needed = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="", encoding="utf-8")

    n_ok = n_fail = 0
    for i, row in df.iterrows():
        cid = row["clip_id"]
        if cid in done:
            continue
        evasion = None
        try:
            with zipfile.ZipFile(vmap[cid]) as z:
                data = z.read(f"{cid}.mp4")
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(data)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            frames_bgr = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames_bgr.append(f)
            cap.release()
            os.unlink(tmp.name)
            if len(frames_bgr) >= 10:
                r = free_space_counts(model, device, frames_bgr, int(row["collision_frame_idx"]),
                                      DEFAULT_FPS)
                if r is not None:
                    n_l, n_r, n_all = r
                    evasion = int(min(n_l, n_r) <= EVASION_MAX_NEIGHBOUR and n_all <= EVASION_MAX_SCENE)
            del frames_bgr
        except Exception as exc:                                 # noqa: BLE001
            print(f"[{i}/{len(df)}] {cid}: EXCEPTION {type(exc).__name__}: {exc}", flush=True)

        if evasion is None:
            n_fail += 1
            evasion = int(row["evasion_space"])  # 실패 시 기존 값 유지
            source = "fallback_geo"
        else:
            n_ok += 1
            source = "ssdlite"

        if device == "cuda":
            torch.cuda.empty_cache()

        pd.DataFrame([{"clip_id": cid, "evasion_space": evasion, "source": source}]) \
            .to_csv(fout, header=header_needed, index=False)
        header_needed = False
        fout.flush()

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(df)}] ok={n_ok} fail={n_fail}", flush=True)

    fout.close()
    print(f"done. ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
