"""
AI Hub 095 이미지 JSON → Stage2 학습 라벨 derive.

입력:  TL/VL_차대차_이미지_*.zip  (per-frame bbox + isObjectA/B)
출력:  data/aihub_labels.csv
       data/aihub_video_map.csv  (clip_id → mp4 zip 경로)
"""
import argparse
import json
import math
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

MIN_AREA_FRAC  = 0.001   # isObjectB 최소 면적 비율 (W*H 대비)
DEFAULT_FPS    = 15.0    # AI Hub 영상 기본 FPS (실측: 15.0fps, 150프레임/클립)

# ── entry_frame/entry_side 기하 상수 (레퍼런스 stage2_entry.py에서 복구한
#    원본 stage2_aihub.derive 로직: 소실점 삼각형 차선 모델) ──────────────
LANE_K       = 0.48   # stage2_aihub의 실효 차선폭(2.1m) 기준 기울기
U_VP_PRIOR   = 0.50
V_VP_PRIOR   = 0.52   # stage2_aihub의 V_HORIZON
VP_CLAMP_U   = (0.35, 0.65)
VP_CLAMP_V   = (0.35, 0.72)
ENTRY_MARGIN = 0.0
MIN_RUN      = 2
MIN_BOX_SIZE = 0.02

# ─────────────────────────────────────────────────────────────────────────────
def _bbox_area(bbox):
    return max(0, bbox[2]) * max(0, bbox[3])


def _bbox_cx(bbox):
    return bbox[0] + bbox[2] / 2.0


def _bbox_cy(bbox):
    return bbox[1] + bbox[3] / 2.0


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _estimate_lane_geom(all_boxes, width, height):
    """전체 차량 bbox → 소실점(u_vp, v_vp) 추정 (stage2_aihub.estimate_geometry 재현).

    모든 차량은 도로 위에 있다는 가정 하에, y좌표 하위 5%(가장 먼 차량)를
    수평선으로, 그 근방 20% 박스들의 x 중앙값을 차선 중심으로 삼는다.
    """
    if width <= 0 or height <= 0 or len(all_boxes) < 30:
        return U_VP_PRIOR, V_VP_PRIOR
    c = sorted(((y + h / 2.0) / height, (x + w / 2.0) / width) for x, y, w, h in all_boxes)
    v_est = c[int(0.05 * (len(c) - 1))][0]
    near = c[:max(1, int(0.20 * len(c)))]
    u_est = sum(p[1] for p in near) / len(near)
    conf = min(1.0, len(c) / 200.0) * 0.5
    u_vp = min(max(U_VP_PRIOR + conf * (u_est - U_VP_PRIOR), VP_CLAMP_U[0]), VP_CLAMP_U[1])
    v_vp = min(max(V_VP_PRIOR + conf * (v_est - V_VP_PRIOR), VP_CLAMP_V[0]), VP_CLAMP_V[1])
    return u_vp, v_vp


def _signed_pos(u, v, u_vp, v_vp):
    """차선 반폭 단위의 횡방향 위치: 0=중심선, ±1=차선 경계."""
    half = max(0.0, LANE_K * (v - v_vp))
    if half <= 1e-6:
        return math.copysign(99.0, u - u_vp)
    return (u - u_vp) / half


def _inner_bottom(bbox, u_vp, width, height):
    """bbox 하단 모서리 중 차선 중심에 가장 가까운 쪽 (u, v)."""
    x, y, w, h = bbox
    v = min(1.0, (y + h) / float(height))
    lo, hi = x / float(width), (x + w) / float(width)
    if hi < u_vp:
        u = hi
    elif lo > u_vp:
        u = lo
    else:
        u = u_vp
    return u, v


def _entry_from_track(track, u_vp, v_vp, width, height, collision_fidx):
    """isObjectB 트랙이 차선 삼각형에 들어온 구간을 찾는다 (stage2_aihub.derive 재현).

    ``track``: [(frame_idx, bbox), ...] fidx 오름차순.
    반환: entry_fidx 또는 None (well-defined 아님 → 폴백 필요).
    """
    n = len(track)
    if n == 0:
        return None
    pos, size = [], []
    for _, bbox in track:
        u, v = _inner_bottom(bbox, u_vp, width, height)
        pos.append(_signed_pos(u, v, u_vp, v_vp))
        size.append(math.sqrt(max(bbox[2], 0) * max(bbox[3], 0)) / float(height))
    inside = [abs(pos[i]) <= 1.0 + ENTRY_MARGIN and size[i] >= MIN_BOX_SIZE for i in range(n)]
    idxs = [i for i, ok in enumerate(inside) if ok]
    if not idxs:
        return None

    runs, start, prev = [], idxs[0], idxs[0]
    for i in idxs[1:]:
        if i - prev > 1:
            if prev - start + 1 >= MIN_RUN:
                runs.append((start, prev))
            start = i
        prev = i
    if prev - start + 1 >= MIN_RUN:
        runs.append((start, prev))
    if not runs:
        return None

    ci = min(range(n), key=lambda i: abs(track[i][0] - collision_fidx))
    run = next(((a, b) for a, b in runs if a <= ci <= b), None)
    if run is None:
        prior_runs = [r for r in runs if r[1] <= ci]
        run = prior_runs[-1] if prior_runs else runs[0]
    a, _b = run
    if a == 0:
        return None  # 트랙 시작 전에 이미 진입해 있었음 → well-defined 아님
    return track[a][0]


def _entry_side_from_anchors(track, entry_fidx, u_vp, width, n_anchor=5):
    """entry 직전 최대 5개 앵커 프레임에서 (bbox 중심 - u_vp) 중앙값 부호로 LEFT/RIGHT."""
    entry_i = next((i for i, (fidx, _) in enumerate(track) if fidx == entry_fidx), None)
    if entry_i is None:
        return None
    lo = max(0, entry_i - n_anchor)
    anchors = track[lo:entry_i] or [track[entry_i]]
    offs = sorted(_bbox_cx(bbox) / width - u_vp for _, bbox in anchors)
    mid = len(offs) // 2
    median = offs[mid] if len(offs) % 2 else (offs[mid - 1] + offs[mid]) / 2.0
    return "LEFT" if median < 0 else "RIGHT"


# ── evasion_space 상수 (stage2_side.py에서 복구한 "the label's own rule":
#    "away lane == 0" — entry_side 반대쪽(도망갈 방향) 차로에 충돌 1초 전
#    구간 동안 사고관련(isObjectOthers) 차량이 하나도 없으면 evasion=1) ──
EVASION_WINDOW_SEC = 1.0


def _evasion_space(frames, collision_fidx, u_vp, v_vp, width, height, entry_side, fps=DEFAULT_FPS):
    """충돌 1초 전~충돌 프레임 윈도우: away lane(진입 반대쪽)에 차량이 없으면 1."""
    lo = collision_fidx - int(round(EVASION_WINDOW_SEC * fps))
    window = [f for f in frames if lo <= f["frame_idx"] <= collision_fidx]
    if not window:
        window = [f for f in frames if f["frame_idx"] == collision_fidx]
    if not window:
        return 1

    away_is_left = (entry_side == "RIGHT")  # 위협이 온 쪽의 반대가 도망갈 방향
    n_away_per_frame = []
    for f in window:
        n_away = 0
        for o in f["objects"]:
            if not o.get("isObjectOthers"):
                continue  # "사고관련 객체"만 카운트 (무관한 배경 차량 제외)
            bbox = o["bbox"]
            cx = _bbox_cx(bbox) / float(width)
            cy = _bbox_cy(bbox) / float(height)
            half = max(0.0, LANE_K * (cy - v_vp))
            on_left = cx < u_vp - half
            on_right = cx > u_vp + half
            if (away_is_left and on_left) or (not away_is_left and on_right):
                n_away += 1
        n_away_per_frame.append(n_away)

    return int(max(n_away_per_frame) == 0)


def derive_clip_labels(frames: list[dict]) -> dict | None:
    """
    frames: sequence_frame_number 순으로 정렬된 프레임 dict 목록
    각 dict: {frame_idx, W, H, objects:[{bbox,isObjectB,isObjectOthers}]}
    """
    if len(frames) < 10:
        return None

    W, H = frames[0]["W"], frames[0]["H"]
    min_area = MIN_AREA_FRAC * W * H

    # ── isObjectA/isObjectB bbox 시계열 ──────────────────────────────────
    # A/B는 항상 같이 나오는 게 아니라 시나리오에 따라 하나만 있는 경우가 많음
    # (샘플 조사 결과 A만/B만 각각 30~40%, 둘 다 있는 경우 25~30%).
    a_frames, b_frames = [], []
    for f in frames:
        a_bboxes = [o["bbox"] for o in f["objects"] if o.get("isObjectA") and _bbox_area(o["bbox"]) > min_area]
        b_bboxes = [o["bbox"] for o in f["objects"] if o.get("isObjectB") and _bbox_area(o["bbox"]) > min_area]
        if a_bboxes:
            best = max(a_bboxes, key=_bbox_area)
            a_frames.append({"fidx": f["frame_idx"], "bbox": best, "area": _bbox_area(best)})
        if b_bboxes:
            best = max(b_bboxes, key=_bbox_area)
            b_frames.append({"fidx": f["frame_idx"], "bbox": best, "area": _bbox_area(best)})

    if len(a_frames) < 3 and len(b_frames) < 3:
        return None

    # ── collision_frame: A/B 둘 다 있는 프레임이 있으면 IoU(접촉) 최대 프레임,
    #    없으면(둘 중 하나만 존재) 그 객체의 면적 최대 프레임 ──────────────
    a_by_fidx = {af["fidx"]: af["bbox"] for af in a_frames}
    b_by_fidx = {bf["fidx"]: bf["bbox"] for bf in b_frames}
    common_fidx = sorted(set(a_by_fidx) & set(b_by_fidx))
    if common_fidx:
        collision_fidx = max(common_fidx, key=lambda fi: _iou(a_by_fidx[fi], b_by_fidx[fi]))
    elif b_frames:
        collision_fidx = max(b_frames, key=lambda x: x["area"])["fidx"]
    else:
        collision_fidx = max(a_frames, key=lambda x: x["area"])["fidx"]
    collision_sec = collision_fidx / DEFAULT_FPS

    # ── entry_frame/entry_side용 "피해차량" 트랙: B가 더 충분하면 B, 아니면 A ──
    target_frames = b_frames if len(b_frames) >= len(a_frames) else a_frames
    b_frames = target_frames  # 아래 폴백 블록이 참조하는 이름 유지

    # ── entry_frame/entry_side: 소실점 차선 삼각형 진입 시점 (stage2_aihub.derive 복구) ──
    track = sorted(((bf["fidx"], bf["bbox"]) for bf in target_frames), key=lambda t: t[0])
    all_boxes = [o["bbox"] for f in frames for o in f["objects"]]
    u_vp, v_vp = _estimate_lane_geom(all_boxes, W, H)
    entry_fidx = _entry_from_track(track, u_vp, v_vp, W, H, collision_fidx)
    entry_side = (_entry_side_from_anchors(track, entry_fidx, u_vp, W)
                  if entry_fidx is not None else None)

    if entry_fidx is None or entry_fidx >= collision_fidx or entry_side is None:
        # 기하 방식이 well-defined 아니면 구 휴리스틱으로 폴백
        entry_bbox = None
        for bf in sorted(b_frames, key=lambda x: x["fidx"]):
            if _bbox_cy(bf["bbox"]) > 0.5 * H:
                entry_fidx, entry_bbox = bf["fidx"], bf["bbox"]
                break
        if entry_fidx is None or entry_bbox is None or entry_fidx >= collision_fidx:
            first_b = sorted(b_frames, key=lambda x: x["fidx"])[0]
            entry_fidx, entry_bbox = first_b["fidx"], first_b["bbox"]
        entry_side = "LEFT" if _bbox_cx(entry_bbox) < 0.5 * W else "RIGHT"

    entry_sec = entry_fidx / DEFAULT_FPS

    # ── evasion_space: 충돌 1초 전 윈도우, 옆차로 점유(최댓값)+장면 혼잡도(중앙값)
    #    (stage2_side.py의 free_space_counts에서 복구한 원본 규칙) ──
    evasion = _evasion_space(frames, collision_fidx, u_vp, v_vp, W, H, entry_side)

    n_frames = frames[-1]["frame_idx"] + 1

    return {
        "collision_frame_idx": collision_fidx,
        "collision_sec":       round(collision_sec, 4),
        "entry_frame_idx":     entry_fidx,
        "entry_sec":           round(entry_sec, 4),
        "entry_side":          entry_side,
        "evasion_space":       evasion,
        "n_frames":            n_frames,
        "width":               W,
        "height":              H,
    }


# ─────────────────────────────────────────────────────────────────────────────
def parse_label_zip(zp: str, split: str) -> tuple[dict, list[str]]:
    """zip 하나 열어서 {clip_id: derive_result} 반환."""
    results = {}
    bad_clips = []

    with zipfile.ZipFile(zp) as z:
        # clip_id → list of json names
        clip_jsons: dict[str, list] = defaultdict(list)
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            parts = name.split("/")
            if len(parts) < 2:
                continue
            clip_jsons[parts[0]].append(name)

        for clip_id, json_names in tqdm(clip_jsons.items(), desc=os.path.basename(zp), leave=False):
            frame_list = []
            W_clip = H_clip = None

            for jname in sorted(json_names):
                try:
                    with z.open(jname) as f:
                        data = json.load(f)
                except Exception:
                    continue

                W = data.get("width", 1920)
                H = data.get("height", 1080)
                W_clip, H_clip = W, H
                fidx = int(data.get("sequence_frame_number", 0))
                objects = data.get("objects", [])

                others = [o["bbox"] for o in objects
                          if o.get("isObjectOthers") and _bbox_area(o["bbox"]) > MIN_AREA_FRAC * W * H]

                frame_list.append({
                    "frame_idx": fidx,
                    "W": W, "H": H,
                    "objects": objects,
                    "others": others,
                })

            if not frame_list:
                bad_clips.append(clip_id)
                continue

            frame_list.sort(key=lambda x: x["frame_idx"])
            res = derive_clip_labels(frame_list)
            if res is None:
                bad_clips.append(clip_id)
                continue

            res["clip_id"] = clip_id
            res["split"]   = split
            results[clip_id] = res

    return results, bad_clips


# ─────────────────────────────────────────────────────────────────────────────
def build_video_map(data_root: str) -> dict[str, str]:
    """clip_id → mp4를 담고 있는 zip 경로 매핑."""
    video_map = {}
    for split_dir in ["1.Training", "2.Validation"]:
        src_dir = os.path.join(data_root, split_dir, "원천데이터_231108_add")
        if not os.path.isdir(src_dir):
            continue
        for zname in sorted(os.listdir(src_dir)):
            if not zname.endswith(".zip") or "영상" not in zname:
                continue
            zp = os.path.join(src_dir, zname)
            try:
                with zipfile.ZipFile(zp) as z:
                    for name in z.namelist():
                        if name.endswith(".mp4"):
                            cid = Path(name).stem
                            video_map[cid] = zp
            except Exception:
                continue
    return video_map


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="C:/Users/kiot_/Downloads/095.교통사고 영상 데이터/01.데이터")
    ap.add_argument("--out_dir",   default="data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for split_dir, split_name in [("1.Training", "train"), ("2.Validation", "val")]:
        label_dir = os.path.join(args.data_root, split_dir, "라벨링데이터_231108_add")
        if not os.path.isdir(label_dir):
            print(f"[skip] {label_dir} 없음")
            continue
        for zname in sorted(os.listdir(label_dir)):
            if not zname.endswith(".zip") or "이미지" not in zname:
                continue
            zp = os.path.join(label_dir, zname)
            res, bad = parse_label_zip(zp, split_name)
            all_results.update(res)
            if bad:
                print(f"  {zname}: 스킵 {len(bad)}개")

    print(f"\n총 derive 완료: {len(all_results)}개 클립")

    # ── labels CSV ────────────────────────────────────────────────────────
    rows = list(all_results.values())
    df = pd.DataFrame(rows, columns=[
        "clip_id", "collision_frame_idx", "collision_sec",
        "entry_frame_idx", "entry_sec", "entry_side", "evasion_space",
        "n_frames", "width", "height", "split",
    ])
    labels_path = os.path.join(args.out_dir, "aihub_labels.csv")
    df.to_csv(labels_path, index=False)
    print(f"라벨 저장: {labels_path}  ({len(df)}행)")
    print(df[["entry_side", "evasion_space"]].value_counts().to_string())

    # ── video map CSV ─────────────────────────────────────────────────────
    print("\n영상 zip 매핑 중...")
    video_map = build_video_map(args.data_root)
    vm_rows = [{"clip_id": k, "zip_path": v} for k, v in video_map.items() if k in all_results]
    vm_df = pd.DataFrame(vm_rows)
    map_path = os.path.join(args.out_dir, "aihub_video_map.csv")
    vm_df.to_csv(map_path, index=False)
    print(f"영상 맵 저장: {map_path}  ({len(vm_df)}개 매칭)")
    print(f"  매칭 실패: {len(all_results) - len(vm_df)}개")


if __name__ == "__main__":
    main()
