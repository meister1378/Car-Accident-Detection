"""
AIHub 597 "교통사고 영상 데이터" Stage1 매니페스트 빌더 + 선택적 추출.

디스크 제약(D: 여유 134GB, 전체 zip 126GB) 때문에 zip 전체를 풀지 않는다.
zipfile 모듈로 각 zip의 목차/JSON만 읽어(압축 해제 없이) 매니페스트를 먼저
만들고, 그 중 학습에 실제로 쓸 영상만 개별적으로 extract한다.

디렉토리 구조 (raw_root 기준):
    1.Training\\원천데이터_231108_add\\TS_<category>_영상_<condition>.zip   (mp4)
    1.Training\\라벨링데이터_231108_add\\TL_<category>_영상_<condition>.zip (json)
    2.Validation\\원천데이터_231108_add\\TS_...zip
    2.Validation\\라벨링데이터_231108_add\\TL_...zip

영상 zip과 라벨 zip은 "TS_" <-> "TL_" 접두어만 다르고 나머지 파일명(카테고리+조건)이
동일하며, 내부 entry는 flat하게 들어있다(하위 폴더 없음). entry stem이 video_name과
동일하다.

filming_way == "bb" (블랙박스)인 것만 사용한다. 파일명이 이미 bb_/cc_ 접두어를
갖고 있지만 신뢰하지 않고 JSON의 filming_way 필드로 다시 검증한다.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# split마다 zip 파일 접두어가 다르다.
# Training: TS_(영상)/TL_(라벨), Validation: VS_(영상)/VL_(라벨)
SPLIT_DIRS = {
    "train": ("1.Training", "TS_", "TL_"),
    "val": ("2.Validation", "VS_", "VL_"),
}

RAW_VIDEO_SUBDIR = "원천데이터_231108_add"
RAW_LABEL_SUBDIR = "라벨링데이터_231108_add"


def discover_zip_pairs(raw_root: Path):
    """
    raw_root 아래에서 (split, category, video_zip, label_zip) 쌍을 찾는다.
    """

    pairs = []

    for split, (split_dir_name, video_prefix, label_prefix) in SPLIT_DIRS.items():
        split_dir = raw_root / split_dir_name

        video_dir = split_dir / RAW_VIDEO_SUBDIR
        label_dir = split_dir / RAW_LABEL_SUBDIR

        if not video_dir.exists() or not label_dir.exists():
            print(f"[WARN] split dir missing, skip: {split_dir}")
            continue

        for video_zip in sorted(video_dir.glob(f"{video_prefix}*.zip")):
            stem = video_zip.stem  # e.g. TS_차대보행자_영상_조건 / VS_...
            suffix = stem[len(video_prefix):]

            label_zip = label_dir / f"{label_prefix}{suffix}.zip"

            if not label_zip.exists():
                print(f"[WARN] matching label zip not found for {video_zip.name}")
                continue

            parts = suffix.split("_")
            category = parts[0] if parts else "unknown"

            pairs.append(
                {
                    "split": split,
                    "category": category,
                    "video_zip": video_zip,
                    "label_zip": label_zip,
                }
            )

    return pairs


def build_full_manifest(zip_pairs) -> pd.DataFrame:
    """
    각 zip 쌍의 목차만 읽어(압축 해제 없이) bb(블랙박스) 영상 매니페스트를 만든다.
    """

    records = []

    total_bb = 0
    total_cctv = 0
    total_unmatched = 0
    total_json_error = 0

    for pair in zip_pairs:
        split = pair["split"]
        category = pair["category"]
        video_zip_path = pair["video_zip"]
        label_zip_path = pair["label_zip"]

        with zipfile.ZipFile(video_zip_path) as zf_video:
            video_infos = {Path(n).stem: zf_video.getinfo(n) for n in zf_video.namelist()}

        with zipfile.ZipFile(label_zip_path) as zf_label:
            label_names = zf_label.namelist()

            for name in label_names:
                stem = Path(name).stem

                video_info = video_infos.get(stem)

                if video_info is None:
                    total_unmatched += 1
                    continue

                try:
                    raw = zf_label.read(name).decode("utf-8-sig")
                    data = json.loads(raw)
                    video_meta = data["video"]
                except Exception as e:
                    total_json_error += 1
                    print(f"[WARN] JSON read failed: {label_zip_path.name}/{name} | {e}")
                    continue

                filming_way = video_meta.get("filming_way")

                if filming_way != "bb":
                    total_cctv += 1
                    continue

                total_bb += 1

                records.append(
                    {
                        "split": split,
                        "category": category,
                        "stem": stem,
                        "video_zip": str(video_zip_path),
                        "video_entry": video_info.filename,
                        "video_size_bytes": video_info.file_size,
                        "filming_way": filming_way,
                        "point_of_view": video_meta.get("video_point_of_view"),
                    }
                )

    df = pd.DataFrame(records)

    print("=== AIHub597 Stage1 Full Manifest (bb only) ===")
    print(f"blackbox(bb): {total_bb}")
    print(f"cctv excluded: {total_cctv}")
    print(f"unmatched(video<->label): {total_unmatched}")
    print(f"json read errors: {total_json_error}")

    if len(df) > 0:
        print()
        print("split x category counts:")
        print(df.groupby(["split", "category"]).size())

    return df


def stratified_sample(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """
    category 비율을 유지하며 최대 limit개까지 재현 가능하게 샘플링한다.
    limit이 None이거나 df가 이미 limit 이하이면 df 전체를 반환한다.
    """

    if limit is None or len(df) <= limit:
        return df.reset_index(drop=True)

    rng = random.Random(seed)

    groups = {cat: sub.index.tolist() for cat, sub in df.groupby("category")}

    for indices in groups.values():
        rng.shuffle(indices)

    total = len(df)

    # category별 목표 개수를 비율대로 배분(반올림)한 뒤, 남는/모자란 만큼
    # 큰 카테고리부터 순서대로 조정한다.
    targets = {}
    for cat, indices in groups.items():
        targets[cat] = round(limit * len(indices) / total)

    # 반올림 오차 보정
    diff = limit - sum(targets.values())
    cats_by_size_desc = sorted(groups, key=lambda c: len(groups[c]), reverse=True)

    i = 0
    while diff != 0 and cats_by_size_desc:
        cat = cats_by_size_desc[i % len(cats_by_size_desc)]
        step = 1 if diff > 0 else -1
        candidate = targets[cat] + step

        if 0 <= candidate <= len(groups[cat]):
            targets[cat] = candidate
            diff -= step

        i += 1

        if i > 10000:
            break

    selected_indices = []

    for cat, indices in groups.items():
        n = min(targets.get(cat, 0), len(indices))
        selected_indices.extend(indices[:n])

    selected_indices = sorted(selected_indices)

    return df.loc[selected_indices].reset_index(drop=True)


def extract_selected(df: pd.DataFrame, extract_dir: Path) -> pd.DataFrame:
    """
    df에 담긴 영상만 개별적으로 extract_dir 아래로 extract한다(zip 전체 압축 해제 안 함).

    extract_dir/<split_dir_name>/<category>/<stem>.mp4 로 저장하고,
    df에 그 상대경로를 담은 'video_path' 컬럼을 추가해 반환한다.
    """

    extract_dir.mkdir(parents=True, exist_ok=True)

    split_to_dirname = {"train": "Training", "val": "Validation"}

    # 같은 zip을 여러 row가 참조하므로 ZipFile을 캐시한다.
    zip_cache: dict[str, zipfile.ZipFile] = {}

    video_paths = []

    extracted = 0
    skipped_existing = 0

    for row in df.itertuples(index=False):
        split_dirname = split_to_dirname[row.split]

        target_dir = extract_dir / split_dirname / row.category
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / f"{row.stem}.mp4"

        # Windows에서 만든 매니페스트를 Linux 컨테이너에서도 그대로 쓸 수 있도록
        # 항상 forward slash로 저장한다(백슬래시는 Linux에서 경로 구분자로 해석되지 않는다).
        rel_path = target_path.relative_to(extract_dir).as_posix()

        if target_path.exists() and target_path.stat().st_size == row.video_size_bytes:
            skipped_existing += 1
            video_paths.append(rel_path)
            continue

        zf = zip_cache.get(row.video_zip)
        if zf is None:
            zf = zipfile.ZipFile(row.video_zip)
            zip_cache[row.video_zip] = zf

        # zf.extract는 내부 archive 경로(=flat 파일명)를 target_dir 아래에 그대로 만든다.
        extracted_path = Path(zf.extract(row.video_entry, path=target_dir))

        if extracted_path != target_path:
            extracted_path.replace(target_path)

        extracted += 1
        video_paths.append(rel_path)

    for zf in zip_cache.values():
        zf.close()

    df = df.copy()
    df["video_path"] = video_paths

    print(f"[extract] newly extracted: {extracted}, already present: {skipped_existing}")

    return df


def main():
    parser = argparse.ArgumentParser(description="AIHub597 Stage1 manifest builder + selective extraction")

    parser.add_argument(
        "--raw_root",
        type=str,
        default=r"D:\workspace_mj\095.교통사고 영상 데이터\01.데이터",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=r"D:\workspace_mj\aihub597_stage1_manifest",
    )
    parser.add_argument(
        "--extract_dir",
        type=str,
        default=r"D:\workspace_mj\aihub597_stage1_extracted",
    )
    parser.add_argument("--train_limit", type=int, default=2000)
    parser.add_argument("--val_limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild_full_manifest",
        action="store_true",
        help="캐시된 full_manifest_bb.csv가 있어도 zip을 다시 스캔한다.",
    )
    parser.add_argument(
        "--no_extract",
        action="store_true",
        help="추출 없이 매니페스트(csv)만 생성한다.",
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    manifest_dir = Path(args.manifest_dir)
    extract_dir = Path(args.extract_dir)

    manifest_dir.mkdir(parents=True, exist_ok=True)

    full_manifest_path = manifest_dir / "full_manifest_bb.csv"

    if full_manifest_path.exists() and not args.rebuild_full_manifest:
        print(f"[cache] loading full manifest from {full_manifest_path}")
        full_df = pd.read_csv(full_manifest_path)
    else:
        zip_pairs = discover_zip_pairs(raw_root)
        print(f"zip pairs found: {len(zip_pairs)}")

        full_df = build_full_manifest(zip_pairs)

        if len(full_df) == 0:
            raise RuntimeError("No blackbox(bb) videos found - check raw_root path.")

        full_df.to_csv(full_manifest_path, index=False)
        print(f"[cache] saved full manifest to {full_manifest_path}")

    train_full = full_df[full_df["split"] == "train"].reset_index(drop=True)
    val_full = full_df[full_df["split"] == "val"].reset_index(drop=True)

    print()
    print(f"train full pool: {len(train_full)}")
    print(f"val full pool:   {len(val_full)}")

    train_sel = stratified_sample(train_full, args.train_limit, args.seed)
    val_sel = stratified_sample(val_full, args.val_limit, args.seed + 1)

    print()
    print(f"train selected: {len(train_sel)}")
    print(train_sel["category"].value_counts())
    print()
    print(f"val selected: {len(val_sel)}")
    print(val_sel["category"].value_counts())

    est_bytes = train_sel["video_size_bytes"].sum() + val_sel["video_size_bytes"].sum()
    print()
    print(f"estimated extracted size: {est_bytes / (1024**3):.2f} GB")

    if args.no_extract:
        train_out = train_sel
        val_out = val_sel
    else:
        train_out = extract_selected(train_sel, extract_dir)
        val_out = extract_selected(val_sel, extract_dir)

    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"

    train_out.to_csv(train_csv, index=False)
    val_out.to_csv(val_csv, index=False)

    print()
    print(f"[done] train manifest -> {train_csv} ({len(train_out)} rows)")
    print(f"[done] val manifest   -> {val_csv} ({len(val_out)} rows)")
    print(f"[done] extract_dir    -> {extract_dir}")


if __name__ == "__main__":
    main()
