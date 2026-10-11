# 재현 매뉴얼

DACON 236753 — 블랙박스 영상 기반 지능형 고의사고 분석 모델

Public Score 복원을 위해 데이터 준비부터 학습·추론까지 실제로 실행한
스크립트와 명령을 Stage 1·2·3 순서로 기록한다. 모든 스크립트는
`01_학습코드`, `02_추론코드` 폴더에 동봉돼 있다. 설계 의도·아키텍처
설명은 `03_보고서/모델_개발_보고서.md`(Stage 1·2) 참고.

## 1) 환경 구성

| 구분 | 도구/라이브러리 | 버전 | 제조사(출처) | 용도 |
|---|---|---|---|---|
| 1 | PyTorch | 2.6.0 (cu124) | Meta AI (오픈소스) | 모델 학습·추론 프레임워크 |
| 2 | timm (PyTorch Image Models) | 1.0.29 | Ross Wightman / Hugging Face (오픈소스) | ConvNeXt 계열 백본 사전학습 가중치 로드 |
| 3 | OpenCV (opencv-python) | 5.0.0 | OpenCV.org (오픈소스) | 영상 디코딩, 광류(optical flow) 계산 |
| 4 | NumPy / Pandas | 최신 안정판 | NumPy/Pandas 커뮤니티 (오픈소스) | 데이터 처리 |
| 5 | CUDA / cuDNN | 12.4 / 9.x | NVIDIA | GPU 가속 연산 |
| 6 | GPU | RTX 4090 | NVIDIA | 학습·추론 하드웨어 |

위 버전대로 라이브러리를 설치하고 GPU 환경(CUDA/cuDNN)을 구성한다.

## 2) 데이터 배치

Stage 1·2는 AI Hub 095(교통사고 영상 데이터), Nexar Collision
Prediction, CCD(Car Crash Dataset) 세 소스를 공식 배포처에서 받아
로컬 경로에 배치한다(출처·라이센스·사용 범위는
`03_보고서/모델_개발_보고서.md` 5-2절 참고).

Stage 3는 이와 별도로 KITTI raw, nuScenes, A2D2, comma2k19, L2D-v3
다섯 개 공개 데이터셋을 사용한다(5절 참고).

## 3) Stage 1 — 매니페스트 생성 및 학습

```bash
# 학습/검증 대상 클립 선별 + 합성 재촬영 변환용 매니페스트 생성
python stage1_aihub_manifest.py \
  --raw_root <AI Hub 095.교통사고 영상 데이터 경로>/01.데이터 \
  --manifest_dir <매니페스트 저장 경로> \
  --extract_dir <선별 클립 추출 경로> \
  --train_limit 4000 --val_limit 2191 --seed 42

# MViT 재촬영 탐지 분류기 학습
python train_stage1_mvit_aihub.py \
  --manifest_dir <위 매니페스트 저장 경로> \
  --extract_dir <위 추출 경로> \
  --save_dir model/stage1 \
  --checkpoint_name mvit_aihub_candidate_full4000.pt \
  --epochs 5 --batch 2 --num_workers 6
```

`--train_limit`/`--val_limit`는 배포 체크포인트의 최종 분할 크기
(train 4,000 / val 2,191)에 맞춘 값이며, 그 외 인자(`--lr`, `--frames`,
`--size`, `--recapture_size`, `--seed` 등)는 스크립트 기본값을 그대로
사용했다.

## 4) Stage 2

### 4-1) 라벨 생성

```bash
python derive_aihub_labels.py \
  --data_root "<AI Hub 095.교통사고 영상 데이터 경로>/01.데이터/1.Training" \
  --out_labels data/aihub_labels.csv \
  --out_map    data/aihub_video_map.csv

python derive_evasion_ssdlite.py \
  --labels_csv data/aihub_labels.csv \
  --video_map  data/aihub_video_map.csv \
  --ssdlite_ckpt <SSDLite320 MobileNetV3 COCO 사전학습 가중치 경로> \
  --out data/aihub_evasion_ssdlite.csv
```

### 4-2) 백본(ImpactNet) 학습

```bash
python train_backbone.py \
  --labels_csv data/aihub_labels.csv \
  --video_map  data/aihub_video_map.csv \
  --backbone_name convnext_tiny.in12k_ft_in1k \
  --epochs 12 --batch 64 --lr 1e-4 --head_lr_mult 10.0 \
  --weight_decay 0.05 --drop_path 0.1 \
  --warmup_steps 500 \
  --w_col 0.55 --w_ent 0.25 --w_side 0.1 --w_evas 0.1 \
  --p_near 0.5 --lag2_p 0.3 --seed 20260828 \
  --frame_cache_dir <프레임 디코딩 캐시 경로, 수백 GB 여유 필요> \
  --out models/stage2/stage2_backbone.pt
```

**`--warmup_steps 500`(500스텝 선형 웜업 + 코사인 감쇠)을 반드시 포함해야
한다.** 이 인자가 없으면 학습 시작 직후(30~60스텝 이내) 백본이 입력과
무관한 상수 출력만 내는 표현 붕괴가 재현된다(모델 개발 보고서 2-3절
상세).

### 4-3) 피처 캐시 구축

```bash
python build_feature_cache.py \
  --backbone_pt models/stage2/stage2_backbone.pt \
  --cache_dir data/stage2_ftF \
  --nexar_dir <Nexar Collision Prediction 경로> \
  --workers 8 --device cuda

python build_feature_cache_ccd.py \
  --backbone_pt models/stage2/stage2_backbone.pt \
  --cache_dir data/stage2_ftF \
  --ccd_zip <CCD(Car Crash Dataset) zip 경로> \
  --device cuda
```

### 4-4) 시계열 헤드(Stage2Net) 학습

```bash
python train_stage2net.py --nexar_eval_fold -1 --cache_dir data/stage2_ftF \
  --backbone_meta models/stage2/stage2_backbone.pt --out models/stage2/best.pt

python train_stage2net.py --nexar_eval_fold 0 --cache_dir data/stage2_ftF \
  --backbone_meta models/stage2/stage2_backbone.pt --out models/stage2/v2_fold0.pt

python train_stage2net.py --nexar_eval_fold 1 --cache_dir data/stage2_ftF \
  --backbone_meta models/stage2/stage2_backbone.pt --out models/stage2/v2_fold1.pt
```

나머지 하이퍼파라미터는 스크립트 기본값을 그대로 사용했다
(in_dim=2382, hidden=256, dilations=[1,2,4,8,16,32,64], epochs=26,
batch=16, lr=3e-4).

## 5) Stage 3

### 5-1) 데이터 수집 및 라벨 유도

Stage 3는 V-JEPA2.1 ViT-B(완전 고정)로 16프레임 클립을 인코딩해
0.1초당 3072차원 특징으로 캐싱하고, 그 위에 가벼운 시계열 헤드만
학습하는 구조다. 데이터는 네 단계로 점차 넓혔다(각 단계가 앞 단계의
체크포인트·홀드아웃 분할을 이어받는다):

| 트랙 | 추가 데이터 | 비고 |
|---|---|---|
| three_heads | KITTI raw(33 drives) | 원본 zip은 사전 확보 가정(수집 스크립트 없음) |
| diverse_v2 | + nuScenes(544세그먼트), A2D2(14파트) | A2D2는 별도 스트리밍 수집 스크립트 사용 |
| diverse_v3 | + comma2k19(333세그먼트) | 정합성 검증(해시, route/date 분리) 포함 |
| l2d500 | + L2D-v3 500에피소드(450train/50val) | 세션 단위로 train/val 분리 |

라벨은 대회 정답이 아니라 센서 기반 물리 임계값(가속도 0.4m/s²,
요레이트 1.5°/s, 정지 히스테리시스 0.3/0.6m/s)으로 약하게 유도했다
(`prepare.py`의 `labels()`).

### 5-2) V-JEPA 특징 추출 및 헤드 학습

트랙별 정확한 실행 순서(전처리 스크립트명·대기 조건 포함)는
`01_학습코드/stage3/README.md`에 전부 기록했다. 요약하면:

```bash
# three_heads: 정렬->라벨 유도->특징 추출->헤드 비교 학습이 한 스크립트에서 진행
python night_experiments.py
python finish_experiments.py

# diverse_v2: A2D2 수집 후 드라이버가 nuScenes 정렬+학습을 자동 진행
python collect_a2d2_stream_v2.py
python diverse_v2_driver.py

# diverse_v3: 정합성 검증+특징 추출+학습이 드라이버 하나로 자동 진행
python diverse_v3_driver.py

# l2d500: 전처리(에피소드 선별, 특징 추출, 반전 특징)를 먼저 끝내야 함
python select_data_v2.py
python prepare_features_cached.py
python flipfeat_l2d.py
python continue_pipeline.py        # 위 산출물 확인 후 학습+1차 패키징 자동 실행
python continue_norm_ab.py
python package_extended.py
```

공통 설정은 `config.json`에 있으나, 실제 배포 체크포인트에 내장된 값
(`epochs=30, learning_rate=0.00015`)은 이 파일 기본값과 달라 최종 학습
시 커맨드라인 인자로 덮어썼을 가능성이 높다.

### 5-3) 앙상블 구성

실제 제출 체크포인트(`head.pt`)는 **28개 스냅샷의 평균 앙상블**이다 —
decoupled_ltc 16개(시드 4개 × 에폭 4개), motion_tcn 8개, vn_tcn 4개.
이 28개 각각을 만든 개별 학습 로그와 앙상블을 실제로 조립한 스크립트는
남아있지 않아, 이 단계는 **기록된 방법론이지 바이트 단위 재현 수단은
아니다**(`01_학습코드/stage3/README.md`의 "알려진 한계" 참고).

## 6) 추론

**Stage 2**: `stage2_backbone.pt`·`v2_fold0.pt`·`v2_fold1.pt`는
`predict.py`가 있는 폴더 자신을 기준으로 찾으므로 반드시 그 폴더 안에
직접 배치해야 하고, `best.pt`만 임의의 `model_dir`에 두면 된다
(`02_추론코드/stage2/README.md` 참고). 배치 후 다음과 같이 호출한다.

```python
from pathlib import Path
import predict
df = predict.predict(Path("<data_dir>/stage2"), Path("<model_dir>"))
# df columns: ID, collision_frame, entry_frame, evasion_space, entry_side
```

**Stage 3**: `encoder.pt`·`head.pt`·`postproc.json`·`vjepa2/`·`vendor/`를
전부 같은 폴더(`model/stage3/`)에 배치한다(`02_추론코드/stage3/README.md`
참고).

```python
from pathlib import Path
import adapter
df = adapter.predict_stage3(Path("<data_dir>/stage3"), Path("<model_dir>"))
# df columns: ID, sample_index, accel_label, steer_label
```

**통합 실행**: `02_추론코드/inference.py`가 Stage 1·2·3을 순서대로 호출해
각각 CSV로 저장하는 최상위 진입점이다.

```bash
python inference.py --data_dir <data_dir> --model_dir <model_dir> --out_dir ./submission
```
