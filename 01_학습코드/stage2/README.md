# Stage 2 학습 코드 (ImpactNet 기반, 처음부터 재현)

DACON 236753 — 충돌시점·차선 진입시점/방향·회피공간 예측.

## 재현 순서

### Step 0. 라벨 생성 (이미 완료된 산출물이 있다면 생략)

```bash
python derive_aihub_labels.py \
  --data_root "<AI Hub 095.교통사고 영상 데이터 경로>/01.데이터/1.Training" \
  --out_labels data/aihub_labels.csv \
  --out_map    data/aihub_video_map.csv

python derive_evasion_ssdlite.py \
  --labels_csv data/aihub_labels.csv \
  --video_map  data/aihub_video_map.csv \
  --ssdlite_ckpt models/stage2_reference/ssdlite320_mobilenet_v3_large_coco-a79551df.pth \
  --out data/aihub_evasion_ssdlite.csv
```

### Step 1. ImpactNet 백본 학습 — **가장 중요한 하이퍼파라미터: LR 웜업**

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
  --frame_cache_dir <캐시 저장 경로, 수백 GB 여유 필요> \
  --out models/stage2/stage2_backbone.pt
```

**주의**: `--warmup_steps 500`(500스텝 선형 웜업 + 코사인 감쇠)이 없으면 학습 시작
직후(30~60스텝 이내) 백본이 입력과 무관한 상수 출력만 내는 표현 붕괴
(representation collapse)가 발생한다. 이 프로젝트에서 가장 오래 걸린 디버깅
포인트였다 — 25개 이상의 다른 가설(데이터 구성, 옵티마이저, torch/timm 버전,
라벨 재생성 시점 등)을 전부 배제한 뒤에야 발견됨. 웜업 하나만 켜면 12에폭 내내
완전히 정상 학습된다(loss 2.18→0.21, 붕괴 없음).

`--backbone_name`은 `convnext_tiny.in12k_ft_in1k`(plain, ImageNet 지도학습) 사용을
권장. `convnext_tiny.dinov3_lvd1689m`(DINOv3 self-supervised) 변형도 아키텍처는
동일하게 동작하지만 이번 프로젝트에서 실측 baseline 벤치마크까지는 검증하지 않았음.

### Step 2. Feature 캐시 구축 (AI Hub + Nexar + CCD)

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

`--workers`는 CPU 디코딩(ThreadPoolExecutor)을 병렬화한다 — 순차 실행 대비 약
4.5~4.7배 빠름(실측). 청크 단위로 future를 제출/소비하도록 구현돼 있는데, 이건
전체를 한 번에 `as_completed`에 넘기면 이미 처리된 결과까지 메모리에 계속
쌓여서(실측 84GB까지 증가) 시스템이 위험해지는 걸 겪고 고친 것 — 반드시 이
방식을 유지할 것.

### Step 3. Stage2Net 학습 (전체 + 2-fold 검증)

```bash
python train_stage2net.py --nexar_eval_fold -1 --cache_dir data/stage2_ftF \
  --backbone_meta models/stage2/stage2_backbone.pt --out models/stage2/best.pt

python train_stage2net.py --nexar_eval_fold 0 --cache_dir data/stage2_ftF \
  --backbone_meta models/stage2/stage2_backbone.pt --out models/stage2/v2_fold0.pt

python train_stage2net.py --nexar_eval_fold 1 --cache_dir data/stage2_ftF \
  --backbone_meta models/stage2/stage2_backbone.pt --out models/stage2/v2_fold1.pt
```

하이퍼파라미터는 스크립트 기본값 그대로(in_dim=2382, hidden=256,
dilations=[1,2,4,8,16,32,64], epochs=26, batch=16, lr=3e-4).

## 산출물

| 파일 | 역할 |
|---|---|
| `stage2_backbone.pt` | ImpactNet 백본(6채널 팽창 ConvNeXt-tiny) |
| `best.pt` | Stage2Net 주 모델(전체 학습) |
| `v2_fold0.pt`, `v2_fold1.pt` | 3-net 앙상블 파트너 |

추론 시 위 4개 파일을 `model/stage2/`에 함께 배치. 자세한 배치 규칙은
`../../02_추론코드/README.md` 참고.

## 데이터 의존성

- AI Hub 095. 교통사고 영상 데이터(01.데이터/1.Training) — mp4 원본
- Nexar Collision Prediction (Hugging Face 공개 데이터셋)
- CCD(Car Crash Dataset) — 클립당 50장 JPG 시퀀스, zip 형태
- 전부 오프라인/로컬 파일 기준. 인터넷 접속 없이 재현 가능(단, ConvNeXt
  ImageNet 사전학습 가중치는 최초 1회 timm이 자동 다운로드하거나, 미리
  `~/.cache/huggingface`에 캐시해둬야 함).
