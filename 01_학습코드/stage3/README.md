# Stage 3 학습 코드 (V-JEPA2.1 ViT-B 기반)

DACON 236753 — 0.1초 단위 가속(4종)·조향(3종) 분류.

실제 제출(`submit_stage123_v10A_ltc_tcn_vntcn_20260928`)에서 쓰인 Stage 3는
이 폴더의 코드로 이어진 V-JEPA2.1 ViT-B 백본 + LTC/TCN/VN-TCN 앙상블
헤드다. 추론에 바로 쓰는 완성된 코드·가중치는
`../../02_추론코드/stage3/`에 있고, 이 폴더는 그 가중치를 만들어낸
**데이터 수집·학습 과정**을 담고 있다.

## 구조 — 4개 실험 트랙

작업은 시간 순으로 데이터를 넓혀가며 진행됐다. 각 트랙은 독립 폴더이며,
뒤 트랙이 앞 트랙의 산출물(체크포인트·홀드아웃 분할)을 이어받는 구조다.

| 트랙 | 추가된 데이터 | 헤드 | 폴더 |
|---|---|---|---|
| three_heads | KITTI raw (33 drives) | TCN / FACT / LTContext 비교 | `diverse_v3/night_experiments.py`, `finish_experiments.py` |
| diverse_v2 | + nuScenes(544세그먼트), A2D2(14파트) | balanced_tcn / decoupled_ltc / motion_tcn + 균등가중 앙상블 | `diverse_v3/diverse_v2_driver.py` 계열 |
| diverse_v3 | + comma2k19(333세그먼트) | ltc_scratch / ltc_finetune(v2 LTC에서 이어 학습) / 앙상블 | `diverse_v3/diverse_v3_driver.py` |
| l2d500 | + L2D-v3 500에피소드(450train/50val) | wide LTC(model_dim 256) 웜스타트 → neck 변형 → 정규화 A/B | `l2d500/continue_pipeline.py` → `continue_norm_ab.py` → `package_extended.py` |

`flip_features/`는 comma·L2D 검증·테스트 클립의 좌우 반전 특징을 따로
추출해 flip-TTA 평가(`ensemble_eval`)에 사용한다.

## 핵심 아키텍처 (모든 트랙 공통)

- **백본은 완전히 고정(freeze)** — V-JEPA2.1 ViT-B(`vjepa2_1_vitb_dist_vitG_384.pt`)를
  파인튜닝 없이 특징 추출기로만 사용한다. LoRA 어댑터는 속도·메모리
  측정용으로만 실험했고(`l2d500/lora_*probe.py`), 실제 학습에는 쓰지 않았다.
- 16프레임 클립(stride 8)을 384px로 레터박스 패딩해 인코딩하고, 토큰을
  좌상/우상/좌하/우하 4분면으로 평균 풀링해 **0.1초 프레임당 3072차원**
  특징으로 캐싱한다(`.npz`).
- 그 위에 가벼운 시계열 헤드(LTContext 또는 dilated TCN)만 학습해 가속
  4종·조향 3종을 프레임마다 예측한다. 라벨은 대회 정답이 아니라 센서
  기반 물리 임계값(가속도 0.4m/s², 요레이트 1.5°/s, 정지 히스테리시스
  0.3/0.6m/s)으로 약하게 유도했다.
- 실제 제출 체크포인트(`../../02_추론코드/stage3/head.pt`)는 **28개
  스냅샷 앙상블**이다 — decoupled_ltc 16개(시드 4개 × 에폭 4개),
  motion_tcn 8개, vn_tcn 4개를 평균한다.

## 재현 순서 (전처리 → 학습)

각 트랙은 **①데이터 수집/정렬 → ②라벨 유도 → ③V-JEPA 특징 추출·캐싱 →
④시계열 헤드 학습** 순서로 진행된다. 트랙별로 어디까지가 별도 스크립트로
남아있고 어디부터 드라이버가 자동으로 이어받는지 아래에 명시한다.

### three_heads (KITTI)

1. **데이터 수집 — 스크립트 없음.** KITTI raw zip이 이미
   `stage3_data/kitti_zips`, `stage3_data/kitti_extra50gb_20260916`에
   받아져 있다고 가정한다(`night_experiments.py` 79~80행). 이 다운로드
   자체를 수행한 스크립트는 이번 코드에 포함돼 있지 않다 — 수작업 또는
   별도 노트북 셀로 받아둔 것으로 보인다.
2. **라벨 유도 + 특징 추출 + 학습**이 한 스크립트 안에서 순서대로
   실행된다(`labels()`는 `prepare.py`에서 가져와 씀):
   ```bash
   python diverse_v3/night_experiments.py   # KITTI 정렬 -> 라벨 유도 -> 특징 추출 -> TCN/FACT/LTContext 비교 학습
   python diverse_v3/finish_experiments.py  # 결과 정리
   ```

### diverse_v2 (+ nuScenes, A2D2)

1. **A2D2 수집**: `python diverse_v3/collect_a2d2_stream_v2.py` —
   스트리밍 다운로드 + 해시 검증, 완료되면 `a2d2_collection_report.json`
   기록.
2. **nuScenes 정렬**은 별도 실행이 필요 없다 — 3단계 드라이버 안에서
   `diverse_v2_data.prepare_nuscenes()`가 자동으로 수행한다.
3. **학습**: `python diverse_v3/diverse_v2_driver.py` — 1번이 끝났다는
   신호(`*_collection_report.json`)를 최대 12시간 대기한 뒤, nuScenes
   정렬 → `diverse_v2_train.py` → `diverse_v2_finish.py`를 자동으로
   순서대로 실행한다.

### diverse_v3 (+ comma2k19)

데이터 정합성 검증(해시, route/date 분리)과 특징 추출, 학습이 드라이버
하나로 묶여 있다:
```bash
python diverse_v3/diverse_v3_driver.py
# 내부적으로 diverse_v3_data.py(정합성 검증+추출) -> diverse_v3_train.py -> diverse_v3_finish.py 순서로 자동 실행
```

### l2d500 (+ L2D-v3 500에피소드)

이 트랙은 전처리 단계를 **반드시 먼저 따로 실행**해야 한다 — 학습
드라이버가 전처리 완료 신호 파일을 기다리기만 하고 직접 실행하지는
않는다.

```bash
# 1) 에피소드 선별 (450 train / 50 val, 세션 단위로 분리)
python l2d500/select_data_v2.py

# 2) V-JEPA 특징 추출·캐싱 (원본 뷰) — features_complete.json이 생성될 때까지 실행
python l2d500/prepare_features_cached.py

# 3) 좌우 반전 특징 (flip-TTA용)
python flip_features/flipfeat_l2d.py

# 4) 위 산출물이 확인되면 학습 + 1차 패키징을 자동 실행
python l2d500/continue_pipeline.py

# 5) 단계 간 확률 정규화 방식(joint9 vs taskwise) A/B 비교 후 최종 패키징
python l2d500/continue_norm_ab.py
python l2d500/package_extended.py
```

`flip_features/flipfeat.py`(comma 검증/테스트용)는 `ensemble_eval` 단계의
flip-TTA 평가에 쓰인다.

### 공통 설정

`diverse_v3/config.json`에 공통 하이퍼파라미터가 있다(단, 실제 배포
체크포인트(`head.pt`)에 내장된 설정은 `epochs=30, learning_rate=0.00015`로
이 파일의 기본값(`epochs=20, learning_rate=0.001`)과 다르다 — 최종 학습
시 커맨드라인 인자로 덮어썼을 가능성이 높다).

## 공유 런타임 의존성

`vjepa_runtime.py`(V-JEPA2 인코더 로딩), `vendor/LTContext`,
`vendor/FACT`(시계열 헤드 원본 구현), `vjepa2/`(V-JEPA2 공식 저장소)는
학습·추론 양쪽에서 공유하는 코드라 중복 보관하지 않고
`../../02_추론코드/stage3/`에 함께 두었다. 이 폴더의 스크립트를 실행하려면
해당 경로를 `sys.path`에 추가하거나 같은 위치에 복사해야 한다.

## 알려진 한계 — 완전한 재현은 불가능

이 폴더만으로는 `head.pt`를 처음부터 다시 만들 수 없다. 실제로 누락된
부분:

- **`train_three_experiments.py`, `train_neck.py`** — l2d500 트랙이
  호출하는 `make_wide`, `batched_loss`, `metric`, 웜스타트 헤드 정의가
  이 스크립트들 안에 있었는데, Colab 노트북 셀로만 존재했을 가능성이
  높아 별도 파일로 남아있지 않다.
- **28개 앙상블 멤버 각각의 개별 체크포인트**와 **`ens3` 앙상블을 실제로
  조립한 스크립트** — `head.pt`에는 최종 평균 결과만 들어있고, 그 전
  단계(에폭별 스냅샷, 시드별 개별 학습 로그)는 남아있지 않다.
- 전부 Colab 환경(`/content/MyDrive/...` 하드코딩 경로) 기준으로 작성돼
  있어, 로컬 재현 시 경로 수정이 필요하다.

즉 이 폴더는 **데이터 수집·전처리·학습 방법론의 실제 기록**이며, 바이트
단위로 동일한 `head.pt`를 재생성하는 용도로는 쓸 수 없다.

## 데이터 누수 주의

comma2k19 세션 `b0c9d2329ad1606b|2018-07-27`의 세그먼트 3·4·5·10·11은
DACON 공식 baseline(OPEN_001~005)과 동일한 클립이다. 이 트랙들의 학습
분할에도 해당 세그먼트가 포함돼 있어, 로컬에서 baseline 영상으로 검증한
수치는 참고용일 뿐 실제 일반화 성능과는 다를 수 있다.
