# Stage 1 학습 코드 (재촬영 탐지, ConvNeXt V2-Nano 기반)

DACON 236753 — 원본(ORIGINAL) / 재촬영(RERECORDED) 이진 분류.

## 핵심 아키텍처

- 영상에서 8프레임을 균등 샘플링 → 640×518로 리사이즈 → left/center/right
  224×224 3-crop.
- ConvNeXt V2-Nano(사전학습)로 크롭별 특징을 뽑고, 8프레임×3크롭 전체에
  대한 시간축 평균+표준편차를 결합해 이진 분류기에 전달.
- **합성 재촬영**: 실제 재촬영 장비로 찍은 데이터가 없어, 원본 클립에
  모아레·스캔라인·베젤·재생 UI 오버레이·JPEG 재압축 등을 합성으로
  적용해 재촬영 클래스를 만든다. 이 변환은 **전처리 단계에서 미리
  생성해두는 게 아니라, 학습 중 데이터 로더가 매 샘플마다 즉석에서
  랜덤 시드로 적용**한다(`stage1_aihub_dataset.py` → `stage1_recapture_transform.py`).
  그래서 "재촬영 영상 생성" 자체를 위한 별도 전처리 스크립트나 캐시
  파일은 없다 — 전처리 단계는 아래 1단계(매니페스트 생성/클립 선별)뿐이다.

## 재현 순서 (전처리 → 학습)

### 1단계 — 전처리: 매니페스트 생성 및 클립 선별

AI Hub 095 원본(zip 또는 압축 해제본)에서 학습/검증에 쓸 클립을 고르고,
해당 클립만 별도 폴더로 추출한다.

```bash
python stage1_aihub_manifest.py \
  --raw_root <AI Hub 095.교통사고 영상 데이터 경로>/01.데이터 \
  --manifest_dir <매니페스트 저장 경로> \
  --extract_dir <선별 클립 추출 경로> \
  --train_limit 4000 --val_limit 2191 --seed 42
```

- `--train_limit`/`--val_limit`는 배포 체크포인트의 최종 분할 크기
  (train 4,000 / val 2,191)에 맞춘 값이다. 스크립트 기본값은
  2,000/400이며, 4,000/2,191로 실행한 정확한 커맨드 로그는 남아있지
  않다 — 재현 시 이 값으로 지정하면 최종 분할 크기는 맞출 수 있지만,
  어떤 클립이 선택될지는 `--seed`와 원본 디렉터리 스캔 순서에 따라
  달라질 수 있다.
- `--rebuild_full_manifest`: 캐시된 전체 매니페스트가 있어도 zip을
  다시 스캔. `--no_extract`: 이미 추출된 클립이 있으면 재추출 생략.

### 2단계 — 학습

매니페스트를 불러와 합성 재촬영 변환(즉석 적용)과 함께 MViT 계열
분류기를 학습한다.

```bash
python train_stage1_mvit_aihub.py \
  --manifest_dir <1단계 매니페스트 저장 경로> \
  --extract_dir <1단계 추출 경로> \
  --save_dir model/stage1 \
  --checkpoint_name mvit_aihub_candidate_full4000.pt \
  --epochs 5 --batch 2 --num_workers 6
```

나머지 인자(`--lr 1e-4`, `--frames 16`, `--size 224`,
`--recapture_size 320`, `--seed 42`, `--amp`)는 스크립트 기본값을
그대로 사용했다.

## 산출물

| 파일 | 역할 |
|---|---|
| `mvit_aihub_candidate_full4000.pt` | Stage 1 분류기 체크포인트 |

추론 시 이 파일을 `model/stage1/`에 배치. 자세한 배치·호출 방법은
`../../02_추론코드/README.md` 및 `모델_개발_보고서.md` "0. 재현
매뉴얼" 참고.

## 데이터 의존성

- AI Hub 095. 교통사고 영상 데이터(01.데이터) — mp4 원본(zip 또는
  압축 해제본, 바이트 단위로 동일함을 MD5로 확인됨).
- ConvNeXt V2-Nano ImageNet 사전학습 가중치 — 최초 1회 timm이 자동
  다운로드하거나 `~/.cache/huggingface`에 미리 캐시해둬야 함(오프라인
  환경이면 인터넷 접속 필요).

## 알려진 한계

- train 4,000 / val 2,191 분할을 만든 정확한 실행 커맨드(또는 중간에
  파라미터를 바꿔 재실행했는지)가 로그로 남아있지 않다. 체크포인트
  파일명(`_full4000`)도 스크립트 기본 출력명(`mvit_aihub_candidate.pt`)과
  달라, 수작업으로 이름을 바꿨거나 별도 실행 스크립트가 있었을
  가능성이 있다.
- 합성 재촬영 변환은 실제 재촬영 장비·환경을 완전히 재현하지 못할 수
  있다(모델 개발 보고서 2장 참고).
