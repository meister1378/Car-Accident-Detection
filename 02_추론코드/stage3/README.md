# Stage 3 추론 코드 (V-JEPA2.1 ViT-B 기반, 실제 제출 버전)

DACON 236753 — 0.1초 단위 가속(ACCELERATING/DECELERATING/CONSTANT/STOPPED)·
조향(LEFT/STRAIGHT/RIGHT) 분류. 실제 제출 파일명
`submit_stage123_v10A_ltc_tcn_vntcn_20260928`에서 그대로 가져온 가중치·
코드다. 학습 과정(데이터 수집·실험 이력)은
`../../01_학습코드/stage3/README.md` 참고.

## 호출 방법

최상위 `../inference.py`의 `predict_stage3(data_dir, model_dir)`가
`model_dir/adapter.py`를 동적 로드해 `adapter.predict_stage3()`를
호출하는 구조다. 직접 호출하려면:

```python
from pathlib import Path
import adapter
df = adapter.predict_stage3(Path("<data_dir>/stage3"), Path("<model_dir>"))
# df columns: ID, sample_index, accel_label, steer_label
```

`data_dir/videos/*.mp4`에서 영상을 읽고, `model_dir` 안의 `encoder.pt`·
`head.pt`·`postproc.json`·`vjepa2/`·`vendor/`를 전부 같은 폴더에서 찾으므로
**이 폴더 구조를 그대로 유지해야 한다.**

## 구조

| 파일/폴더 | 역할 |
|---|---|
| `encoder.pt` (347MB) | V-JEPA2.1 ViT-B 가중치(공식 `vjepa2_1_vitb_dist_vitG_384.pt` 기반, 완전 고정) |
| `head.pt` (537MB) | 28개 스냅샷 앙상블(decoupled_ltc 16 + motion_tcn 8 + vn_tcn 4) |
| `postproc.json` | 후처리 설정(아래 참고) |
| `adapter.py` | 영상 → 프레임 추출 → 특징 추출 → 앙상블 예측 → 결과표, 전체 흐름의 진입점 |
| `vjepa_runtime.py` | 인코더 로딩, 클립 단위 특징 추출(4분면 평균 풀링), 시간축 보간 |
| `diverse_v2_models.py` | 헤드 정의(LTC/TCN/VN-TCN), 앙상블 평균·후처리 적용 |
| `extended_ltc_runtime.py`, `ltc_task_normalization.py` | LTContext 헤드의 단계 간(stage-wise) 확률 정규화 모드 전환 |
| `experiment_heads.py`, `extra_heads.py` | 추가 헤드 종류(bigru/tfm/mstcn 등) 정의 — 이번 배포 앙상블엔 미포함, 코드 공유 목적으로 함께 존재 |
| `vendor/LTContext/`, `vendor/FACT/` | 시계열 헤드 원본 구현(각각 공식 GitHub 레포, 라이센스는 각 폴더 내 `LICENSE` 참고) |
| `vjepa2/` | V-JEPA2 공식 저장소(Meta, Apache-2.0) — 백본 아키텍처 정의 |

## 처리 흐름

1. 영상을 10fps로 가정하고(`9.5~10.5fps` 범위 벗어나면 에러) 프레임을
   384×384로 레터박스 패딩.
2. 16프레임 클립(stride 8)을 V-JEPA2.1 ViT-B에 통과시켜 토큰을 얻고,
   화면을 좌상/우상/좌하/우하 4분면으로 나눠 평균 풀링 → 분면당 768차원
   ×4 = 3072차원. 클립 간 겹치는 구간은 평균, tubelet(프레임 2장당 1개
   출력)은 선형보간으로 프레임 단위로 복원.
3. 28개 앙상블 멤버 각각에 넣어 가속 4종·조향 3종 확률을 얻고, 멤버별
   가중치(`wprof`, 기본은 과제별 동일 가중치)로 평균.
4. **후처리**(`postproc.json`, 실제 배포값 `win_accel=5, win_steer=1,
   bias_left=-0.6, bias_right=-0.3, flip_tta=true`):
   - 가속 확률에 5프레임 이동평균(STOPPED으로 이미 확정된 프레임은 제외)
   - 조향 확률에 로그도메인 클래스 바이어스 보정(LEFT/RIGHT 쪽을 둘 다
     낮춰 STRAIGHT 과다예측을 줄이는 방향)
   - **좌우 반전 TTA**: 영상을 좌우로 뒤집어 같은 과정을 한 번 더
     돌리고, 조향 확률은 LEFT/RIGHT을 바꿔서 평균
5. 프레임별 argmax로 최종 라벨 산출, `ID, sample_index, accel_label,
   steer_label` 형식으로 반환.

## 가중치 배치 시 주의

`adapter.py`가 `model_dir` 자신을 기준으로 `encoder.pt`, `head.pt`,
`postproc.json`, `vjepa2/`, `vendor/`를 전부 찾기 때문에, 이 폴더
전체(≈850MB)를 쪼개지 말고 한 덩어리로 `model/stage3/`에 배치해야 한다.

## 참고 — 외부 코드 출처

| 구분 | 출처 | 라이센스 |
|---|---|---|
| V-JEPA2(백본) | Meta AI, GitHub `facebookresearch/vjepa2` | Apache-2.0(`vjepa2/LICENSE`) |
| LTContext(시계열 헤드) | GitHub 공개 구현(`vendor/LTContext`) | 해당 폴더 `LICENSE` 표기 기준 |
| FACT(시계열 헤드, 비교용) | GitHub 공개 구현(`vendor/FACT`) | 해당 폴더 `LICENSE` 표기 기준 |

가중치(`encoder.pt`의 V-JEPA2 사전학습 부분)는 Meta 공식 배포본을
그대로 사용했고, `head.pt`(앙상블)는 본 팀이 처음부터 직접 학습했다.
