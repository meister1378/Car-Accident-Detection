# Stage 2 추론 코드 (선택 제공 — 리더보드 제출 zip으로 대체 가능)

`predict.py`/`stage2_*.py`/`s2*.py`는 `01_학습코드/stage2`로 새로 학습한 가중치를 연결해 실측
검증했다.

## 가중치 배치 시 주의 (실제로 겪은 함정)

`predict(data_dir, model_dir)`에 넘긴 `model_dir`은 `best.pt`(주 모델)를 찾는
데만 쓰인다. **`stage2_backbone.pt`와 `v2_fold0.pt`/`v2_fold1.pt`(앙상블 파트너)는
`model_dir`을 무시하고 이 폴더(`stage2_model.py`가 있는 위치) 자신을 기준으로
찾는다** (`stage2_model.py`의 `_ft_backbone_file()`/`_ensemble_nets()` 참고 —
원래 제출 zip이 코드와 가중치를 한 폴더에 같이 두는 구조라 이렇게 짜여 있음).

즉 실제로 쓰려면:

```
02_추론코드/stage2/
├── predict.py, stage2_*.py, s2*.py   (코드)
├── stage2_backbone.pt                 (여기에 직접!)
├── v2_fold0.pt                        (여기에 직접!)
└── v2_fold1.pt                        (여기에 직접!)
```

`best.pt`만 `model_dir`(별도 폴더여도 무방)에 두면 된다. 이걸 모르면
"체크포인트는 있는데 백본을 못 찾았다"며 조용히 OpenCV 휴리스틱 폴백으로
떨어지는데, 에러 없이 실행은 되기 때문에 매우 발견하기 어렵다(실제로 이번
프로젝트에서 겪음).

## 호출

```python
from pathlib import Path
import predict
df = predict.predict(Path("<data_dir>/stage2"), Path("<model_dir>"))
# df columns: ID, collision_frame, entry_frame, evasion_space, entry_side
```

## 실측 검증 결과

DACON 공식 baseline 5개(000001~000005, `t_collision` 실측 라벨 존재)로
확인한 결과, 예측 충돌 프레임이 실제 사고 시점과 근접한 구간에 안정적으로
위치함을 확인했다(자세한 시각화는 `../../03_보고서/모델_개발_보고서.md`
4장 참고).
