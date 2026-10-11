"""
Stage1 opt-in 후보: torchvision MViTv2-S(Kinetics-400 pretrained) 기반
ORIGINAL/RERECORDED 2-class 영상 분류 모델.

기존 배포된 Stage1 파이프라인(model/stage1/best.pt, stage1_sequence_features.py의
rule_based_nudge)과는 완전히 분리된 별도 opt-in 후보다. 이 파일은
train_stage1.py / inference.py 어디에서도 import되지 않는다.

16프레임 클립을 통째로 하나의 3D-conv/attention 네트워크에 넣어 native
temporal modeling을 수행한다 - 프레임별 독립 분류가 아니다. 이 점이
stage1_recapture_transform.py의 "실제 timestamp로 위상을 구동하는" augmentation과
짝을 이루는 핵심 설계다.
"""

from __future__ import annotations

from torch import nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


class Stage1MViT(nn.Module):
    """
    forward(x): x = [B, C, T, H, W] -> logits [B, 2]
        label 0 = ORIGINAL, label 1 = RERECORDED
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        self.net = mvit_v2_s(weights=weights)

        # Kinetics-400(400 class) head -> 2-class(ORIGINAL/RERECORDED) head로 교체
        self.net.head[1] = nn.Linear(self.net.head[1].in_features, 2)

    def forward(self, x):
        return self.net(x)
