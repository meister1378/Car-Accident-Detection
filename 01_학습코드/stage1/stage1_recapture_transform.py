"""
Stage1 opt-in 후보(MViT + AIHub597) 전용 synthetic recapture augmentation.

배경
----
기존 이 프로젝트의 stage1_sequence_features.py는 "프레임별 독립 아키텍처로는
rolling-band 패턴을 구조적으로 잡을 수 없다"고 진단했다. 이 진단의 핵심은
디스플레이 재촬영 아티팩트(모아레/에일리어싱, 플리커, 롤링 밴드)가 전부
"실제 프레임 타임스탬프"를 축으로 위상(phase)이 이어지는 시간적 신호라는
점이다 - 프레임을 독립적으로 골라서 개별 처리하면 이 위상 연속성 자체가
사라진다.

여기 구현하는 RecaptureTransform은 그 반대를 한다: 하나의 클립(T개 프레임)
전체에 대해, 프레임 index가 아니라 "영상에서 실제로 그 프레임이 sampled된
시각(초)"을 축으로 삼아 flicker/aliasing/rolling-band의 위상을 계산한다.
그 결과를 16프레임을 통째로 보는 MViT(Stage1MViT, native temporal modeling)에
넣어야만 이 신호가 의미를 가진다 - 프레임 단위 분류기에는 애초에 이 augmentation의
이점이 없다.

포함하는 artifact (재촬영에 비교적 직접적으로 연관된 것만):
    - display_aliasing : RGB 서브픽셀 격자 + 공간 불규칙성 + 시간 위상 드리프트
    - perspective       : reflect-pad 후 warp+center-crop (검은 삼각형 방지)
    - camera_jitter     : 프레임간 random-walk 평행이동/회전
    - display_flicker   : 실제 timestamp로 구동되는 주파수+배음 sine 밝기 변조
    - rolling_band      : 공간+시간 위상이 함께 변하는 밝기 밴드

포함하지 않는 것(=ORIGINAL/RERECORDED 양쪽에 공통으로 적용해야 하는 것):
    일반적인 blur, 일반적인 resize 열화, static gamma 변화.
    -> CommonVideoTransform에서 두 class에 동일 분포로 독립 적용한다.

Input/Output: float Tensor [T, C, H, W], range [0, 1].
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def _smooth_random_field(h: int, w: int, rng: random.Random, grid: int, device, dtype) -> torch.Tensor:
    """
    작은 grid x grid 랜덤 필드를 bicubic으로 h x w로 upsample한 [-1, 1] 범위의
    부드러운 spatial perturbation. display lattice가 화면 전체에서 완전히
    똑같은 수학적 격자가 되는 것을 막는 데 쓴다.
    """

    gen = torch.Generator(device=device)
    gen.manual_seed(rng.randrange(0, 2**31 - 1))

    field = torch.rand((1, 1, grid, grid), generator=gen, device=device, dtype=dtype)
    field = field * 2.0 - 1.0

    field = F.interpolate(field, size=(h, w), mode="bicubic", align_corners=False)

    return field[0, 0]


def _severity_range(severity: str, weak, medium, strong) -> tuple[float, float]:
    return {"weak": weak, "medium": medium, "strong": strong}[severity]


class CommonVideoTransform:
    """
    ORIGINAL/RERECORDED 양쪽 모두에 나타날 수 있는 일반적인 화질 변화
    (gamma, blur, resample). 두 class에 독립적으로 같은 확률로 적용해
    "화질이 나쁘면 RERECORDED"라는 shortcut을 학습하지 못하게 한다.
    """

    def __call__(self, clip: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        rng = random.Random(seed)
        clip = clip.clone()

        if rng.random() < 0.35:
            gamma = rng.uniform(0.92, 1.08)
            clip = clip.clamp(1e-6, 1.0).pow(gamma)

        if rng.random() < 0.25:
            sigma = rng.uniform(0.20, 0.70)
            clip = TF.gaussian_blur(clip, kernel_size=[3, 3], sigma=[sigma, sigma])

        if rng.random() < 0.25:
            _, _, h, w = clip.shape
            scale = rng.uniform(0.82, 0.98)
            sh, sw = max(32, int(h * scale)), max(32, int(w * scale))
            clip = TF.resize(clip, [sh, sw], interpolation=InterpolationMode.BILINEAR, antialias=True)
            clip = TF.resize(clip, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)

        return clip.clamp(0.0, 1.0)


class RecaptureTransform:
    """
    profile="train" | "holdout" (holdout=validation용, train과 확률/range를
    약간 다르게 해서 val score가 학습 분포에 그대로 과적합되어 보이는 것을 줄인다.
    실제 물리적 재촬영 검증을 대체하지는 않는다).

    __call__(clip, seed, timestamps) -> clip
        clip:       [T, C, H, W] float [0,1]
        timestamps: [T] 초 단위 실제 sampled frame 시각. None이면 [0,1] 정규화 시간으로 대체.
    """

    NEAR_CLEAN_TRANSFORMS = ("display_aliasing", "display_flicker", "camera_jitter")

    def __init__(self, profile: str = "train"):
        if profile not in ("train", "holdout"):
            raise ValueError(f"unsupported profile: {profile}")

        self.profile = profile
        # 일부 RERECORDED는 아티팩트를 아주 약하게 하나만 넣어 "아티팩트=RERECORDED"
        # shortcut을 줄인다.
        self.near_clean_prob = 0.15

        if profile == "train":
            self.artifact_prob = {
                "display_aliasing": 0.35,
                "perspective": 0.30,
                "display_flicker": 0.50,
                "rolling_band": 0.30,
                "camera_jitter": 0.40,
            }
            self.severity_weights = [0.35, 0.50, 0.15]  # weak, medium, strong
        else:
            self.artifact_prob = {
                "display_aliasing": 0.25,
                "perspective": 0.20,
                "display_flicker": 0.50,
                "rolling_band": 0.35,
                "camera_jitter": 0.30,
            }
            self.severity_weights = [0.60, 0.35, 0.05]

    # ------------------------------------------------------------------
    def __call__(
        self,
        clip: torch.Tensor,
        seed: int | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clip.ndim != 4 or clip.shape[1] != 3:
            raise ValueError("clip must have shape [T, 3, H, W]")

        rng = random.Random(seed)
        clip = clip.clone()
        times = self._resolve_timestamps(clip, timestamps)

        if rng.random() < self.near_clean_prob:
            name = rng.choice(self.NEAR_CLEAN_TRANSFORMS)
            return self._dispatch(name, clip, rng, "weak", times).clamp(0.0, 1.0)

        severity = rng.choices(["weak", "medium", "strong"], weights=self.severity_weights, k=1)[0]
        applied = []

        # display-side effect 먼저
        if rng.random() < self.artifact_prob["display_aliasing"]:
            clip = self._display_aliasing(clip, rng, severity, times)
            applied.append("display_aliasing")

        # camera geometry
        if rng.random() < self.artifact_prob["perspective"]:
            clip = self._perspective(clip, rng, severity)
            applied.append("perspective")

        if rng.random() < self.artifact_prob["camera_jitter"]:
            clip = self._camera_jitter(clip, rng, severity)
            applied.append("camera_jitter")

        # display-camera 시간적 상호작용
        if rng.random() < self.artifact_prob["display_flicker"]:
            clip = self._display_flicker(clip, rng, severity, times)
            applied.append("display_flicker")

        if rng.random() < self.artifact_prob["rolling_band"]:
            clip = self._rolling_band(clip, rng, severity, times)
            applied.append("rolling_band")

        if not applied:
            # 아무 것도 우연히 선택되지 않았다면 ORIGINAL과 완전히 동일한
            # positive를 만들지 않도록 약한 artifact 하나를 강제로 넣는다.
            name = rng.choice(self.NEAR_CLEAN_TRANSFORMS)
            clip = self._dispatch(name, clip, rng, "weak", times)

        return clip.clamp(0.0, 1.0)

    def _dispatch(self, name, clip, rng, severity, timestamps):
        if name == "display_aliasing":
            return self._display_aliasing(clip, rng, severity, timestamps)
        if name == "perspective":
            return self._perspective(clip, rng, severity)
        if name == "display_flicker":
            return self._display_flicker(clip, rng, severity, timestamps)
        if name == "rolling_band":
            return self._rolling_band(clip, rng, severity, timestamps)
        if name == "camera_jitter":
            return self._camera_jitter(clip, rng, severity)
        raise ValueError(f"unknown transform: {name}")

    @staticmethod
    def _resolve_timestamps(clip: torch.Tensor, timestamps: torch.Tensor | None) -> torch.Tensor:
        t = clip.shape[0]

        if timestamps is None:
            return torch.linspace(0.0, 1.0, steps=t, device=clip.device, dtype=clip.dtype)

        ts = torch.as_tensor(timestamps, device=clip.device, dtype=clip.dtype).reshape(-1)

        if ts.numel() != t:
            raise ValueError("timestamps length must match clip length")

        # 절대 시작 시각은 의미가 없으므로 첫 sampled frame을 t=0으로 맞춘다.
        return ts - ts[0]

    # ------------------------------------------------------------------
    # display aliasing : RGB 서브픽셀 격자 + 시간 위상 드리프트
    # ------------------------------------------------------------------
    def _display_aliasing(self, clip, rng, severity, timestamps):
        t, _, h, w = clip.shape
        device, dtype = clip.device, clip.dtype

        strength = rng.uniform(*_severity_range(severity, (0.003, 0.010), (0.007, 0.018), (0.012, 0.030)))

        if self.profile == "train":
            pitch_x = rng.uniform(2.2, 5.0)
            pitch_y = rng.uniform(2.4, 5.5)
        else:
            pitch_x = rng.uniform(1.9, 5.8)
            pitch_y = rng.uniform(2.0, 6.2)

        angle = rng.uniform(-math.radians(8.0), math.radians(8.0))
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        yy = torch.arange(h, device=device, dtype=dtype)
        xx = torch.arange(w, device=device, dtype=dtype)
        y, x = torch.meshgrid(yy, xx, indexing="ij")

        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        x0, y0 = x - cx, y - cy

        # display 격자를 카메라 시야각에 맞춰 회전시킨 좌표계
        u = cos_a * x0 + sin_a * y0
        v = -sin_a * x0 + cos_a * y0

        warp_u = _smooth_random_field(h, w, rng, rng.randint(4, 8), device, dtype)
        warp_v = _smooth_random_field(h, w, rng, rng.randint(4, 8), device, dtype)
        warp_amount = rng.uniform(0.05, 0.30)

        u = u + warp_amount * warp_u
        v = v + warp_amount * warp_v

        layouts = [[0.0, 1 / 3, 2 / 3], [2 / 3, 1 / 3, 0.0], [0.0, 0.5, 0.25]]
        channel_offsets = rng.choice(layouts)

        channel_scales = torch.tensor(
            [rng.uniform(0.85, 1.15) for _ in range(3)], device=device, dtype=dtype
        ).reshape(3, 1, 1)

        # display-camera 상대 움직임을 lattice 위상의 느린 drift로 근사한다.
        # timestamps[i]가 클수록(=실제로 더 나중 프레임일수록) drift가 더 진행된다.
        phase_velocity = rng.uniform(-0.25, 0.25)

        frames = []
        for i in range(t):
            temporal_shift = phase_velocity * timestamps[i]

            channel_patterns = []
            for c in range(3):
                offset = channel_offsets[c] + temporal_shift

                cell_x = torch.remainder(u / pitch_x - offset, 1.0)
                dist_x = torch.minimum(cell_x, 1.0 - cell_x)
                aperture_x = torch.exp(-((dist_x / 0.22) ** 2))

                cell_y = torch.remainder(v / pitch_y, 1.0)
                dist_y = torch.minimum(cell_y, 1.0 - cell_y)
                aperture_y = torch.exp(-((dist_y / 0.30) ** 2))

                pattern = 0.80 * aperture_x + 0.20 * aperture_y
                pattern = pattern - pattern.mean()
                channel_patterns.append(pattern)

            modulation = torch.stack(channel_patterns, dim=0) * channel_scales

            frame = clip[i]
            luminance = frame.mean(dim=0, keepdim=True)
            local_strength = 0.35 + 0.65 * luminance

            frame = frame * (1.0 + strength * modulation * local_strength)
            frames.append(frame)

        return torch.stack(frames, dim=0)

    # ------------------------------------------------------------------
    # perspective : 화면-카메라 광축 미정렬. reflect-pad로 검은 삼각형 방지
    # ------------------------------------------------------------------
    def _perspective(self, clip, rng, severity):
        _, _, h, w = clip.shape

        amount = rng.uniform(*_severity_range(severity, (0.002, 0.008), (0.005, 0.018), (0.010, 0.030)))
        pad = max(8, int(max(h, w) * 0.10))

        padded = F.pad(clip, (pad, pad, pad, pad), mode="reflect")
        _, _, hp, wp = padded.shape

        dx = max(1, int(wp * amount))
        dy = max(1, int(hp * amount))

        startpoints = [[0, 0], [wp - 1, 0], [wp - 1, hp - 1], [0, hp - 1]]
        endpoints = [
            [rng.randint(0, dx), rng.randint(0, dy)],
            [wp - 1 - rng.randint(0, dx), rng.randint(0, dy)],
            [wp - 1 - rng.randint(0, dx), hp - 1 - rng.randint(0, dy)],
            [rng.randint(0, dx), hp - 1 - rng.randint(0, dy)],
        ]

        warped = TF.perspective(
            padded, startpoints=startpoints, endpoints=endpoints,
            interpolation=InterpolationMode.BILINEAR, fill=0,
        )

        return TF.center_crop(warped, [h, w])

    # ------------------------------------------------------------------
    # camera jitter : 프레임간 random-walk translate/rotate
    # ------------------------------------------------------------------
    def _camera_jitter(self, clip, rng, severity):
        _, _, h, w = clip.shape

        trans_frac = rng.uniform(*_severity_range(severity, (0.001, 0.003), (0.002, 0.005), (0.004, 0.008)))
        max_angle = rng.uniform(*_severity_range(severity, (0.02, 0.10), (0.05, 0.25), (0.15, 0.50)))

        max_dx = max(1, int(w * trans_frac))
        max_dy = max(1, int(h * trans_frac))
        pad = max(6, int(max(h, w) * 0.05))

        dx = dy = 0
        angle = 0.0
        frames = []

        for frame in clip:
            dx = max(-max_dx, min(max_dx, dx + rng.choice([-1, 0, 1])))
            dy = max(-max_dy, min(max_dy, dy + rng.choice([-1, 0, 1])))
            angle += rng.uniform(-max_angle * 0.25, max_angle * 0.25)
            angle = max(-max_angle, min(max_angle, angle))

            padded = F.pad(frame, (pad, pad, pad, pad), mode="reflect")
            transformed = TF.affine(
                padded, angle=angle, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR, fill=0,
            )
            frames.append(TF.center_crop(transformed, [h, w]))

        return torch.stack(frames, dim=0)

    # ------------------------------------------------------------------
    # display flicker : 실제 timestamp로 구동되는 주파수+배음 밝기 변조
    # ------------------------------------------------------------------
    def _display_flicker(self, clip, rng, severity, timestamps):
        amplitude = rng.uniform(*_severity_range(severity, (0.003, 0.012), (0.008, 0.028), (0.018, 0.045)))

        beat_freq = rng.uniform(0.4, 5.0) if self.profile == "train" else rng.uniform(0.2, 7.0)
        phase = rng.uniform(0.0, 2 * math.pi)

        # 완벽한 단일 sine signature를 약화시키기 위한 작은 배음
        harmonic_strength = rng.uniform(0.05, 0.25)
        harmonic_phase = rng.uniform(0.0, 2 * math.pi)

        primary = torch.sin(2.0 * math.pi * beat_freq * timestamps + phase)
        secondary = torch.sin(2.0 * math.pi * (beat_freq * 2.0) * timestamps + harmonic_phase)

        brightness = 1.0 + amplitude * (primary + harmonic_strength * secondary)
        brightness = brightness.reshape(-1, 1, 1, 1)

        return clip * brightness

    # ------------------------------------------------------------------
    # rolling band : 공간 방향 + 시간 위상이 함께 변하는 밝기 밴드
    # ------------------------------------------------------------------
    def _rolling_band(self, clip, rng, severity, timestamps):
        t, _, h, w = clip.shape
        device, dtype = clip.device, clip.dtype

        amplitude = rng.uniform(*_severity_range(severity, (0.003, 0.010), (0.007, 0.022), (0.015, 0.035)))

        yy = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        y, x = torch.meshgrid(yy, xx, indexing="ij")

        slope = rng.uniform(-0.20, 0.20)  # 완전 수평 stripe로 고정하지 않는다.
        direction = y + slope * x

        spatial_freq = rng.uniform(0.6, 2.2)
        temporal_rate = rng.uniform(0.4, 5.0)
        phase = rng.uniform(0.0, 2 * math.pi)

        envelope = 0.70 + 0.30 * _smooth_random_field(h, w, rng, rng.randint(3, 6), device, dtype)

        frames = []
        for i in range(t):
            temporal_phase = 2.0 * math.pi * temporal_rate * timestamps[i]
            band = torch.sin(2.0 * math.pi * spatial_freq * direction + temporal_phase + phase)
            band = band * envelope
            modulation = 1.0 + amplitude * band
            frames.append(clip[i] * modulation.unsqueeze(0))

        return torch.stack(frames, dim=0)
