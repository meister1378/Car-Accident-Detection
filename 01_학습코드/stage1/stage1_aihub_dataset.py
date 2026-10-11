"""
Stage1 opt-in 후보(MViT + AIHub597) 학습용 Dataset.

stage1_aihub_manifest.py가 만든 train.csv / val.csv(컬럼: video_path, category,
split, point_of_view, ...)와, 그 video_path가 가리키는 개별 추출된 mp4 파일을
읽어서 (ORIGINAL, RERECORDED) 쌍을 만든다.

핵심 원칙(ORIGINAL과 RERECORDED가 동일한 spatial pipeline에서 출발):

    decoded source
          |
          v
    shared intermediate resize (recapture_size, 예: 320)
          |
      base clip
       /      \\
      /        \\
ORIGINAL   RERECORDED
                |
         RecaptureTransform (stage1_recapture_transform.py)
      \\        /
       \\      /
          |
   동일한 resize -> size(224)
          |
   CommonVideoTransform (blur/gamma/resample, 두 class 동일 분포로 독립 적용)
          |
      normalize

이렇게 해야 "resize 경로 자체가 label을 누설"하는 shortcut을 막을 수 있다.

Dataset의 index 하나 = source video 하나. __getitem__은
    clips:  [2, C, T, H, W]   (0=ORIGINAL, 1=RERECORDED)
    labels: [2]               (0, 1)
를 반환한다. DataLoader에서 flatten(0,1)하면 실제 MViT batch가 된다.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from stage1_recapture_transform import CommonVideoTransform, RecaptureTransform


def _frame_to_tensor(frame_rgb_uint8) -> torch.Tensor:
    t = torch.from_numpy(frame_rgb_uint8).permute(2, 0, 1).contiguous()
    return t.float() / 255.0


class AIHubStage1MViTDataset(Dataset):
    def __init__(
        self,
        root,
        manifest,
        frames: int = 16,
        size: int = 224,
        recapture_size: int = 320,
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        train: bool = True,
    ):
        self.root = Path(root)
        self.manifest = Path(manifest)

        self.frames = frames
        self.size = size
        # RecaptureTransform은 size보다 큰 intermediate 해상도에서 적용하고
        # 마지막에 downsample한다(aliasing이 실제로는 다운샘플링 과정에서 두드러지므로).
        self.recapture_size = max(recapture_size, size)
        self.train = train

        self.df = pd.read_csv(self.manifest)

        if len(self.df) == 0:
            raise RuntimeError(f"empty manifest: {self.manifest}")

        if "video_path" not in self.df.columns:
            raise RuntimeError("manifest must contain a 'video_path' column")

        self.mean = torch.as_tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

        profile = "train" if train else "holdout"
        self.recapture = RecaptureTransform(profile=profile)
        self.common = CommonVideoTransform()

    def __len__(self):
        # index 하나 = source video 하나 (ORIGINAL/RERECORDED 쌍을 만든다)
        return len(self.df)

    def _load_shared_clip(self, video_path: Path):
        """
        source video에서 self.frames개를 균등 샘플링해 recapture_size로
        resize한 shared base clip을 만든다. ORIGINAL/RERECORDED가 동일한
        decode/resize 경로를 공유한다.

        Returns: (clip [T,C,rs,rs], timestamps [T] 또는 None)
        """

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise RuntimeError(f"invalid frame count: {video_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 1e-6:
                fps = None

            indices = torch.linspace(0, total - 1, steps=self.frames).round().long().tolist()

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()

                if not ok or frame is None:
                    # 드물게 특정 index를 못 읽으면 직전 프레임을 재사용한다
                    # (source video가 손상됐다고 해서 학습 전체가 죽지 않도록).
                    if frames:
                        frames.append(frames[-1].clone())
                        continue
                    raise RuntimeError(f"could not decode frame {idx} from {video_path}")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                    frame, (self.recapture_size, self.recapture_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                frames.append(_frame_to_tensor(frame))

            base_clip = torch.stack(frames, dim=0)

            timestamps = None
            if fps is not None:
                timestamps = torch.tensor([i / fps for i in indices], dtype=torch.float32)

            return base_clip, timestamps
        finally:
            cap.release()

    def _resize_to_model(self, clip: torch.Tensor) -> torch.Tensor:
        return TF.resize(
            clip, [self.size, self.size],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )

    def _normalize(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self.mean) / self.std

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # 매니페스트가 Windows에서 만들어졌을 수도 있으므로 구분자를 정규화한다.
        rel_path = str(row["video_path"]).replace("\\", "/")
        video_path = self.root / rel_path
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        base_clip, timestamps = self._load_shared_clip(video_path)

        original = base_clip.clone()
        rerecorded = base_clip.clone()

        if self.train:
            recapture_seed = random.getrandbits(63)
        else:
            # validation은 source마다 항상 같은 synthetic condition을 재현해
            # deterministic evaluation이 되게 한다.
            recapture_seed = int(index) + 10_000_000

        rerecorded = self.recapture(rerecorded, seed=recapture_seed, timestamps=timestamps)

        original = self._resize_to_model(original)
        rerecorded = self._resize_to_model(rerecorded)

        if self.train:
            original = self.common(original, seed=random.getrandbits(63))
            rerecorded = self.common(rerecorded, seed=random.getrandbits(63))

        original = self._normalize(original)
        rerecorded = self._normalize(rerecorded)

        # [T,C,H,W] -> [C,T,H,W] (MViT 입력 형식)
        original = original.permute(1, 0, 2, 3).contiguous()
        rerecorded = rerecorded.permute(1, 0, 2, 3).contiguous()

        clips = torch.stack([original, rerecorded], dim=0)
        labels = torch.tensor([0, 1], dtype=torch.long)

        return clips, labels
