"""Frozen V-JEPA 2.1 encoder and a small temporal classifier; offline loading."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn


def load_encoder(repo, checkpoint, device):
    sys.path.insert(0, str(Path(repo).resolve()))
    from app.vjepa_2_1.models.vision_transformer import vit_base
    encoder = vit_base(img_size=(384, 384), num_frames=64, tubelet_size=2,
                       use_sdpa=True, use_rope=True, uniform_power=False,
                       img_temporal_dim_size=1, interpolate_rope=True)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = ckpt.get('ema_encoder', ckpt)
    state = {k.removeprefix('module.').removeprefix('backbone.'): v for k, v in state.items()}
    encoder.load_state_dict(state, strict=True)
    return encoder.requires_grad_(False).to(device).eval()


def image_tensor(image, resolution):
    # Preserve full road width with letterbox; deliberate downstream preprocessing.
    image = ImageOps.pad(image.convert('RGB'), (resolution, resolution), color=(124,116,104))
    x = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


@torch.inference_mode()
def extract_sequence(load_image, count, encoder, cfg, device):
    if count < 1:
        raise ValueError('Empty recording')
    length, stride, resolution = cfg['clip_frames'], cfg['clip_stride'], cfg['resolution']
    if length % 2 or not 0 < stride <= length:
        raise ValueError('Need even clip length and stride <= length')
    spatial = resolution//16
    total, weight = None, np.zeros(count, dtype=np.float32)
    for start in range(0, count, stride):
        ids = np.minimum(np.arange(start, start+length), count-1)
        clip = torch.stack([image_tensor(load_image(int(i)), resolution) for i in ids], dim=1)[None].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda', dtype=torch.float16):
            tokens = encoder(clip)
        expected = (length//2)*spatial*spatial
        if tokens.ndim != 3 or tokens.shape[1] != expected:
            raise ValueError(f'Unexpected encoder shape {tokens.shape}; expected {expected} tokens')
        grid = tokens.float().reshape(1, length//2, spatial, spatial, -1)
        # Four spatial regions preserve left/right and near/far layout.
        half = spatial//2
        features = torch.cat([grid[:,:,ys:ye,xs:xe].mean((2,3)) for ys,ye,xs,xe in
                              [(0,half,0,half),(0,half,half,spatial),(half,spatial,0,half),(half,spatial,half,spatial)]], -1)[0].cpu().numpy()
        # Each tubelet represents two frames. Interpolation restores the output grid,
        # but does not manufacture independently observed 0.1-second information.
        coordinates = np.arange(length//2)*2+.5
        expanded = np.stack([np.interp(np.arange(length), coordinates, features[:,j]) for j in range(features.shape[1])], 1)
        if total is None:
            total = np.zeros((count, features.shape[1]), dtype=np.float32)
        valid = min(length, count-start)
        total[start:start+valid] += expanded[:valid]
        weight[start:start+valid] += 1
    return total/weight[:,None]


class TemporalHead(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Conv1d(input_dim, hidden, 1)
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d),
                                    nn.GELU(), nn.Dropout(.1)) for d in [1,2,4,8]])
        self.accel = nn.Conv1d(hidden,4,1)
        self.steer = nn.Conv1d(hidden,3,1)

    def forward(self, x):
        x = self.project(self.norm(x).transpose(1,2))
        for layer in self.layers:
            x = x + layer(x)
        return self.accel(x).transpose(1,2), self.steer(x).transpose(1,2)


def macro_f1(truth, pred, classes):
    truth, pred = np.asarray(truth), np.asarray(pred)
    if not len(truth):
        return None
    per_class = []
    for c in range(classes):
        tp = np.sum((truth==c)&(pred==c))
        den = np.sum(truth==c)+np.sum(pred==c)
        per_class.append(float(2*tp/den) if den else 0.0)
    return {'macro_f1': float(np.mean(per_class)), 'per_class_f1': per_class,
            'support': np.bincount(truth,minlength=classes).tolist()}
