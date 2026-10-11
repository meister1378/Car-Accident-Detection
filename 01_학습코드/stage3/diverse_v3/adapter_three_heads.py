"""Standalone Stage3 adapter. Integrate after GPU/runtime validation; not a full submission."""
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from vjepa_runtime import load_encoder, extract_sequence
from experiment_heads import make_head, predict_sequence, decode_labels


def predict_stage3(data_dir, model_dir):
    model_dir=Path(model_dir)
    saved=torch.load(model_dir/'head.pt',map_location='cpu',weights_only=True)
    cfg=saved['config'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder=load_encoder(model_dir/'vjepa2',model_dir/'encoder.pt',device)
    assert saved.get('joint_labels') is True, 'Expected trained joint-label checkpoint'
    head=make_head(saved['kind'], saved['input_dim']).to(device).eval()
    head.load_state_dict(saved['model'],strict=True)
    rows=[]
    extensions={'.mp4','.avi','.mov','.mkv','.m4v','.hevc'}
    videos=sorted(p for p in (Path(data_dir)/'videos').rglob('*') if p.suffix.lower() in extensions)
    if not videos: raise ValueError('No evaluation videos')
    if len({p.stem for p in videos})!=len(videos): raise ValueError('Duplicate video IDs')
    with torch.inference_mode():
        for video in videos:
            cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
            if not 9.5<=fps<=10.5:
                cap.release()
                raise ValueError(f'{video.name}: official inference requires 10Hz video, got {fps}. Resample public examples separately.')
            # Temporary disk-backed RGB buffer bounds RAM; never write into evaluation data_dir.
            with tempfile.TemporaryDirectory(prefix='s3_') as tmp:
                files=[]
                try:
                    while True:
                        ok,bgr=cap.read()
                        if not ok: break
                        dest=Path(tmp)/f'{len(files):06d}.png'
                        # Lossless resize cache; same letterbox applied in training and inference.
                        from PIL import ImageOps
                        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                        im=ImageOps.pad(im,(cfg['resolution'],cfg['resolution']),color=(124,116,104))
                        im.save(dest);files.append(dest)
                finally: cap.release()
                def load(i):
                    with Image.open(files[i]) as im: return im.convert('RGB')
                if not files: raise ValueError(f'Empty video: {video.name}')
                features=extract_sequence(load,len(files),encoder,cfg,device)
                if not files: raise ValueError(f'Empty video: {video.name}')
                pred=predict_sequence(head,saved['kind'],features,device,core=256,context=64)
                aa,ss=decode_labels(pred)
                for i,(ac,st) in enumerate(zip(aa.tolist(),ss.tolist())):
                    rows.append({'ID':video.stem,'sample_index':i,
                                 'accel_label':cfg['accel_classes'][ac],'steer_label':cfg['steer_classes'][st]})
    return pd.DataFrame(rows,columns=['ID','sample_index','accel_label','steer_label'])
