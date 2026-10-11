"""Offline video-only inference for mixed-dataset decoupled models."""
from pathlib import Path
import tempfile
import cv2
import pandas as pd
from PIL import Image,ImageOps
import torch
from vjepa_runtime import load_encoder,extract_sequence
from diverse_v2_models import load_members,predict

def predict_stage3(data_dir,model_dir):
    model_dir=Path(model_dir);saved=torch.load(model_dir/'head.pt',map_location='cpu',weights_only=False)
    try:_flip_tta=bool(__import__('json').loads((model_dir/'postproc.json').read_text()).get('flip_tta',False))
    except Exception:_flip_tta=False
    cfg=saved['config'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder=load_encoder(model_dir/'vjepa2',model_dir/'encoder.pt',device);members=load_members(saved,device)
    videos=sorted(p for p in (Path(data_dir)/'videos').rglob('*') if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv','.m4v','.hevc'})
    assert videos and len({p.stem for p in videos})==len(videos)
    rows=[]
    for video in videos:
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
        assert 9.5<=fps<=10.5,(video.name,fps)
        with tempfile.TemporaryDirectory(prefix='v2_s3_') as tmp:
            files=[]
            try:
                while True:
                    ok,bgr=cap.read()
                    if not ok:break
                    p=Path(tmp)/f'{len(files):06d}.png'
                    im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                    ImageOps.pad(im,(384,384),color=(124,116,104)).save(p);files.append(p);_fsize=im.size
            finally:cap.release()
            assert files,'Empty video'
            def load(i):
                with Image.open(files[i]) as im:return im.convert('RGB')
            features=extract_sequence(load,len(files),encoder,cfg,device)
            features_flip=None
            if _flip_tta:
                def load_flip(i):
                    with Image.open(files[i]) as im:return ImageOps.mirror(im.convert('RGB'))
                features_flip=extract_sequence(load_flip,len(files),encoder,cfg,device)
            aa,ss=predict(members,features,device,features_flip=features_flip,frame_size=_fsize)
            for i,(a,s) in enumerate(zip(aa.tolist(),ss.tolist())):
                rows.append({'ID':video.stem,'sample_index':i,'accel_label':cfg['accel_classes'][a],'steer_label':cfg['steer_classes'][s]})
    return pd.DataFrame(rows,columns=['ID','sample_index','accel_label','steer_label'])
