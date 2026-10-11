"""L2D 검증 50개 클립(1920x1080, 10fps) -> 원본/반전 V-JEPA 특징. 제출 어댑터와 같은 전처리(PNG pad 384). 캐시 v0 와 상관 확인.
결과: ensemble_eval_20260927/l2d_val_flipfeats.pkl {id: (f0, f1)} float16"""
from pathlib import Path
import sys, json, time, pickle
import numpy as np, torch, cv2
from PIL import Image, ImageOps

ROOT = Path('/content/MyDrive/MyDrive'); RUNS = ROOT / 'stage3_vjepa_runs'; E = RUNS / 'ensemble_eval_20260927'; L2 = RUNS / 'l2d500_w256_b48_20260927'
S3 = Path('/content/ens3code/model/stage3')
if not S3.exists():
    import zipfile
    S3 = Path('/content/t5/model/stage3')
sys.path.insert(0, str(S3))
from vjepa_runtime import load_encoder, extract_sequence
new = json.loads((L2 / 'selected_episodes.json').read_text())
V = [r for r in new['records'] if r['split'] == 'val']
dev = torch.device('cuda')
cfg = torch.load(S3 / 'head.pt', map_location='cpu', weights_only=False)['config']
enc = load_encoder(S3 / 'vjepa2', S3 / 'encoder.pt', dev)
OUTP = E / 'l2d_val_flipfeats.pkl'; RES = pickle.loads(OUTP.read_bytes()) if OUTP.exists() else {}
t0 = time.time()
for k, r in enumerate(V):
    if r['id'] in RES: continue
    cap = cv2.VideoCapture(str(L2 / 'clips' / f"{r['id']}.mp4")); frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(ImageOps.pad(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), (384, 384), color=(124, 116, 104)))
    cap.release()
    n = len(frames)
    f0 = extract_sequence(lambda i: frames[i], n, enc, cfg, dev); f1 = extract_sequence(lambda i: ImageOps.mirror(frames[i]), n, enc, cfg, dev)
    f0 = np.asarray(f0.cpu() if torch.is_tensor(f0) else f0, np.float32); f1 = np.asarray(f1.cpu() if torch.is_tensor(f1) else f1, np.float32)
    if len(RES) < 2:
        with np.load(L2 / 'features' / f"{r['id']}_v0.npz") as z: c0 = z['features'].astype(np.float32)
        m = min(len(c0), len(f0))
        print('CHECK', r['id'], f0.shape, c0.shape, 'labels', len(r['accel']), 'corr', round(float(np.corrcoef(f0[:m].ravel(), c0[:m].ravel())[0, 1]), 4), flush=True)
    RES[r['id']] = (f0.astype(np.float16), f1.astype(np.float16))
    if len(RES) % 10 == 0: OUTP.write_bytes(pickle.dumps(RES))
    print('FLIP', len(RES), '/', len(V), r['id'], n, round(time.time() - t0), 's', flush=True)
OUTP.write_bytes(pickle.dumps(RES)); print('FLIP_DONE', len(RES), flush=True)
