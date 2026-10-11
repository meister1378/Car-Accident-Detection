"""comma 검증/test 95개 녹화의 좌우반전(flip) V-JEPA 특징 생성 -> comma 에서도 TTA 포함 평가 가능하게.
원본 zip(Chunk_1/3) 에서 video.hevc 꺼내 record['images'] 프레임 번호만 디코드 -> 384 pad(JPEG q92, 학습 때와 동일) -> 원본/반전 특징.
첫 녹화에서 캐시된 v0 특징과 상관 확인. 결과: ensemble_eval_20260927/comma_ho_flipfeats.pkl {id: (f0, f1)} (float16)"""
from pathlib import Path
import sys, json, io, time, pickle, zipfile, shutil, importlib.util
import numpy as np, torch, cv2
from PIL import Image, ImageOps

ROOT = Path('/content/MyDrive/MyDrive'); RUNS = ROOT / 'stage3_vjepa_runs'; E = RUNS / 'ensemble_eval_20260927'
S3 = Path('/content/ens3code/model/stage3'); sys.path.insert(0, str(S3))
from vjepa_runtime import load_encoder, extract_sequence
spec = importlib.util.spec_from_file_location('prior_train', RUNS / 'v3_w256_B_15ep_20260926/train_three_experiments.py')
prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
old = json.loads((RUNS / 'diverse_v3_20260917/manifest.json').read_text())
HO = [r for r in old['records'] if r['domain'] == 'comma2k19' and r['split'] in ('val', 'test')]
dev = torch.device('cuda')
cfg = torch.load(S3 / 'head.pt', map_location='cpu', weights_only=False)['config']
cfg = {k: (int(v) if k in ('clip_frames', 'clip_stride', 'resolution') else v) for k, v in cfg.items()}
enc = load_encoder(S3 / 'vjepa2', S3 / 'encoder.pt', dev)
OUTP = E / 'comma_ho_flipfeats.pkl'
RES = pickle.loads(OUTP.read_bytes()) if OUTP.exists() else {}
ZIPS = {c: zipfile.ZipFile(ROOT / f'stage3_data/diverse_v3_20260917/comma2k19_Chunk_{c}.zip') for c in (1, 3)}
NAMES = {c: set(z.namelist()) for c, z in ZIPS.items()}
TMP = Path('/content/flip_tmp'); TMP.mkdir(exist_ok=True)
t0 = time.time()
for k, r in enumerate(HO):
    if r['id'] in RES: continue
    pre = r['source']; c = 1 if pre + '/video.hevc' in NAMES[1] else 3
    idx = [int(Path(p).stem) for p in r['images']]; want = set(idx)
    vp = TMP / 'video.hevc'
    with ZIPS[c].open(pre + '/video.hevc') as f, vp.open('wb') as out: shutil.copyfileobj(f, out, 1 << 24)
    cap = cv2.VideoCapture(str(vp)); n = 0; frames = {}
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if n in want:
            im = ImageOps.pad(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), (384, 384), color=(124, 116, 104))
            b = io.BytesIO(); im.save(b, format='JPEG', quality=92); frames[n] = b.getvalue()
        n += 1
    cap.release()
    assert all(i in frames for i in idx), (r['id'], n, len(frames))

    def load0(i): return Image.open(io.BytesIO(frames[idx[i]])).convert('RGB')
    def load1(i): return ImageOps.mirror(Image.open(io.BytesIO(frames[idx[i]])).convert('RGB'))
    f0 = extract_sequence(load0, len(idx), enc, cfg, dev); f1 = extract_sequence(load1, len(idx), enc, cfg, dev)
    f0 = np.asarray(f0.cpu() if torch.is_tensor(f0) else f0, np.float32); f1 = np.asarray(f1.cpu() if torch.is_tensor(f1) else f1, np.float32)
    if k < 2 or len(RES) < 2:
        with np.load(prior.cache(r, 0)) as z: c0 = z['features'].astype(np.float32)
        print('CHECK', r['id'], f0.shape, c0.shape, 'corr', round(float(np.corrcoef(f0.ravel(), c0.ravel())[0, 1]), 4),
              'rel_err', round(float(np.abs(f0 - c0).mean() / np.abs(c0).mean()), 4), flush=True)
    RES[r['id']] = (f0.astype(np.float16), f1.astype(np.float16))
    if len(RES) % 5 == 0 or len(RES) == len(HO): OUTP.write_bytes(pickle.dumps(RES))
    print('FLIP', len(RES), '/', len(HO), r['id'], round(time.time() - t0), 's', flush=True)
OUTP.write_bytes(pickle.dumps(RES)); print('FLIP_DONE', len(RES), flush=True)
