"""Actual packaged Stage1/2/3 inference on short synthetic inputs, offline.

This verifies interface/label/range contracts, finite complete rows and video
reset. It is not a score evaluation or a full private-test runtime benchmark.
Dependencies must already be installed before entering the guarded process.
"""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main(args):
    out, fixtures = Path(args.out).resolve(), Path(args.fixtures)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    original = socket.socket.connect
    def deny(self, address):
        if self.family == socket.AF_UNIX:
            return original(self, address)
        raise RuntimeError('Offline smoke blocked Python socket connection')
    socket.socket.connect = deny
    try:
        sys.path.insert(0, os.environ['SUBMISSION_TIMM_DIR'])
        import timm
        import torch
        assert timm.__version__ == '1.0.15'
        sys.path.insert(0, str(out))
        module = importlib.import_module('inference')
        assert Path(module.__file__).resolve() == out/'inference.py'
        cfg = torch.load(out/'model/stage3/head.pt', map_location='cpu', weights_only=True)['config']
        assert cfg['accel_classes'] == ['ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED']
        assert cfg['steer_classes'] == ['LEFT', 'STRAIGHT', 'RIGHT']
        report = {'kind': args.kind, 'checkpoint_sha256': digest(out/'model/stage3/head.pt'),
                  'hf_offline_and_python_socket_connect_guard': True, 'tests': {},
                  'torch': torch.__version__, 'timm': timm.__version__,
                  'limitations': 'Short synthetic structural inputs; not private-test accuracy, total runtime or an OS-level network sandbox.'}
        for stage in [1, 2, 3]:
            started = time.monotonic()
            df = getattr(module, f'predict_stage{stage}')(fixtures/f's{stage}', out/f'model/stage{stage}')
            assert not df.isna().any().any()
            if stage == 1:
                assert df.columns.tolist() == ['ID', 'answer'] and len(df) == 1
                assert df.ID.astype(str).tolist() == ['000001']
                assert set(df.answer) <= {'ORIGINAL', 'RERECORDED'}
            elif stage == 2:
                assert df.columns.tolist() == ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']
                assert len(df) == 1 and df.ID.astype(str).tolist() == ['000001']
                for key in ['collision_frame', 'entry_frame']:
                    value = df[key].iloc[0]
                    assert float(value).is_integer() and 0 <= int(value) < 64
                assert df.evasion_space.iloc[0] in [0, 1]
                assert df.entry_side.iloc[0] in ['LEFT', 'RIGHT']
            else:
                assert df.columns.tolist() == ['ID', 'sample_index', 'accel_label', 'steer_label']
                assert set(df.accel_label) <= set(cfg['accel_classes'])
                assert set(df.steer_label) <= set(cfg['steer_classes'])
                assert set(df.ID) == {'000001', '000002'}
                for video, length in [('000001', 64), ('000002', 17)]:
                    assert df[df.ID == video].sample_index.tolist() == list(range(length))
                alone = module.predict_stage3(fixtures/'s3_alone', out/'model/stage3')
                assert alone.reset_index(drop=True).equals(df[df.ID == '000002'].reset_index(drop=True))
            report['tests'][str(stage)] = {'passed': True, 'rows': len(df),
                'columns': df.columns.tolist(), 'seconds': time.monotonic()-started}
            if stage == 3:
                report['tests']['3']['per_video_reset_passed'] = True
            print('OFFLINE_STAGE_PASS', stage, report['tests'][str(stage)], flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        Path(args.report).write_text(json.dumps(report, indent=2))
    finally:
        socket.socket.connect = original


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['out', 'fixtures', 'report', 'kind']:
        p.add_argument('--'+key, required=True)
    main(p.parse_args())
