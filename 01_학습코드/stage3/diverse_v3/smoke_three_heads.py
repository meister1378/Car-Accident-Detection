"""Real offline inference checks for one staged complete submission."""
import argparse, importlib, json, os, shutil, socket, sys, time
from pathlib import Path
import cv2
import torch
from package_experiments import digest


def main(out, fixtures, report_path, kind):
    out,fixtures=Path(out),Path(fixtures)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,str(out))
    module=importlib.import_module('inference')
    report={'kind':kind,'checkpoint_sha256':digest(out/'model/stage3/head.pt'),
            'offline_network_blocked':True,'tests':{},'torch':torch.__version__,
            'limitations':'Short structural smoke, not full private-test runtime or accuracy'}
    original=socket.socket.connect
    def deny(self,address):
        if self.family==socket.AF_UNIX:return original(self,address)
        raise RuntimeError('Offline smoke blocked network connection')
    socket.socket.connect=deny
    try:
        for stage in [1,2,3]:
            started=time.monotonic()
            df=getattr(module,f'predict_stage{stage}')(fixtures/f's{stage}',out/f'model/stage{stage}')
            assert not df.isna().any().any()
            test={'passed':True,'seconds':time.monotonic()-started,'rows':len(df),'columns':df.columns.tolist()}
            if stage==1:
                assert df.columns.tolist()==['ID','answer'] and len(df)==1
                assert set(df.answer)<= {'ORIGINAL','RERECORDED'}
            elif stage==2:
                assert df.columns.tolist()==['ID','collision_frame','entry_frame','evasion_space','entry_side'] and len(df)==1
                assert 0<=int(df.collision_frame.iloc[0])<64 and 0<=int(df.entry_frame.iloc[0])<64
                assert int(df.evasion_space.iloc[0]) in [0,1] and df.entry_side.iloc[0] in ['LEFT','RIGHT']
            else:
                assert df.columns.tolist()==['ID','sample_index','accel_label','steer_label']
                assert set(df.accel_label)<=set(module.ACCEL) and set(df.steer_label)<=set(module.STEER)
                assert set(df.ID)=={'000001','000002'}
                for video,n in [('000001',64),('000002',17)]:
                    assert df[df.ID==video].sample_index.tolist()==list(range(n))
                # Second video must be identical when processed alone.
                alone=module.predict_stage3(fixtures/'s3_alone',out/'model/stage3')
                assert alone.reset_index(drop=True).equals(df[df.ID=='000002'].reset_index(drop=True))
                test.update(per_video_reset_passed=True,schema_and_frame_counts_passed=True)
            report['tests'][str(stage)]=test
            print('OFFLINE_STAGE_PASS',kind,stage,test,flush=True)
    finally:socket.socket.connect=original
    Path(report_path).write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--fixtures');p.add_argument('--report');p.add_argument('--kind')
    a=p.parse_args();main(a.out,a.fixtures,a.report,a.kind)
