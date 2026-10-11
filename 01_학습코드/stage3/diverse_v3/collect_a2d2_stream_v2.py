"""Verify entire official tar stream, retaining every third front-camera frame.
Reduces storage, not network bytes. Exact original metadata; resized images.
"""
import io,sys,json,hashlib,tarfile,time,shutil,traceback
from pathlib import Path
import requests
from PIL import Image,ImageOps

ROOT=Path('/content/MyDrive/MyDrive/stage3_data/diverse_v2_20260917')
LOCAL=Path('/content/diverse_v2')
EXPECTED={'20190401121727':('66bf8642c0ee221d1c5664c7b6362a6c',98142607360),
          '20190401145936':('46934d1b0426b477cd5f5b0149ee57ba',78954700800)}

class HashedStream:
    def __init__(self,raw,stamp):self.raw=raw;self.h=hashlib.md5();self.n=0;self.last=time.time();self.stamp=stamp
    def read(self,n=-1):
        data=self.raw.read(n);self.h.update(data);self.n+=len(data)
        if time.time()-self.last>30:
            print('A2D2_STREAM',self.stamp,self.n,flush=True);self.last=time.time()
        return data

def main(stamp):
    ROOT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(exist_ok=True)
    reportpath=ROOT/(stamp+'_stream_report.json')
    if reportpath.exists() and json.loads(reportpath.read_text()).get('verified'):return
    md5,size=EXPECTED[stamp];name=f'camera_lidar-{stamp}_camera_frontcenter.tar'
    url='https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/'+name
    target=LOCAL/(stamp+'_selected.tar');temp=target.with_suffix('.partial.tar')
    cached=None;cache_index={}
    if temp.exists():
        resume=temp.with_name(stamp+'_resume_'+str(time.time_ns())+'.tar');temp.rename(resume)
        try:
            cached=tarfile.open(resume,'r:')
            cache_index={m.name:m for m in cached.getmembers() if m.isfile()}
            print('REUSING_SELECTED_FRAME_CACHE',stamp,len(cache_index),flush=True)
        except tarfile.TarError:
            if cached:cached.close()
            cached=None;cache_index={}
    count=0
    with requests.get(url,stream=True,timeout=(60,180)) as response:
        response.raise_for_status();assert int(response.headers['Content-Length'])==size
        stream=HashedStream(response.raw,stamp)
        with tarfile.open(fileobj=stream,mode='r|',bufsize=1024*1024) as source,tarfile.open(temp,'w') as dest:
            for m in source:
                if not m.isfile() or '/cam_front_center/' not in m.name:continue
                p=Path(m.name)
                if p.suffix not in ['.png','.json'] or int(p.stem.split('_')[-1])%3:continue
                assert not p.is_absolute() and '..' not in p.parts
                if m.name in cache_index:
                    data=cached.extractfile(cache_index[m.name]).read()
                else:data=source.extractfile(m).read()
                if p.suffix=='.png' and m.name not in cache_index:
                    with Image.open(io.BytesIO(data)) as im:
                        buf=io.BytesIO();ImageOps.pad(im.convert('RGB'),(384,384),color=(124,116,104)).save(buf,format='PNG');data=buf.getvalue()
                info=tarfile.TarInfo(m.name);info.size=len(data);dest.addfile(info,io.BytesIO(data));count+=1
        # tarfile stops at archive EOF blocks; drain trailing padding into MD5.
        while stream.read(8*1024*1024):pass
        assert stream.n==size and stream.h.hexdigest()==md5,(stream.n,stream.h.hexdigest())
    temp.replace(target)
    if cached:cached.close()
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    expected_sha=sha(target);shutil.copyfile(target,ROOT/target.name);assert sha(ROOT/target.name)==expected_sha
    report={'source':url,'source_bytes':size,'official_md5':md5,'full_source_md5_verified':True,
        'selected_file':target.name,'selected_bytes':target.stat().st_size,'selected_sha256':expected_sha,
        'files':count,'verified':True,'processing':'front-center index modulo3=0; RGB 384 letterbox PNG; original camera JSON'}
    reportpath.write_text(json.dumps(report,indent=2));print('A2D2_STREAM_VERIFIED',json.dumps(report),flush=True)

if __name__=='__main__':
    try:main(sys.argv[1])
    except Exception:
        text=traceback.format_exc();print(text,flush=True);(ROOT/(sys.argv[1]+'_failure.txt')).write_text(text);raise
