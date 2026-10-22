"""Phase identities and bounded nonoverlapping scientific accounting."""
import json
import time
from contextlib import contextmanager
import os
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
class IntegrityError(ValueError):pass
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha256(data):return hashlib.sha256(data).hexdigest()
def stamp(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def atomic_json(path,value):
    temp=path.with_name(path.name+'.writing');write_new(temp,value);os.replace(temp,path)
BASE='1b6f25ee48945d624cfde61af4824efcd04bb96d'
ROOT_AUTH_SHA256='12a2814f8d7163a9066f3fb8e31f915325efb4ffc3582f5d0ca859f449b466e4'
PRIVATE=ROOT/'mist_transfer_runs/mist_cove/phase14_v1'
PUBLIC=ROOT/'reports/mist_cove/phase14_v1'
def read_json(path):return json.loads(path.read_text())
@contextmanager
def measured(name,**counts):
    path=PUBLIC/'resource_stages.json';prior=read_json(path) if path.exists() else {}
    if name in prior:raise IntegrityError('stage already attempted; no silent retries')
    started=time.perf_counter();status='COMPLETE'
    try:yield counts
    except Exception:status='FAILED';raise
    finally:
        current=read_json(path) if path.exists() else {}
        current[name]={'status':status,'wall_seconds':time.perf_counter()-started,**counts};atomic_json(path,current)
def check_budget(started,queries=0):
    path=PUBLIC/'resource_stages.json';spent=sum(r['wall_seconds'] for r in read_json(path).values()) if path.exists() else 0
    if spent+time.perf_counter()-started>24000 or queries>90000000:raise TimeoutError('Phase XIV ceiling')
def ledger(ids,status,findings,artifacts=(),science=None,block=None):
    path=PUBLIC/'progress.json';value=read_json(path)
    for row in value['work_packages']:
        if int(row['id'][2:]) not in ids:continue
        row.update(status=status,engineering_status='PASS' if status=='COMPLETE' else status,findings=[findings],artifacts=list(artifacts),scientific_status=science or 'NOT_APPLICABLE',block_reason=block)
    atomic_json(path,value)


def authorized_root():
    path=PRIVATE/'local_authorization.json';fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if before.st_nlink!=1 or not 0<before.st_size<=4096:raise IntegrityError('bounded local authorization required')
        data=os.pread(fd,before.st_size,0)
        if stamp(before)!=stamp(os.fstat(fd)) or stamp(before)!=stamp(os.stat(path,follow_symlinks=False)):raise IntegrityError('local authorization changed')
    finally:os.close(fd)
    value=json.loads(data)
    if set(value)!={'root'} or sha256(value['root'].encode())!=ROOT_AUTH_SHA256:raise IntegrityError('explicit authorized root differs')
    return value['root']
