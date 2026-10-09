#!/usr/bin/env python
"""Record focused/regression exits plus exact interrupted CUDA resume evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mist_transfer.provenance import atomic_json,git_info,sha256_file
from mist_transfer.experiment import reused_implementation_hash

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path('mist_transfer_runs/p3b'))
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);info=git_info()
    if info['dirty']:raise SystemExit('Commit the implementation before generating the launch correctness receipt.')
    logs={}
    commands=[('focused',['tests/transfer/test_full_experiment.py','tests/transfer/test_source_protocol.py'],False),
        ('regression',['tests'],False),('cuda_resume',['tests/transfer/test_full_experiment.py','-k','cuda'],True)]
    for name,tests,cuda in commands:
        path=a.output_dir/(name+'.txt');env=os.environ.copy();env['PYTHONPATH']='src'
        if not cuda:env['CUDA_VISIBLE_DEVICES']=''
        cmd=[sys.executable,'-m','pytest',*tests,'-ra'];print('CHECK '+name+': '+' '.join(cmd),flush=True)
        with path.open('w') as log:
            process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            status=process.wait()
        if status:raise SystemExit('P3B correctness failed: '+name)
        if cuda and ('4 passed' not in path.read_text() or 'skipped' in path.read_text().split('short test summary info')[-1]):
            raise SystemExit('All four CUDA resume fixtures must actually execute.')
        logs[str(path)]=sha256_file(path)
    after=git_info()
    if after!=info:raise SystemExit('Implementation changed during correctness checks')
    atomic_json({'gate':'PASS','git_sha':info['sha'],'implementation_hash':info['implementation_hash'],
        'reused_implementation_hash':reused_implementation_hash(),'cuda_exact_resume_all_four':True,'logs':logs,
        'reserved_access':False},a.output_dir/'correctness.json')
    print('P3B CORRECTNESS PASS: CPU suites and all-four exact interrupted CUDA resume',flush=True)
