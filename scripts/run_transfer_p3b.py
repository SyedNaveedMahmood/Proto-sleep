#!/usr/bin/env python
"""Dedicated authorized full-data P3B runner; P3A entrypoint stays smoke-only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import fcntl
from pathlib import Path
import signal
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mist_transfer.full_experiment import run_full,DEFAULT_OUTPUT
from mist_transfer.protocol import DEFAULT_SPLIT
from mist_transfer.provenance import atomic_json,atomic_text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,default=DEFAULT_SPLIT)
    p.add_argument('--config',type=Path,default=Path('configs/p3b_sc_matched_v1.json'))
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--mode',choices=['gate-a','gate-b'],default='gate-a')
    p.add_argument('--correctness',type=Path)
    p.add_argument('--authorize-full-sc-train-val',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();lock_path=Path(__file__).resolve().parents[1]/'mist_transfer_runs/p3b/GPU.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    def request_pause(signum,frame):
        atomic_text('Operational pause requested; stop after next saved epoch.\n',a.output_dir/'PAUSE_REQUESTED')
    signal.signal(signal.SIGUSR1,request_pause)
    try:
        with lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('another P3B process holds the GPU lock')
            if not a.dry_run:
                gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
                if any(line.strip().isdigit() and int(line.strip())!=os.getpid() for line in gpu_pids.splitlines()):
                    raise ValueError('GPU has another compute process; leave it untouched')
            sys.exit(run_full(a.split,a.config,a.output_dir,a.mode,a.resume,a.authorize_full_sc_train_val,a.correctness,a.dry_run))
    except (ValueError,OSError,KeyError,RuntimeError,FloatingPointError) as exc:
        print('P3B BLOCKED: '+str(exc),file=sys.stderr,flush=True)
        atomic_json({'status':'BLOCKED_NOT_COMPLETE','reason':str(exc),'mode':a.mode,'reserved_access':False},a.output_dir/'BLOCKED.json')
        sys.exit(2)
