"""CPU-only runtime/identity monitor for the authorized frozen Gate B child."""
import hashlib,json,os
from pathlib import Path
import signal,subprocess,sys,time
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];RUNS=BASE/'training_v1'
ready=json.loads((BASE/'gate_b_launch_ready_v1.json').read_text())
packet=json.loads((BASE/'gate_b_preresume_private_v1.json').read_text())
start=time.monotonic();initial_wall=ready['gate_a_seconds'];problem=None
log=(BASE/'gate_b_console_v1.txt').open('x')
child=subprocess.Popen(ready['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
with (BASE/'gate_b_process_v1.json').open('x') as f:json.dump({'pid':child.pid,'supervisor_pid':os.getpid(),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'monitor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f)
def save(obj,path):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
try:
 while child.poll() is None:
  elapsed=time.monotonic()-start
  if initial_wall+elapsed>24*3600:problem='actual operational wall exceeded 24 GPU hours'
  # Only explicit known model paths are visited; never discover EEG/manifests.
  seconds={name:[] for name in packet['config_sha256'][:0]}
  done=[];active=[];max_alloc=0;max_reserve=0
  variants=['q1','availability','background','rounded_q1','raw_context','evidence']
  seconds={v:[] for v in variants}
  for seed in [123,456,789]:
   for name in variants:
    run=RUNS/f'seed_{seed}'/name
    history=run/'history.csv'
    # CSV is atomically replaced. History rows supply cumulative costs.
    if history.exists():
     import csv
     with history.open() as f:rows=list(csv.DictReader(f))
     if rows:
      seconds[name].extend(float(h['seconds']) for h in rows)
      last=rows[-1];active.append({'seed':seed,'variant':name,'epoch':int(last['epoch']),'best_epoch':int(last['best_epoch']),'stale':int(last['stale']),'val_f1':float(last['val_subject_macro_f1'])})
      max_alloc=max(max_alloc,int(last['peak_gpu_allocated_bytes']));max_reserve=max(max_reserve,int(last['peak_gpu_reserved_bytes']))
    if (run/'COMPLETE_SEAL.json').exists():done.append([seed,name])
  estimate=180*sum(max(sum(seconds[v])/len(seconds[v]), packet['budget_gpu_hours']*0+json.loads((RUNS/'gate_a.json').read_text())['epoch_seconds'][v]) if seconds[v] else json.loads((RUNS/'gate_a.json').read_text())['epoch_seconds'][v] for v in variants)/3600
  if estimate>24:problem=f'projected maximum workload {estimate:.3f} GPU hours exceeds 24'
  # Hash frozen scientific source against the reviewed implementation inventory.
  for rel,h in packet['implementation']['files'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:problem='implementation changed: '+rel;break
  gpus=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
  for line in gpus.splitlines():
   if line.strip() and line.split(',')[0].strip()!=str(child.pid):problem='another compute process appeared; stop this project, leave it untouched'
  status={'status':'RUNNING' if not problem else 'STOPPING_INTEGRITY_OR_RESOURCE_GATE','elapsed_gate_b_seconds':elapsed,
   'completed_runs':done,'latest_epochs':active,'maximum_budget_projected_gpu_hours':estimate,'peak_last_epoch_allocated_bytes':max_alloc,
   'peak_last_epoch_reserved_bytes':max_reserve,'gpu_processes':gpus,'problem':problem}
  save(status,BASE/'gate_b_live_status_v1.json')
  if problem:
   child.terminate();break
  time.sleep(10)
 rc=child.wait();log.close()
 save({'status':'CHILD_EXITED_OK' if rc==0 and not problem else 'BLOCKED','returncode':rc,'problem':problem,
 'gate_b_elapsed_wall_seconds':time.monotonic()-start,'gate_a_epoch_seconds':initial_wall},BASE/'gate_b_execution_v1.json')
 print('Gate B child exit:',rc,'problem:',problem,flush=True)
 sys.exit(0 if rc==0 and not problem else 1)
except BaseException:
 if child.poll() is None:child.terminate();child.wait()
 raise
