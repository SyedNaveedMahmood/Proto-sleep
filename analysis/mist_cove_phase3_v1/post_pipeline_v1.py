"""Wait for the frozen experiment, then run private read-only final analyses.

Does not train, retry a failed gate, change scientific code or publish anything.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
names = ['collect_phase3_results_v1.py', 'waveform_analysis_v1.py', 'individual_attribution_summary_v1.py']
hashes = {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in names}
status_path = BASE/'post_pipeline_status_v1.json'


def status(state, **extra):
    obj = {'status': state, 'analysis_hashes': hashes, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, status_path)


if status_path.exists():
    raise RuntimeError('Existing pipeline state: inspect instead of overwrite')
status('WAITING_FOR_ALL_18_TRAINING_RUNS')
while not (BASE/'gate_b_execution_v1.json').exists():
    time.sleep(10)
execution = json.loads((BASE/'gate_b_execution_v1.json').read_text())
if execution['status'] != 'CHILD_EXITED_OK':
    status('BLOCKED_TRAINING_GATE', execution=execution)
    sys.exit(1)
for name in names:
    if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != hashes[name]:
        status('BLOCKED_ANALYSIS_IDENTITY', file=name)
        sys.exit(1)
    status('RUNNING_PRIVATE_POSTFIT_ANALYSIS', step=name)
    with (BASE/(name.removesuffix('.py')+'_console.txt')).open('x') as log:
        child = subprocess.Popen([sys.executable, str(BASE/name)], cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    if code:
        status('BLOCKED_POSTFIT_ANALYSIS', step=name, returncode=code)
        sys.exit(code)
status('PRIVATE_POSTFIT_ANALYSES_COMPLETE_REQUIRES_REVIEW_BEFORE_PUBLICATION')
