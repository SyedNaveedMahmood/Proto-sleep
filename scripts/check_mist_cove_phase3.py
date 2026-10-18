#!/usr/bin/env python
"""Environment-bound CPU/CUDA checks, generated data only, no source access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mist_cove.phase3.source import private_path, require_idle_gpu, reconciled_environment, file_proof
from mist_cove.phase2.provenance import implementation_snapshot, git_head
from mist_cove.phase2.training import configure_precision
from mist_evidence.runtime import atomic_json
import torch


def counts(path):
    suites = ET.parse(path).getroot().iter('testsuite')
    totals = {k: 0 for k in ('tests', 'failures', 'errors', 'skipped')}
    for suite in suites:
        for k in totals:
            totals[k] += int(suite.attrib.get(k, 0))
    return {**totals, 'passed': totals['tests'] - totals['failures'] - totals['errors'] - totals['skipped']}


def main(output):
    output = private_path(output)
    if output.exists():
        raise FileExistsError('choose a fresh correctness directory')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('commit implementation before environment-bound correctness checks')
    configure_precision('cuda'); torch.set_num_threads(2)
    require_idle_gpu()
    env, code = reconciled_environment(), implementation_snapshot()
    output.mkdir(parents=True)
    child = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    cpu_xml, gpu_xml = output / 'cpu.xml', output / 'cuda_pytest.xml'
    commands = [
        ('cpu', [sys.executable, '-m', 'pytest', 'tests', '-ra', f'--junitxml={cpu_xml}'], dict(child, CUDA_VISIBLE_DEVICES='')),
        ('cuda_pytest', [sys.executable, '-m', 'pytest', 'tests/mist_cove/test_phase2_training.py',
                        'tests/transfer/test_full_experiment.py', 'tests/evidence/test_model.py', '-k', 'cuda', '-ra', f'--junitxml={gpu_xml}'], child),
        ('cuda', [sys.executable, 'scripts/check_mist_cove_phase2_cuda.py', '--output-dir', str(output / 'cuda')], child)]
    for name, command, variables in commands:
        print('Running ' + name, flush=True)
        with (output / (name + '.txt')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=variables, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect private log')
    cpu, gpu = counts(cpu_xml), counts(gpu_xml)
    if cpu['failures'] or cpu['errors'] or gpu['passed'] != 11 or gpu['failures'] or gpu['errors'] or gpu['skipped']:
        raise ValueError('required correctness coverage did not pass')
    if code != implementation_snapshot():
        raise ValueError('code changed during correctness checks')
    receipt = {'status': 'PASS', 'scope': 'SYNTHETIC_SOFTWARE_ONLY', 'implementation_commit': git_head(),
               'implementation_sha256': code['sha256'], 'environment': env,
               'passed': cpu['passed'], 'failed': cpu['failures'], 'errors': cpu['errors'], 'skipped': cpu['skipped'],
               'cuda_pytest': gpu, 'artifacts': {p.name: file_proof(p) for p in [cpu_xml, gpu_xml, output/'cpu.txt', output/'cuda_pytest.txt', output/'cuda.txt']}}
    atomic_json(receipt, output / 'correctness.json')
    print(json.dumps({'cpu': cpu, 'cuda_pytest': gpu, 'status': 'PASS'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
