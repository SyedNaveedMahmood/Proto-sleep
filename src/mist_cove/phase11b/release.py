"""Reviewable publication allowlist and fail-closed aggregate ledger checks."""
import re
from pathlib import PurePosixPath

def validate_public_text(text,identifiers=()):
    if re.search(r'/(?:home|mnt|media|Users)/',text):raise ValueError('private absolute filesystem path')
    if re.search(r'\bSC\d{3,}[A-Za-z0-9_-]*\b',text):raise ValueError('real-source participant or recording identifier')
    for value in identifiers:
        if re.search(r'(?<![A-Za-z0-9])'+re.escape(str(value))+r'(?![A-Za-z0-9])',text):raise ValueError('private source identifier')

def validate_publication_path(path):
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or 'mist_transfer_runs' in p.parts:raise ValueError('private publication path')
    if p.suffix.lower() in ('.edf','.pt','.npz','.npy','.pkl','.pickle'):raise ValueError('private data/model artifact')

def validate_task_ledger(ledger,final=False):
    tasks=ledger['work_packages']
    if [t['id'] for t in tasks]!=['WP'+str(i).zfill(2) for i in range(1,37)]:raise ValueError('all36 ordered task IDs required')
    for task in tasks:
        if task['status'].startswith('COMPLETE') and not task['artifacts']:raise ValueError('completion without evidence')
        if task['status']=='BLOCKED' and not task['block_reason']:raise ValueError('block without exact reason')
        if final and task['status'] in ('PENDING','IN PROGRESS'):raise ValueError('unfinished authorized work')

def validate_resource_receipt(receipt):
    if receipt['core_new_neural_fits']>9 or receipt['scientific_optimizer_updates']>2700:raise ValueError('fit/update budget')
    if receipt['cached_head_queries']>50000000 or receipt['combined_full_and_cache_examples']>250000:raise ValueError('inference budget')
    if receipt['total_scientific_CPU_wall_seconds']>7200 or receipt['GPU_seconds']>36000:raise ValueError('time budget')
