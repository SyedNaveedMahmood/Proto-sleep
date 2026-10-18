"""Versioned real-source gates; no fitting, discovery or automatic approval."""
import fcntl
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from mist_cove.phase2.provenance import ROOT, digest, file_sha, implementation_snapshot, environment, git_head
from mist_cove.phase2.source import SPLIT, SPLIT_SHA, VARIANTS, validate_source_config, validate_normalization, training_config, load_role
from mist_cove.phase2.training import configure_precision, sealed_train, validate_views
from mist_cove.phase2.adapter import build_adapter, adapter_logits
from mist_evidence.runtime import atomic_json, seed_all, EpochPaused

PRIVATE = ROOT / 'mist_transfer_runs/mist_cove/phase3_source_v1'
AUTHORIZATION = 'User 2026-10-18 MASTER PROMPT — MIST-COVE PHASE III: REAL EEG TRAINING AND SCIENTIFIC VALIDATION, sections 3, 8 and final instruction'
TRUSTED_FIT_SHA = '3f384f0f568b2c7a0ebcb9627f59686fb18f8bd45c207bfeb3d5d6e8edbcbcf3'
TRUSTED_NORMALIZATION_COMPLETION_SHA = 'a037ded9691206470430c27627303b8f72400891976af8e7308458aedd6427e5'


def prepared(c):
    c = dict(c)
    if c.pop('maximum_gpu_hours', None) != 24 or c.get('version') != 'mist-cove-phase3-source-v1' or c.get('status') != 'AUTHORIZED_PREFLIGHT_REVIEW_REQUIRED_FOR_TRAINING':
        raise ValueError('Phase III frozen identity/budget violated')
    c.update(version='mist-cove-source-prepared-v1', status='PREPARED_NOT_AUTHORIZED_OR_EXECUTED')
    validate_source_config(c)
    return c


def private_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to(PRIVATE.resolve()) or p == PRIVATE.resolve():
        raise ValueError('private Phase III subdirectory required')
    return p


def file_proof(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_sha(path)}


def check_proof(proof):
    if file_sha(proof['path']) != proof['sha256']:
        raise ValueError('artifact digest changed before use')
    return Path(proof['path'])


def reconciled_environment():
    result = environment('cuda')
    result['executable'] = sys.executable
    result['imports'] = {}
    for package, module in [('numpy', 'numpy'), ('torch', 'torch'), ('pyEDFlib', 'pyedflib'),
                            ('scikit-learn', 'sklearn'), ('scipy', 'scipy'), ('pytest', 'pytest'), ('mne', 'mne')]:
        loaded = import_module(module)
        actual, installed = str(loaded.__version__), metadata.version(package)
        if actual.split('+')[0] != installed.split('+')[0]:
            raise ValueError('dependency import/metadata mismatch: ' + package)
        result['imports'][package] = {'imported': actual, 'metadata': installed, 'path': loaded.__file__}
    return result


def require_idle_gpu():
    # Do not kill, signal or otherwise interfere with another GPU user.
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    other = [s.strip() for s in raw.splitlines() if s.strip() and s.strip() != str(os.getpid())]
    if other:
        raise RuntimeError('GPU busy with other compute processes: ' + ', '.join(other))
    free, total = torch.cuda.mem_get_info()
    if free < 1024**3:
        raise RuntimeError('less than 1 GiB free GPU memory; no CPU fallback')
    return {'free_bytes': free, 'total_bytes': total}


def verify_source_files(record):
    # Frozen descriptor identity was validated by load_development. Do not infer
    # identity or silently redirect to a similarly named file in another directory.
    if record.get('split') not in {'train', 'val'}:
        raise ValueError('unauthorized role before waveform access')
    paths = [Path(record[k]).resolve() for k in ('path', 'annotation_source')]
    if paths[0].parent != paths[1].parent or paths[0].parent.name != 'sleep-cassette':
        raise ValueError('frozen sources are not paired original Sleep-Cassette EDF')
    proofs = []
    for path, sha in zip(paths, ('source_sha256', 'annotation_sha256')):
        actual = file_sha(path)
        if actual != record[sha]:
            raise ValueError('frozen original EDF hash mismatch')
        proofs.append({'path': str(path), 'sha256': actual})
    return proofs


def load_sources(c):
    """Explicit authorized preflight caller only; guards precede source access."""
    prepared(c)
    split_path = ROOT / 'experiments/splits/p3a_sc_source_v1.json'
    if file_sha(split_path) != SPLIT_SHA:
        raise ValueError('frozen split SHA mismatch')
    from mist_transfer.protocol import read_frozen_split
    frozen = read_frozen_split(split_path)
    if frozen['split_fingerprint'] != SPLIT:
        raise ValueError('frozen split fingerprint mismatch')
    p3a, p3b = ROOT / 'mist_transfer_runs/p3a/cuda_smoke', ROOT / 'mist_transfer_runs/p3b/v1'
    if file_sha(p3a / 'COMPLETE.json') != TRUSTED_NORMALIZATION_COMPLETION_SHA or file_sha(p3b / 'fit_artifacts.receipt.json') != TRUSTED_FIT_SHA:
        raise ValueError('trusted historical artifact receipt changed')
    completion = json.loads((p3a / 'COMPLETE.json').read_text())
    fit = json.loads((p3b / 'fit_artifacts.receipt.json').read_text())
    norm_path = p3b / 'normalization.json'
    if file_sha(norm_path) != completion['outputs']['normalization.json'] or file_sha(norm_path) != fit['outputs']['normalization.json']:
        raise ValueError('trusted TRAIN normalization changed; never refit')
    train_rows, val_rows = load_role(frozen, 'train'), load_role(frozen, 'val')
    # Verify exact configured source location without discovering directory contents.
    import yaml
    canonical = Path(yaml.safe_load((ROOT / 'configs/data_paths.example.yaml').read_text())['sleep_edfx_sc_raw']).resolve()
    from mist_transfer.source_data import SourceRecording, validate_bank
    from mist_transfer.full_experiment import verify_grid
    source_proofs = []
    for r in train_rows + val_rows:
        source_proofs.extend(verify_source_files(r))
    norm = validate_normalization(json.loads(norm_path.read_text()), train_rows, val_rows)
    train = [SourceRecording(r, norm, smoke=False) for r in train_rows]
    val = [SourceRecording(r, norm, smoke=False) for r in val_rows]
    validate_views(train, val)
    grid = verify_grid(train, val)
    banks, bank_proofs = {}, {}
    for seed in c['seeds']:
        rel = f'seed_{seed}/training_waveform_bank.pt'
        p = p3b / rel
        if file_sha(p) != fit['outputs'][rel]:
            raise ValueError('trusted TRAIN bank hash mismatch')
        bank = torch.load(p, map_location='cpu', weights_only=True)
        if bank['seed'] != seed:
            raise ValueError('TRAIN bank seed mismatch')
        validate_bank(bank, train_rows, norm, verify_waveforms=True)
        banks[seed], bank_proofs[str(seed)] = bank, file_proof(p)
    proofs = {'normalization': file_proof(norm_path), 'normalization_completion': file_proof(p3a / 'COMPLETE.json'),
              'bank_receipt': file_proof(p3b / 'fit_artifacts.receipt.json'), 'banks': bank_proofs, 'sources': source_proofs,
              'source_location': {'configured_directory': str(canonical), 'configured_directory_exists': canonical.exists(),
                  'recorded_directories': sorted({str(Path(p['path']).parent) for p in source_proofs}),
                  'configured_directory_matches_frozen': all(Path(p['path']).parent == canonical for p in source_proofs),
                  'policy': 'Use exact frozen original EDF paths and hashes; never substitute or regenerate'}}
    return train, val, norm, banks, grid, proofs


def numerical_probe(train, val, norm, banks, device='cuda'):
    """Bounded true-source feasibility; gradients only, zero optimizer updates.

    Validation labels are never used. This is not an epoch throughput estimate
    and supplies no accuracy, checkpoint selection or trained model.
    """
    configure_precision(device)
    x = torch.from_numpy(train[0].x[:16]).to(device)
    vx = torch.from_numpy(val[0].x[:16]).to(device)
    labels = torch.from_numpy(train[0].y[:16]).to(device)[None]
    valid = labels >= 0
    results = []
    for name in VARIANTS:
        seed_all(123, True)
        model = build_adapter(name, norm, banks[123]).to(device)
        initial = {k: p.detach().clone() for k, p in model.named_parameters()}
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        logits = adapter_logits(model, x)
        loss = model.loss(logits[None], labels, valid)
        if not torch.isfinite(loss):
            raise ValueError('real TRAIN smoke loss nonfinite')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('real TRAIN smoke gradient nonfinite')
        model.eval()
        with torch.no_grad():
            v_logits = adapter_logits(model, vx)
            for z in (logits, v_logits):
                if z.shape != (16, 5) or not torch.isfinite(z).all():
                    raise ValueError('real five-stage logits nonfinite or wrong shape')
                torch.testing.assert_close(z.softmax(-1).sum(-1), torch.ones(16, device=device), rtol=0, atol=1e-6)
            residual = None
            if name not in {'raw_context', 'evidence'}:
                ev = model.network(vx, return_evidence=True)
                residual = float((ev.logits - ev.background_logits - ev.contributions.sum(1)).abs().max())
                if residual > 1e-5:
                    raise ValueError('real waveform additive reconciliation failed')
                intervals = ev.event_intervals[ev.valid]
                if ((intervals[:, 0] < 0) | (intervals[:, 1] > 3000) | (intervals[:, 1] - intervals[:, 0] != 200)).any():
                    raise ValueError('real waveform original interval invalid')
            for k, p in model.named_parameters():
                if not torch.equal(initial[k], p):
                    raise ValueError('preflight unexpectedly changed a model parameter')
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize()
        results.append({'variant': name, 'batch': 16, 'finite_logits_loss_gradients': True, 'optimizer_updates': 0,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'additive_max_abs_residual': residual,
                        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if torch.device(device).type == 'cuda' else 0})
        del model, initial, logits, loss, v_logits
        if name not in {'raw_context', 'evidence'}:
            del ev
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    return {'status': 'PASS', 'optimizer_updates': 0, 'validation_labels_used': False,
            'scope': 'UNTRAINED_BATCH16_NUMERICAL_FEASIBILITY_NOT_EPOCH_THROUGHPUT', 'runs': results}


def preflight(config_path, output, *, authorized=False):
    if not authorized:
        raise ValueError('explicit source TRAIN/VAL authorization required before any file access')
    output = private_path(output)
    if output.exists():
        raise FileExistsError('preflight output exists; preserve previous receipts')
    c = json.loads(Path(config_path).read_text())
    prepared(c)
    configure_precision('cuda')
    torch.set_num_threads(2)
    hardware = require_idle_gpu()
    env = reconciled_environment()
    code = implementation_snapshot()
    start = time.perf_counter()
    train, val, norm, banks, grid, proofs = load_sources(c)
    # Exercise the genuine lazy training path, every physical position including unscored.
    scan = {}
    for role, rows in [('train', train), ('val', val)]:
        began = time.perf_counter()
        count = 0
        for row in rows:
            for lo in range(0, len(row.y), 16):
                x = row.x[lo:lo + 16]
                if x.dtype != np.float32 or not np.isfinite(x).all():
                    raise ValueError('real normalized input is nonfinite or not FP32')
                count += len(x)
        scan[role] = {'physical_epochs': count, 'seconds': time.perf_counter() - began,
                      'read_batches': sum(r.x.read_batches for r in rows), 'peak_batch_bytes': max(r.x.peak_bytes for r in rows)}
    numerical = numerical_probe(train, val, norm, banks)
    if code != implementation_snapshot():
        raise ValueError('implementation changed during preflight')
    output.mkdir(parents=True)
    receipt = {'status': 'PREFLIGHT_PASS_REVIEW_REQUIRED_NO_TRAINING', 'scope': 'SC_TRAIN_VAL_ONLY_AUTHORIZED',
               'authorization_reference': AUTHORIZATION, 'config_sha256': digest(c), 'config_file': file_proof(config_path),
               'implementation': code, 'implementation_commit': git_head(), 'environment': env, 'hardware': hardware,
               'split_fingerprint': SPLIT, 'split_file_sha256': SPLIT_SHA, 'grid': grid, 'proofs': proofs,
               'source_scan': scan, 'normalization_refitted': False, 'bank_source_crops_exact': True,
               'real_source_numerical_probe': numerical,
               'training_executed': False, 'reserved_access': False, 'st_access': False, 'shhs_access': False,
               'wall_seconds': time.perf_counter() - start}
    atomic_json(receipt, output / 'preflight.json')
    return receipt


def validate_review(proposal_path, review_path):
    """A recorded external review is required; this module never writes one."""
    proposal = json.loads(Path(proposal_path).read_text())
    review = json.loads(Path(review_path).read_text())
    if (review.get('proposal_sha256') != file_sha(proposal_path) or review.get('decision') != 'APPROVE_GATE_A'
            or review.get('scope') != 'SC_TRAIN_VAL_ONLY' or not review.get('reviewer')
            or review.get('reviewer') in {'agent', 'codex', 'self'} or not review.get('user_review_reference')):
        raise ValueError('independent user launch review missing/mismatched')
    if proposal.get('status') != 'READY_FOR_INDEPENDENT_REVIEW' or proposal.get('config_sha256') is None:
        raise ValueError('launch proposal is not ready')
    if proposal['implementation_sha256'] != implementation_snapshot()['sha256']:
        raise ValueError('reviewed implementation changed')
    for proof in proposal['artifacts'].values():
        check_proof(proof)
    pre = json.loads(check_proof(proposal['artifacts']['preflight']).read_text())
    if pre['status'] != 'PREFLIGHT_PASS_REVIEW_REQUIRED_NO_TRAINING' or pre['config_sha256'] != proposal['config_sha256']:
        raise ValueError('reviewed preflight mismatch')
    if pre['environment'] != reconciled_environment():
        raise ValueError('reviewed environment changed')
    if pre.get('real_source_numerical_probe', {}).get('status') != 'PASS':
        raise ValueError('real source numerical feasibility missing')
    validate_correctness(proposal['artifacts'], pre['environment'])
    return proposal


def validate_correctness(artifacts, env):
    cpu = json.loads(check_proof(artifacts['cpu']).read_text())
    cuda = json.loads(check_proof(artifacts['cuda']).read_text())
    if (cpu.get('status') != 'PASS' or cpu.get('failed') != 0 or cpu.get('errors') != 0
            or cpu.get('passed', 0) < 357 or cpu.get('implementation_sha256') != implementation_snapshot()['sha256']
            or cpu.get('environment') != env):
        raise ValueError('CPU correctness not bound to current source environment/code')
    for proof in cpu['artifacts'].values():
        check_proof(proof)
    if (cuda.get('status') != 'PASS' or cuda.get('real_data_accessed') is not False
            or cuda.get('implementation', {}).get('sha256') != implementation_snapshot()['sha256']
            or cuda.get('environment') != environment('cuda')
            or {r.get('method') for r in cuda.get('runs', [])} != set(VARIANTS)
            or len(cuda['runs']) != 6 or not all(r.get('model_optimizer_rng_history_exact') is True for r in cuda['runs'])):
        raise ValueError('six-method CUDA exact-resume correctness missing/mismatched')


def launch_proposal(preflight_path, cpu_path, cuda_path, output):
    """Collect evidence for review, without writing an approval record."""
    output = private_path(output)
    if output.exists():
        raise FileExistsError('launch proposal already exists')
    pre = json.loads(Path(preflight_path).read_text())
    if pre.get('status') != 'PREFLIGHT_PASS_REVIEW_REQUIRED_NO_TRAINING' or pre['implementation']['sha256'] != implementation_snapshot()['sha256']:
        raise ValueError('passing current preflight required')
    artifacts = {'preflight': file_proof(preflight_path), 'cpu': file_proof(cpu_path), 'cuda': file_proof(cuda_path)}
    validate_correctness(artifacts, pre['environment'])
    proposal = {'status': 'READY_FOR_INDEPENDENT_REVIEW', 'approval_recorded': False,
                'scope': 'SC_TRAIN_VAL_ONLY', 'authorization_reference': AUTHORIZATION,
                'config_sha256': pre['config_sha256'], 'implementation_sha256': implementation_snapshot()['sha256'],
                'artifacts': artifacts, 'requested_review_scope': 'Gate A six seed-123 initial epochs; Gate B only if measured maximum budget <=24 GPU hours',
                'real_source_throughput': 'NOT_TESTED_UNTIL_REVIEWED_GATE_A', 'training_executed': False}
    atomic_json(proposal, output)
    return proposal


def run_reviewed(config_path, proposal_path, review_path, output, *, gate, authorized=False):
    if not authorized:
        raise ValueError('explicit source authorization required before any file access')
    output = private_path(output)
    configure_precision('cuda')
    torch.set_num_threads(2)
    c = json.loads(Path(config_path).read_text())
    original = prepared(c)
    proposal = validate_review(proposal_path, review_path)
    if digest(c) != proposal['config_sha256'] or gate not in {'a', 'b'}:
        raise ValueError('reviewed config/gate mismatch')
    require_idle_gpu()
    output.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'gpu.writer.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        train, val, norm, banks, grid, proofs = load_sources(c)
        provenance = {'scope': 'SC_TRAIN_VAL_ONLY', 'config': c, 'reviewed_proposal_sha256': file_sha(proposal_path),
                      'grid': grid, 'proofs': proofs}
        if gate == 'b':
            marker = json.loads((output / 'gate_a.json').read_text())
            if marker['proposal_sha256'] != file_sha(proposal_path) or marker['status'] != 'PASS' or marker['maximum_gpu_hours'] > 24:
                raise ValueError('Gate A budget/identity did not pass')
            for proof in marker['boundaries'].values():
                check_proof(proof)
        results, durations, boundaries = [], [], {}
        for seed in ([123] if gate == 'a' else c['seeds']):
            for name in VARIANTS:
                require_idle_gpu()
                seed_all(seed, True)
                model = build_adapter(name, norm, banks[seed])
                run = output / f'seed_{seed}' / name
                if gate == 'a' and (run / 'EPOCH_SEAL.json').exists():
                    raise ValueError('Gate A output already paused; preserve it for reviewed Gate B or recovery review')
                try:
                    _, metrics, _ = sealed_train(model, train, val, training_config(original, seed), run,
                        provenance=provenance, device='cuda', bank=banks[seed], resume=(run / 'EPOCH_SEAL.json').exists(),
                        pause_after_epoch=1 if gate == 'a' else None)
                    results.append({'seed': seed, 'variant': name, 'metrics': metrics,
                                    'parameters': sum(p.numel() for p in model.parameters())})
                except EpochPaused:
                    if gate != 'a':
                        raise
                if gate == 'a':
                    boundary = json.loads((run / 'EPOCH_SEAL.json').read_text())
                    durations.append(boundary['resources']['seconds'] + boundary['checkpoint_write_seconds'])
                    atomic_json(boundary, run / 'gate_a_boundary.json')
                    boundaries[name] = file_proof(run / 'gate_a_boundary.json')
                del model
                torch.cuda.empty_cache()
        if gate == 'a':
            estimate = 180 * sum(durations) / 3600
            result = {'status': 'PASS' if estimate <= 24 else 'BLOCKED_OVER_24_GPU_HOURS', 'maximum_gpu_hours': estimate,
                      'epoch_seconds': dict(zip(VARIANTS, durations)), 'boundaries': boundaries, 'proposal_sha256': file_sha(proposal_path)}
            atomic_json(result, output / 'gate_a.json')
        else:
            result = {'status': 'ALL_18_SOURCE_RUNS_COMPLETE', 'runs': results}
            atomic_json(result, output / 'results.json')
        return result
