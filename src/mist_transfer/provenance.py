"""Reuse v1 digests/atomic JSON; add audit transaction and CPU-only provenance."""
from __future__ import annotations
import csv
import io
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from mist_evidence.data import fingerprint, sha256_file
from mist_evidence.runtime import atomic_json


def atomic_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_csv(rows, fields, path):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(stream.getvalue(), path)


def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': dict(sorted((d.metadata['Name'], d.version) for d in metadata.distributions()
                                    if d.metadata['Name']))}


def git_info():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    # Source digest protects resume while implementation is uncommitted. Output
    # reports/logs are deliberately excluded from the implementation digest.
    files = [p for base in ('src/mist_transfer', 'scripts') for p in (root/base).glob('*.py')]
    return {'sha': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'dirty': bool(git('status', '--porcelain')),
            'implementation_hash': fingerprint({str(p.relative_to(root)): sha256_file(p) for p in sorted(files)})}


class AuditRun:
    """Single-writer, atomic completion; interrupted audit can restart on same inputs.

    Completion is established only after every output digest is verified. This
    deliberately leaves the v1 training/checkpoint protocol untouched.
    """
    def __init__(self, output, config, sources, seed=123, resume=False, split_ids=None):
        self.output = Path(output).resolve()
        self.identity = {'git': git_info(), 'environment': environment(), 'config': config,
                         'config_hash': fingerprint(config), 'source_recording_hashes': sources,
                         'seed': seed, 'split_ids': split_ids or {}, 'output_directory': str(self.output),
                         'leakage_flags': {'split_selection': False, 'test_metrics': False,
                                           'shhs_access': False}}
        # Dirty status can change as reports are written; code hash still binds code.
        signature_identity = json.loads(json.dumps(self.identity))
        signature_identity['git'].pop('dirty')
        self.signature = fingerprint(signature_identity)
        self.output.mkdir(parents=True, exist_ok=True)
        self.complete = self.output/'COMPLETE.json'
        self.provenance = self.output/'provenance.json'
        if self.provenance.exists():
            if not resume:
                raise FileExistsError('output exists: use --resume or a new output directory')
            previous = json.loads(self.provenance.read_text())
            if previous['signature'] != self.signature:
                raise ValueError('resume signature mismatch: source/config/code/environment changed')
            if self.complete.exists():
                done = json.loads(self.complete.read_text())
                if done['signature'] != self.signature:
                    raise ValueError('completion signature mismatch')
                for name, digest in done['outputs'].items():
                    if sha256_file(self.output/name) != digest:
                        raise ValueError('completion output digest mismatch')
        else:
            atomic_json({**self.identity, 'signature': self.signature}, self.provenance)

    def finish(self, outputs, gate):
        atomic_json({'signature': self.signature, 'outputs': {name: sha256_file(self.output/name)
                    for name in outputs}, 'gate': gate, 'training_performed': False}, self.complete)
