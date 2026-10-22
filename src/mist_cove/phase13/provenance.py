"""Phase-owned identities and TRAIN-only access; historical code is unchanged."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from mist_cove.phase11b.historical import ROOT, canonical, authenticate_model, lineage, named_state_digest
from mist_cove.phase11b.security import authenticated_snapshot, restricted_load, sha256, IntegrityError
from mist_cove.phase11b.source import SourceAccess

BASE = '798af5629b7067af5230f64681926829e1178007'
PUBLIC = ROOT/'reports/mist_cove/phase13_v1'
PRIVATE = ROOT/'mist_transfer_runs/mist_cove/phase13_v1'
OLD_PUBLIC = ROOT/'reports/mist_cove/phase11b_v1'
OLD_PRIVATE = ROOT/'mist_transfer_runs/mist_cove/phase11b_v1'


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.writing')
    write_new(temporary, value)
    os.replace(temporary, path)


class TrainOnlyAccess(SourceAccess):
    def records(self, role):
        if role != 'train':
            raise IntegrityError('Phase XIII permits TRAIN only, before descriptor access')
        return super().records(role)

    def require_record(self, record):
        if record.get('split') != 'train':
            raise IntegrityError('Phase XIII forbids VAL and excluded roles before source access')
        return super().require_record(record)


def setup_source():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, payload, root, child, receipt = authenticate_model()
    frozen, normalization, provenance = lineage(payload, root, child)
    gate = json.loads((PUBLIC/'authentication.json').read_text())
    if gate['status'] != 'PASS' or gate['named_state_digest'] != receipt['named_state_digest']:
        raise IntegrityError('fresh Phase XIII authenticated trained-adapter gate required')
    return model, TrainOnlyAccess(frozen, normalization, gate['adapter'])


def authenticated_json_identity(path, expected_canonical, limit=2*1024*1024):
    # These historical phase-owned JSON identities were published as CANONICAL
    # digests. Parse bounded nonexecutable JSON, then verify before using values.
    from mist_cove.phase11a.provenance import strict_json
    # Existing bounded reader is not used here: keep no-follow snapshot defenses
    # without claiming that a canonical digest is a serialized-byte reference.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NOATIME)
    from .seals import read_descriptor_bytes
    try:
        data = read_descriptor_bytes(fd, path, limit)
    finally:
        os.close(fd)
    value = strict_json(data)
    if sha256(canonical(value)) != expected_canonical:
        raise IntegrityError('historically anchored canonical JSON differs')
    return value


@contextmanager
def isolated_rng(seed):
    python_state = random.getstate()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            torch.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)


def authenticate():
    from mist_cove.phase11b.adapter_checks import trained_adapter_checks
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, payload, root, child, receipt = authenticate_model()
    _, _, lineage_receipt = lineage(payload, root, child)
    adapter = trained_adapter_checks(model)
    result = {**receipt, 'adapter': adapter, 'lineage': lineage_receipt,
              'wall_seconds': time.perf_counter()-start,
              'raw_TRAIN_read': False, 'raw_VAL_read': False}
    write_new(PUBLIC/'authentication.json', result)
    print(json.dumps({'authentication': result['status'], 'adapter': adapter['status']}))
