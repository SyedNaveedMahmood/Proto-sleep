"""New-state creation seals and default-only restricted loading."""
import io
import json
import os
import stat
import torch
from mist_cove.phase11b.security import (AuthenticatedBytes, authenticated_snapshot,
                                        restricted_load, sha256, stamp, IntegrityError)
from mist_cove.phase11b.historical import named_state_digest, canonical


def read_descriptor_bytes(fd, path, limit):
    original = os.fstat(fd)
    if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1 or not 0 < original.st_size <= limit:
        raise IntegrityError('regular, unlinked, bounded artifact required')
    blocks = []
    offset = 0
    while offset < original.st_size:
        block = os.pread(fd, min(65536, original.st_size-offset), offset)
        if not block:
            raise IntegrityError('truncated artifact')
        blocks.append(block)
        offset += len(block)
    if stamp(original) != stamp(os.fstat(fd)) or stamp(original) != stamp(os.stat(path, follow_symlinks=False)):
        raise IntegrityError('artifact identity or modification changed')
    return b''.join(blocks)


def save_sealed(path, payload, identity):
    from .provenance import write_new
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    if len(data) > 64*1024*1024:
        raise IntegrityError('new state exceeds64MiB')
    temporary = path.with_name(path.name+'.writing')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise IntegrityError('refusing to replace a sealed state')
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    seal = {'byte_sha256': sha256(data), 'bytes': len(data), 'identity': identity,
            'state_digest': named_state_digest(payload['state_dict'])}
    write_new(path.with_suffix('.seal.json'), seal)
    return seal


def load_sealed(path, identity):
    # The seal is phase-owned creation metadata; its identity must match the
    # prospectively frozen run contract supplied by the caller.
    seal = json.loads(path.with_suffix('.seal.json').read_text())
    if seal['identity'] != identity:
        raise IntegrityError('checkpoint resume identity mismatch before parsing')
    with authenticated_snapshot(path, seal['byte_sha256'], allowlist=[path]) as snapshot:
        payload = restricted_load(snapshot)
    if named_state_digest(payload['state_dict']) != seal['state_digest']:
        raise IntegrityError('checkpoint named state differs from creation seal')
    return payload, seal
