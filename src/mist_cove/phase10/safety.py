"""Fail-closed supported entrypoints. No authorization flag unlocks data IO."""
import json
import os
from pathlib import Path
import tempfile
from .dependency_graph import ROOT

PUBLIC=ROOT/'reports/mist_cove/phase10_v1'
PRIVATE=ROOT/'mist_transfer_runs/mist_cove/phase10_v1'


def source_access(role, path=None, **kwargs):
    # Intentionally do not stringify, resolve, stat or open caller paths.
    raise PermissionError('Phase X has no source artifact or dataset reader for any role')


def mock_request(role='synthetic_mock', *, input_path=None, checkpoint_path=None, bank_path=None):
    if role!='synthetic_mock' or any(v is not None for v in (input_path,checkpoint_path,bank_path)):
        raise PermissionError('only generated in-memory mocks are supported')


def confined_output(path):
    p=Path(path)
    if not p.is_absolute():raise ValueError('internal absolute output path required')
    if p.is_relative_to(PUBLIC):relative=p.relative_to(ROOT)
    elif p.is_relative_to(PRIVATE):relative=p.relative_to(ROOT)
    else:raise ValueError('output outside Phase X roots')
    if '..' in relative.parts:raise ValueError('output traversal forbidden')
    current=ROOT
    for part in relative.parts:
        current=current/part
        if current.is_symlink():raise ValueError('symlink output forbidden')
    return p


def write_json(path,value,*,replace=False):
    p=confined_output(path);payload=json.dumps(value,indent=2,allow_nan=False).encode()+b'\n'
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.phase10-')
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        if replace:os.replace(tmp,p)
        else:os.link(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def no_sensitive_exports(value):
    forbidden={'state_dict','waveforms','normalization_values','source_path','annotation_path','checkpoint_weights','full_truth_table','margins'}
    if isinstance(value,dict):
        if set(value)&forbidden:raise ValueError('sensitive export schema')
        for v in value.values():no_sensitive_exports(v)
    elif isinstance(value,(tuple,list)):
        for v in value:no_sensitive_exports(v)
    elif isinstance(value,str):
        if value.startswith(('/', '~','file:')) or 'mist_transfer_runs/' in value:
            raise ValueError('private/sensitive path export')
