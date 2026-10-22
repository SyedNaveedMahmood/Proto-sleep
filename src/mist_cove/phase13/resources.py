"""Measured nonoverlapping phase stages, separate study and engineering costs."""
import json
import time
from contextlib import contextmanager
from .provenance import PUBLIC, atomic_json


@contextmanager
def measured(stage, track, **counts):
    path = PUBLIC/'resource_stages.json'
    prior = json.loads(path.read_text()) if path.exists() else {}
    if stage in prior:
        raise ValueError('stage already executed; no silent scientific retry')
    if sum(v['wall_seconds'] for v in prior.values() if v['track'] != 'ENGINEERING') >= 14400:
        raise TimeoutError('scientific CPU budget exhausted')
    started = time.perf_counter()
    status = 'COMPLETE'
    try:
        yield counts
    except Exception:
        status = 'FAILED'
        raise
    finally:
        prior[stage] = {'track':track,'status':status,'wall_seconds':time.perf_counter()-started,**counts}
        atomic_json(path, prior)


def check_remaining(started):
    path = PUBLIC/'resource_stages.json'
    prior = json.loads(path.read_text()) if path.exists() else {}
    spent = sum(v['wall_seconds'] for v in prior.values() if v['track'] != 'ENGINEERING')
    if spent+time.perf_counter()-started > 14400:
        raise TimeoutError('scientific CPU budget would be exceeded')
