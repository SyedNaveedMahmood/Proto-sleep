import io
import os
import pytest
import torch
from mist_cove.phase11b.security import authenticated_snapshot,restricted_load,AuthenticatedBytes,sha256,IntegrityError,LoadingBlocked


def test_snapshot_identity_digest_and_toctou(tmp_path):
    p=tmp_path/'generated';p.write_bytes(b'generated')
    with authenticated_snapshot(p,sha256(b'generated'),allowlist=[p]) as s:assert s.data==b'generated'
    with pytest.raises(IntegrityError,match='digest'):
        with authenticated_snapshot(p,'0'*64,allowlist=[p]):pass
    with pytest.raises(IntegrityError,match='changed'):
        with authenticated_snapshot(p,sha256(b'generated'),allowlist=[p]):p.write_bytes(b'mutation!')


@pytest.mark.parametrize('attack',['outside','symlink','ancestor','hardlink','oversize'])
def test_paths_rejected_before_content_access(tmp_path,attack):
    p=tmp_path/'generated';p.write_bytes(b'fixture');limit=64
    if attack=='outside':allowed=[]
    else:allowed=[p]
    if attack=='symlink':q=tmp_path/'target';q.write_bytes(b'fixture');p.unlink();p.symlink_to(q)
    elif attack=='ancestor':d=tmp_path/'linked';d.symlink_to(tmp_path,target_is_directory=True);p=d/'generated';allowed=[p]
    elif attack=='hardlink':os.link(p,tmp_path/'second')
    elif attack=='oversize':limit=2
    with pytest.raises((IntegrityError,OSError)):
        with authenticated_snapshot(p,sha256(b'fixture'),allowlist=allowed,limit=limit):pass


def test_restricted_tensor_roundtrip_or_old_version_rejection():
    stream=io.BytesIO();torch.save({'value':torch.tensor([1.,2.])},stream);data=stream.getvalue();s=AuthenticatedBytes(data,sha256(data))
    if tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2])<(2,10):
        with pytest.raises(LoadingBlocked,match='patched'):restricted_load(s)
    else:assert torch.equal(restricted_load(s)['value'],torch.tensor([1.,2.]))


def test_unverified_snapshot_never_parsed(monkeypatch):
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('parse before verification'))
    with pytest.raises(IntegrityError):restricted_load(AuthenticatedBytes(b'invalid','0'*64))


def test_unsafe_generated_object_never_executed(tmp_path):
    sentinel=tmp_path/'SHOULD_NOT_EXIST'
    class Unsafe:
        def __reduce__(self):return os.system,('touch '+str(sentinel),)
    stream=io.BytesIO();torch.save({'bad':Unsafe()},stream);data=stream.getvalue()
    with pytest.raises(LoadingBlocked):restricted_load(AuthenticatedBytes(data,sha256(data)))
    assert not sentinel.exists()
