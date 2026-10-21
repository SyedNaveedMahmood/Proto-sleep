"""Bounded authenticated snapshots and patched, default-only CPU deserialization."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import io
import os
from pathlib import Path
import re
import stat
import zipfile


class IntegrityError(ValueError):
    pass


class LoadingBlocked(RuntimeError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stamp(s):
    return s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns


@dataclass(frozen=True)
class AuthenticatedBytes:
    data: bytes
    digest: str


@contextmanager
def authenticated_snapshot(path, expected, *, allowlist, limit=64*1024*1024):
    """No discovery. Pin every parent; compare original path and held FD identity.

    Caller must obtain expected digest from a separate trusted historical chain.
    Generic transport accepts only an explicit per-operation path allowlist.
    """
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts or str(path) not in {str(p) for p in allowlist}:
        raise IntegrityError('path outside explicit allowlist')
    if not re.fullmatch('[0-9a-f]{64}',expected):
        raise IntegrityError('independent SHA256 required')
    dirs=[];links=[];fd=None
    try:
        dirs.append(os.open('/',os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW))
        for name in path.parts[1:-1]:
            parent=dirs[-1];before=os.stat(name,dir_fd=parent,follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):raise IntegrityError('symlink/non-directory ancestor')
            held=os.open(name,os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);dirs.append(held)
            after=os.fstat(held)
            if (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino):raise IntegrityError('ancestor race')
            links.append((parent,name,held))
        before=os.stat(path.name,dir_fd=dirs[-1],follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:raise IntegrityError('nonregular/symlink/hardlinked artifact')
        if not 0<before.st_size<=limit:raise IntegrityError('artifact size ceiling')
        fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_NOATIME,dir_fd=dirs[-1])
        original=stamp(os.fstat(fd))
        if stamp(before)!=original:raise IntegrityError('artifact open race')
        def stable():
            for parent,name,held in links:
                named=os.stat(name,dir_fd=parent,follow_symlinks=False);actual=os.fstat(held)
                if not stat.S_ISDIR(named.st_mode) or (named.st_dev,named.st_ino)!=(actual.st_dev,actual.st_ino):raise IntegrityError('ancestor substituted')
            if stamp(os.fstat(fd))!=original or stamp(os.stat(path.name,dir_fd=dirs[-1],follow_symlinks=False))!=original:
                raise IntegrityError('artifact identity or metadata changed')
        stable();blocks=[];count=0
        while count<=limit:
            block=os.pread(fd,min(65536,limit+1-count),count)
            if not block:break
            blocks.append(block);count+=len(block)
        stable();data=b''.join(blocks)
        if len(data)!=before.st_size or len(data)>limit:raise IntegrityError('artifact read size changed')
        digest=sha256(data)
        if not hmac.compare_digest(digest,expected):raise IntegrityError('historical byte digest mismatch')
        yield AuthenticatedBytes(data,digest)
        stable()
    finally:
        if fd is not None:os.close(fd)
        for held in reversed(dirs):os.close(held)


def restricted_load(snapshot):
    """No custom globals; no unsafe fallback. Only authenticated immutable bytes."""
    import torch
    if type(snapshot) is not AuthenticatedBytes or type(snapshot.data) is not bytes or sha256(snapshot.data)!=snapshot.digest:
        raise IntegrityError('authenticated immutable snapshot required')
    version=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:3])
    if version<(2,10,0):raise LoadingBlocked('patched PyTorch >=2.10.0 required')
    if torch.serialization.get_safe_globals():raise LoadingBlocked('custom serialization globals present')
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.data)) as archive:
            entries=archive.infolist();names=[e.filename for e in entries]
            if not 1<=len(entries)<=512 or len(set(names))!=len(names):raise LoadingBlocked('archive member count/duplicates')
            if sum(e.file_size for e in entries)>64*1024*1024:raise LoadingBlocked('expanded archive ceiling')
            for entry in entries:
                parts=entry.filename.split('/')
                if entry.filename.startswith('/') or any(p in ('','..','.') for p in parts) or '\\' in entry.filename:
                    raise LoadingBlocked('archive traversal')
                if entry.compress_type!=zipfile.ZIP_STORED or entry.flag_bits&1:raise LoadingBlocked('unreviewed compressed/encrypted archive')
        if torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(snapshot.data)):
            raise LoadingBlocked('payload requires non-default serialization globals')
        payload=torch.load(io.BytesIO(snapshot.data),map_location='cpu',weights_only=True)
    except LoadingBlocked:raise
    except Exception as error:
        # Never export exception strings containing private state or paths.
        raise LoadingBlocked('restricted parsing rejected: '+type(error).__name__) from None
    return payload
