"""Exact-root capability: role checks precede stat/open; no symlink traversal."""
import os
import stat
import hashlib
from contextlib import contextmanager
from pathlib import PurePosixPath
from .common import DATA_ROOT,IntegrityError
from .common import stamp

ROOT_FILES={'RECORDS','RECORDS-v1','SHA256SUMS.txt','SC-subjects.xls','ST-subjects.xls'}


class RootAccess:
    def __init__(self,root=DATA_ROOT,allowed=None):
        self.root=root;self.allowed=dict(allowed or {});self.counts={'metadata_reads':0,'source_hash_reads':0,'header_reads':0,'waveform_reads':0,'hypnogram_reads':0,'bytes_hashed':0,'forbidden_opened':0}
        fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
        try:
            for part in PurePosixPath(root).parts[1:]:
                next_fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=next_fd
            self.rootfd=os.dup(fd)
        finally:os.close(fd)
    def close(self):os.close(self.rootfd)
    @contextmanager
    def file(self,relative,kind='metadata'):
        parts=PurePosixPath(relative).parts
        if not parts or relative.startswith('/') or any(p in ('..','.') for p in parts):raise IntegrityError('unsafe relative path')
        if kind=='metadata':
            if relative not in ROOT_FILES:raise IntegrityError('unapproved root metadata')
        else:
            entry=self.allowed.get(relative)
            if entry is None or entry['role'] not in ('SC_TRAIN','ST_EVAL'):raise IntegrityError('role/allowlist denied before stat or open')
            if parts[0] not in ('sleep-cassette','sleep-telemetry') or len(parts)!=2:raise IntegrityError('unapproved source layout')
        fd=os.dup(self.rootfd)
        try:
            for part in parts[:-1]:
                new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
            parent=fd;fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NOATIME,dir_fd=parent);os.close(parent)
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or not 0<before.st_size<=128*1024*1024:raise IntegrityError('bounded single-link regular source required')
            yield fd,before
            if stamp(before)!=stamp(os.fstat(fd)):raise IntegrityError('source changed during read')
            # Check identity at the opened directory chain, plus current full path.
            if stamp(before)!=stamp(os.stat(self.root+'/'+relative,follow_symlinks=False)):raise IntegrityError('source path substitution')
        finally:os.close(fd)
    def bytes(self,relative,kind='metadata',limit=2*1024*1024,expected=None):
        with self.file(relative,kind) as (fd,s):
            if s.st_size>limit:raise IntegrityError('artifact exceeds bounded size')
            blocks=[];offset=0
            while offset<s.st_size:
                block=os.pread(fd,min(65536,s.st_size-offset),offset)
                if not block:raise IntegrityError('truncated source')
                blocks.append(block);offset+=len(block)
            data=b''.join(blocks)
        if expected and hashlib.sha256(data).hexdigest()!=expected:raise IntegrityError('authenticated source bytes differ')
        self.counts['metadata_reads' if kind=='metadata' else 'source_hash_reads']+=1
        return data
    def hash(self,relative,expected):
        h=hashlib.sha256()
        with self.file(relative,'source') as (fd,s):
            offset=0
            while offset<s.st_size:
                block=os.pread(fd,min(1048576,s.st_size-offset),offset)
                if not block:raise IntegrityError('truncated source')
                h.update(block);offset+=len(block)
            size=s.st_size
        self.counts['source_hash_reads']+=1;self.counts['bytes_hashed']+=size
        if h.hexdigest()!=expected:raise IntegrityError('source checksum mismatch')
        return {'sha256':h.hexdigest(),'bytes':size}
