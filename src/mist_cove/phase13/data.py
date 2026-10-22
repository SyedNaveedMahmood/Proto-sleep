"""Rebuild authorized physical samples; preserve identities in private storage."""
import io
import json
import os
import time
import hashlib
import numpy as np
from mist_transfer.preprocessing import epoch_targets
from mist_cove.phase11b.security import authenticated_snapshot, IntegrityError
from mist_cove.phase11b.evaluation import signal_quality
from .provenance import PUBLIC, PRIVATE, setup_source, write_new
from .sampling import frozen_selection
from .resources import measured


def key(c):
    return c['recording_id'],c['epoch']


def file_digest(path):
    from .seals import read_descriptor_bytes
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NOATIME)
    try:
        data = read_descriptor_bytes(fd,path,256*1024*1024)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


def prepare():
    with measured('source_cache','SHARED') as counts:
        selection = frozen_selection()
        _, access = setup_source()
        records = access.records('train')
        fit,dev = selection['training']['fit'],selection['training']['internal_dev']
        samples = fit+dev
        lookup = {key(c):i for i,c in enumerate(samples)}
        if len(lookup) != len(samples):raise IntegrityError('duplicate fit/development coordinates')
        annotation = selection['primary']+selection['secondary']
        wave = np.empty((len(samples),1,3000),np.float32)
        transitions = np.zeros(len(samples),bool)
        context = np.zeros((len(annotation),3,3000),np.float32)
        context_valid = np.zeros((len(annotation),3),bool)
        for record in records:
            chosen = [i for i,c in enumerate(samples) if c['recording_id']==record['recording_id']]
            panel = [i for i,c in enumerate(annotation) if c['recording_id']==record['recording_id']]
            needed = {samples[i]['epoch'] for i in chosen}
            for i in panel:
                for offset in (-1,0,1):
                    epoch=annotation[i]['epoch']+offset
                    if 0<=epoch<record['epoch_grid']['n_epochs']:needed.add(epoch)
            if not needed:continue
            needed=sorted(needed);values={}
            with access.reader(record) as reader:
                for start in range(0,len(needed),256):
                    indices=needed[start:start+256];_,w=access.read(record,reader,indices)
                    values.update({epoch:v[0] for epoch,v in zip(indices,w)})
            y,mask,*_=epoch_targets(record)
            for i in chosen:
                epoch=samples[i]['epoch'];wave[i,0]=values[epoch]
                if not mask[epoch] or y[epoch]!=samples[i]['stage']:raise IntegrityError('physical stage identity differs')
                transitions[i]=epoch>0 and mask[epoch-1] and y[epoch-1]!=y[epoch]
            for i in panel:
                for j,offset in enumerate((-1,0,1)):
                    epoch=annotation[i]['epoch']+offset
                    if epoch in values:context[i,j]=values[epoch];context_valid[i,j]=True
        norm=selection['normalization'];normalized=(wave-norm['mean_uV'])/norm['std_uV']
        truth=np.array([c['stage'] for c in samples],dtype=np.int64)
        subjects=np.array([c['subject'] for c in samples]);record_ids=np.array([c['recording_id'] for c in samples])
        np.savez_compressed(PRIVATE/'data.npz',x=normalized,wave_uV=wave,truth=truth,subjects=subjects,
                            records=record_ids,physical_epochs=np.array([c['epoch'] for c in samples]),
                            transition=transitions,quality=signal_quality(wave),n_fit=len(fit))
        np.savez_compressed(PRIVATE/'annotation_context.npz',wave_uV=context,valid=context_valid)
        receipt={'status':'PASS_TRAIN_ONLY','data_sha256':file_digest(PRIVATE/'data.npz'),
            'annotation_context_sha256':file_digest(PRIVATE/'annotation_context.npz'),
            'fit_examples':len(fit),'internal_dev_examples':len(dev),'primary':len(selection['primary']),
            'source_access':access.counts,'raw_VAL_reads':0,'excluded_source_reads':0,
            'normalization_fit_people':10,'normalization_historical_P3B_modified':False,
            'cache':'newly verified calibrated physical samples with original coordinates; not legacy trimmed NPZ',
            'context':'only immediately adjacent physical epochs in the same approved TRAIN recording'}
        write_new(PUBLIC/'source_scope_receipt.json',receipt)
        counts.update(access.counts)
        print(json.dumps({k:receipt[k] for k in ('status','fit_examples','internal_dev_examples','raw_VAL_reads')}))


def load_cache(name='data.npz'):
    receipt=json.loads((PUBLIC/'source_scope_receipt.json').read_text())
    if name not in ('data.npz','annotation_context.npz'):raise IntegrityError('explicit phase cache only')
    digest=receipt['data_sha256' if name=='data.npz' else 'annotation_context_sha256']
    path=PRIVATE/name
    with authenticated_snapshot(path,digest,allowlist=[path],limit=256*1024*1024) as snapshot:
        with np.load(io.BytesIO(snapshot.data),allow_pickle=False) as archive:
            return {k:archive[k] for k in archive.files}
