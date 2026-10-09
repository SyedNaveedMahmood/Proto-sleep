"""TRAIN-only statistics/anchors and bounded lazy source recording views."""
from __future__ import annotations
import math
import time
import numpy as np
import pyedflib
from pathlib import Path
from mist_evidence.data import Recording, build_anchors
from mist_evidence.model import ModelConfig
from .preprocessing import iter_recording, SourceProofs, signal_contract, epoch_targets
from .provenance import fingerprint


def check_development(records,role=None):
    if not records or any(r.get('split') not in ({role} if role else {'train','val'}) for r in records):
        raise ValueError('development role required before source opening')


def fit_normalization(records):
    check_development(records,'train')
    n=0;mean=0.;m2=0.;logs=[];epochs=0
    for r in records:
        for batch in iter_recording(r,include_labels=False):
            x=batch['x'].astype(np.float64);k=x.size;mu=float(x.mean());local=float(((x-mu)**2).sum())
            delta=mu-mean;combined=n+k;m2+=local+delta*delta*n*k/combined;mean+=delta*k/combined;n=combined
            logs.extend(np.log(x[:,0].std(axis=1).clip(min=1e-6)).tolist());epochs+=len(x)
    std=math.sqrt(m2/n)
    if not np.isfinite([mean,std]).all() or std<=1e-8:raise ValueError('invalid source-TRAIN normalization')
    log=np.asarray(logs)-math.log(std)
    return {'version':'source-train-global-uV-zscore-v1','mean_uV':mean,'std_uV':std,
        'sample_count':n,'physical_epoch_count':epochs,'fit_subjects':sorted({r['subject_id'] for r in records}),
        'source_hashes':{r['recording_id']:r['source_sha256'] for r in records},
        'label_access':False,'scored_epoch_selection':False,'fit_role':'train',
        'amplitude_stats':[float(log.mean()),max(.1,float(log.std()))]}


class LazyEpochArray:
    """Array slice interface consumed by the proven v1 trainer; no all-night cache."""
    def __init__(self,record,indices,normalization):
        check_development([record]);self.record=record;self.indices=indices;self.normalization=normalization
        self.shape=(len(indices),1,3000);self.dtype=np.dtype('float32');self.peak_bytes=0;self.read_epochs=0
        self.io_seconds=0.;self.read_batches=0
        proofs=SourceProofs()
        for path,sha,saved in [('path','source_sha256','source_stat_proof'),('annotation_source','annotation_sha256','annotation_stat_proof')]:
            proofs.checked[str(Path(record[path]).resolve())]=record[saved];proofs.verify(record[path],record[sha])
        grid,contract=signal_contract({**record,'channels':[record['signal_contract']['calibration']],
            'start_datetime':record['signal_contract']['start_datetime'],'duration_seconds':record['signal_contract']['duration_seconds'],
            'n_epochs':record['epoch_grid']['n_epochs']})
        if grid!=record['epoch_grid'] or contract!=record['signal_contract']:raise ValueError('source contract changed')
        self.proofs=proofs

    def __len__(self):return len(self.indices)

    def __getitem__(self,selection):
        started=time.perf_counter()
        chosen=self.indices[selection];single=np.ndim(chosen)==0;chosen=np.atleast_1d(chosen)
        wave=np.empty((len(chosen),1,3000),dtype=np.float32)
        if len(chosen):
            r=self.record;self.proofs.verify(r['path'],r['source_sha256'])
            with pyedflib.EdfReader(r['path']) as reader:
                cuts=np.r_[0,np.flatnonzero(np.diff(chosen)!=1)+1,len(chosen)]
                for a,b in zip(cuts[:-1],cuts[1:]):
                    data=reader.readSignal(r['signal_contract']['channel_index'],start=int(chosen[a])*3000,n=int(b-a)*3000)
                    if len(data)!=(b-a)*3000 or not np.isfinite(data).all():raise ValueError(r['recording_id']+': invalid streamed waveforms')
                    wave[a:b]=data.reshape(-1,1,3000)
            self.proofs.postflight()
        wave=(wave-self.normalization['mean_uV'])/self.normalization['std_uV']
        if not np.isfinite(wave).all():raise ValueError('nonfinite normalized source')
        self.peak_bytes=max(self.peak_bytes,wave.nbytes);self.read_epochs+=len(wave)
        self.read_batches+=1;self.io_seconds+=time.perf_counter()-started
        return wave[0] if single else wave


class SourceRecording:
    def __init__(self,record,normalization,smoke=False):
        check_development([record]);n=record['epoch_grid']['n_epochs']
        indices=np.arange(n,dtype=np.int64)
        if smoke:
            # Four four-epoch windows at 20/40/60/80% of physical duration.
            # Fixed geometry before labels; all nights and people remain included.
            starts=[min(max(0,n-4),max(0,int(f*n))) for f in [.2,.4,.6,.8]]
            indices=np.unique(np.concatenate([np.arange(i,i+min(4,n)) for i in starts]))
        y,mask,_,_,_=epoch_targets(record)
        self.descriptor=record;self.indices=indices;self.y=y[indices];self.scoring_mask=mask[indices]
        self.x=LazyEpochArray(record,indices,normalization);self.path=record['path'];self.subject=record['subject_id']
        self.file_hash=record['source_sha256'];self.indices_known=True;self.continuity_assumed=False
        self.split=record['split'];self.recording_id=record['recording_id']

    def segments(self):
        # Supervised cores only; never bridge either an unscored position or a
        # geometry-subsampled gap. The full inference view stays separate/intact.
        valid=np.flatnonzero(self.scoring_mask)
        if not len(valid):return []
        cuts=np.r_[0,np.flatnonzero((np.diff(valid)!=1)|(np.diff(self.indices[valid])!=1))+1,len(valid)]
        return [(int(valid[a]),int(valid[b-1]+1)) for a,b in zip(cuts[:-1],cuts[1:])]


def training_bank(records,normalization,seed=123):
    check_development(records,'train');rng=np.random.default_rng(seed);candidates=[]
    for r in records:
        source=SourceRecording(r,normalization);chosen=[]
        for stage in range(5):
            pool=np.flatnonzero(source.y==stage)
            if len(pool):chosen.extend(rng.choice(pool,min(4,len(pool)),replace=False).tolist())
        indices=np.array(sorted(chosen),dtype=np.int64)
        if not len(indices):raise ValueError('TRAIN recording has no scored anchor candidates')
        x=source.x[indices]
        candidates.append(Recording(r['path'],r['subject_id'],x,source.y[indices],source.indices[indices],
                                    r['source_sha256'],False,True,True))
    cfg=ModelConfig(radius=0,use_crf=False)
    bank=build_anchors(candidates,cfg,seed,per_subject=16,method='medoid')
    bank['amplitude_stats']=tuple(normalization['amplitude_stats'])
    bank['normalization_fingerprint']=fingerprint(normalization)
    bank['amplitude_statistics_fit']='all complete physical source TRAIN epochs; normalized using TRAIN-only global statistics'
    lookup={r['path']:r for r in records}
    for metadata in bank['metadata']:
        for anchor in metadata:
            row=lookup[anchor['source_path']]
            if row['split']!='train':raise ValueError('non-TRAIN waveform anchor')
            anchor.update(recording_id=row['recording_id'],night=row['night'],study='SC',split='train',
                          absolute_start_sample=anchor['original_epoch']*3000+anchor['start_sample'],
                          annotation_sha256=row['annotation_sha256'],units='source-TRAIN zscore of calibrated uV')
    return bank


def validate_bank(bank,records,normalization,verify_waveforms=False):
    """Reject non-TRAIN anchors; optionally compare each medoid with its source."""
    check_development(records,'train');lookup={r['path']:r for r in records}
    if bank['training_subjects']!=sorted({r['subject_id'] for r in records}) or bank['normalization_fingerprint']!=fingerprint(normalization):
        raise ValueError('anchor TRAIN/normalization provenance changed')
    if len(bank['waveforms'])!=len(bank['metadata']) or len(bank['waveforms'])!=len(bank['config']['scales']):
        raise ValueError('anchor bank dimensions disagree')
    sources={}
    for length,waves,metadata in zip(bank['config']['scales'],bank['waveforms'],bank['metadata']):
        if tuple(waves.shape)!=(bank['config']['prototypes_per_scale'],length) or len(metadata)!=len(waves) or not np.isfinite(waves.numpy()).all():
            raise ValueError('invalid waveform anchor dimensions/values')
        for wave,m in zip(waves,metadata):
            r=lookup.get(m['source_path']);ep=m['original_epoch'];a=m['start_sample']
            if r is None or m['split']!='train' or m['subject']!=r['subject_id'] or m['recording_id']!=r['recording_id'] or m['source_sha256']!=r['source_sha256'] or m['annotation_sha256']!=r['annotation_sha256']:
                raise ValueError('non-TRAIN or changed anchor source provenance')
            if not isinstance(ep,int) or not 0<=ep<r['epoch_grid']['n_epochs'] or a<0 or a+length>3000 or m['length_samples']!=length or m['absolute_start_sample']!=ep*3000+a or m['clinical_event_label'] is not None:
                raise ValueError('invalid anchor original epoch/sample provenance')
            if verify_waveforms:
                if r['path'] not in sources:sources[r['path']]=SourceRecording(r,normalization)
                source=sources[r['path']]
                if not source.scoring_mask[ep] or source.y[ep]!=m['stage_at_source_epoch']:raise ValueError('anchor annotation provenance mismatch')
                if not np.array_equal(wave.numpy(),source.x[ep][0,a:a+length]):raise ValueError('anchor is not a real source waveform')
