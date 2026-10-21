"""Full physical-grid diagnostics, with VAL embargo and private arrays."""
import json
import time
import resource
import numpy as np
import torch
from mist_transfer.preprocessing import epoch_targets
from mist_cove.phase10.source_adapter import SourceAdapter
from .campaign import PUBLIC,PRIVATE,setup,write_new
from .historical import named_state_digest,STATE_SHA
from .evaluation import public_metrics,signal_quality,quality_associations,QUALITY_NAMES

def run(role='train'):
    if role not in ('train','val'):raise ValueError('unauthorized role')
    if role=='val':
        decision=json.loads((PUBLIC/'model_development_decision.json').read_text())
        if not decision['choices_frozen_before_VAL']:raise ValueError('VAL embargo')
    destination=PRIVATE/(role+'_predictions.npz')
    if destination.exists():raise ValueError('no silent repetition')
    started=time.perf_counter();model,access=setup();adapter=SourceAdapter(model);all_z=[];all_y=[];all_s=[];all_r=[];all_q=[];all_b=[]
    previous=json.loads((PUBLIC/'audit_execution.json').read_text())['resources']
    seconds=previous['wall_seconds'];forwards=previous['native_full_example_forwards']+previous['feature_cache_example_forwards'];native_seconds=0.
    for ordinal,record in enumerate(access.records(role)):
        y,mask,*_=epoch_targets(record);n=record['epoch_grid']['n_epochs'];z=[];q=[]
        with access.reader(record) as reader:
            for start in range(0,n,128):
                count=min(128,n-start)
                if time.perf_counter()-started+seconds>7100 or forwards+count>250000:raise TimeoutError('scientific resource ceiling')
                x,w=access.read(record,reader,np.arange(start,start+count));t=time.perf_counter()
                with torch.no_grad():z.append(adapter.native(torch.from_numpy(x)).numpy())
                native_seconds+=time.perf_counter()-t;forwards+=count;q.append(signal_quality(w))
        all_z.append(np.concatenate(z));all_y.append(np.where(mask,y,-1));all_s.extend([record['subject_id']]*n);all_r.extend([record['recording_id']]*n);all_q.append(np.concatenate(q))
        boundary=np.zeros(n,bool);boundary[:2]=True;boundary[-2:]=True;all_b.append(boundary)
        print(json.dumps({'role':role,'recordings_complete':ordinal+1,'physical_epochs':sum(len(x) for x in all_y),'wall_seconds':time.perf_counter()-started}),flush=True)
    z=np.concatenate(all_z);y=np.concatenate(all_y);s=np.asarray(all_s);r=np.asarray(all_r);q=np.concatenate(all_q);b=np.concatenate(all_b)
    expected={'train':(79984,79025),'val':(21796,21789)}[role]
    if (len(y),int((y>=0).sum()))!=expected:raise ValueError('frozen physical/scored counts differ')
    np.savez_compressed(destination,logits=z,truth=y,subjects=s,records=r,quality=q,boundary=b)
    result={'role':role,'exposure':'HISTORICALLY_TRAIN_FITTED_IN_SAMPLE' if role=='train' else 'REPEATEDLY_REUSED_DEVELOPMENT_VAL',
            'metrics':public_metrics(y,z,s,r),'quality_associations':quality_associations(y,z,s,q,b),
            'quality_names':QUALITY_NAMES,'source_access':access.counts,'wall_seconds':time.perf_counter()-started,
            'native_seconds':native_seconds,'native_full_example_forwards':len(y),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'state_unchanged':named_state_digest(model.state_dict())==STATE_SHA,'forbidden_sources_accessed':False}
    write_new(PUBLIC/(role+'_descriptive_metrics.json'),result)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('role',choices=['train','val']);run(p.parse_args().role)
