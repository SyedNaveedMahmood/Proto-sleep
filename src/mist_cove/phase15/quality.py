"""Stage-conditioned fixed-detector quality flags, aggregate only."""
import json
from collections import defaultdict

import numpy as np

from .forensics import collect,_stratum
from .governance import OLD_PRIVATE,PUBLIC,LABEL,read,new,verify_plan_commit


FLAGS=('EEG Fpz-Cz|flat','EEG Fpz-Cz|step','EEG Fpz-Cz|extreme','EEG Fpz-Cz|clip',
       'EEG Pz-Oz|flat','EEG Pz-Oz|step','EEG Pz-Oz|extreme','EEG Pz-Oz|clip')


def _epoch_set(events,n):
    out=set()
    for event in events:
        lo=max(0,int(np.floor(event['start']/30)))
        hi=min(n-1,int(np.floor((event['end']-1e-10)/30)))
        out.update(range(lo,hi+1))
    return out


def run():
    verify_plan_commit();rows,_,_=collect()
    by_record=defaultdict(list)
    for r in rows:
        if r['truth']==4:by_record[r['recording']].append(r)
    grouped=defaultdict(list);persons=defaultdict(lambda:defaultdict(list))
    for rec,rr in by_record.items():
        item=read(OLD_PRIVATE/'ST_EVAL'/(rec+'_events.json'))
        detections={k:_epoch_set(item['events'].get(k,[]),item['n_epochs']) for k in FLAGS}
        for r in rr:
            row={'stratum':_stratum(r),'person':r['person'],
                 'flags':{k:r['epoch'] in detected for k,detected in detections.items()}}
            grouped[row['stratum']].append(row);persons[row['person']][row['stratum']].append(row)
    result={}
    for s,rr in grouped.items():
        result[s]={'n':len(rr),'by_channel_flag':{key:{'n':sum(r['flags'][key] for r in rr),
            'fraction':sum(r['flags'][key] for r in rr)/len(rr)} for key in FLAGS}}
    comparison={}
    for name in ('REM_to_W','REM_to_N1','REM_to_N2'):
        comparison[name]={}
        for key in FLAGS:
            deltas=[]
            for person,g in persons.items():
                if g[name] and g['correct_REM']:
                    deltas.append(np.mean([r['flags'][key] for r in g[name]])-
                                  np.mean([r['flags'][key] for r in g['correct_REM']]))
            comparison[name][key]={'people':len(deltas),'person_mean_error_minus_correct_REM':float(np.mean(deltas))}
    new(PUBLIC/'quality_results.json',{'status':LABEL,'fixed_phase14_event_flags':list(FLAGS),
          'ST_records':len(by_record),'scored_REM_epochs':sum(len(x) for x in by_record.values()),
          'by_REM_outcome':result,'within_person_contrasts':comparison,
          'interpretation':'automated quality flags only; high signal energy is not automatically artifact'})
    print(json.dumps({'ST_records':len(by_record),'REM':sum(len(x) for x in by_record.values())}))


if __name__=='__main__':run()
