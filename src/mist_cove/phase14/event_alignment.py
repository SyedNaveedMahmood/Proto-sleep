"""Proxy concordance and fair matched-interval controls, never clinical precision."""
import numpy as np
from mist_cove.phase13.annotation_metrics import mask_intervals,overlap_metrics,controls
from .event_schema import event_intervals,shifted_intervals


def matched_candidates(wave,mask,greedy,gradient_ranking,seed):
    if mask is None:return None
    k=int(mask).bit_count();x=np.asarray(wave).reshape(15,200);rms=x.std(1)
    power=abs(np.fft.rfft(x-x.mean(1,keepdims=True),axis=1))**2;f=np.fft.rfftfreq(200,.01)
    def top(values):return sum(1<<int(i) for i in np.argsort(-values,kind='stable')[:k])
    result={'exact_positive':int(mask),'greedy_own_budget':int(greedy),'RMS_matched':top(rms),'gradient_matched':sum(1<<int(i) for i in gradient_ranking[:k]),
            'random_matched':controls(mask,seed)['length_matched_mask'],'position_matched':controls(mask,seed)['position_matched_mask']}
    # Rank vector must assign importance by physical atom, not inverse permutation.
    result['gradient_matched']=sum(1<<int(i) for i in gradient_ranking[:k])
    for name,band in [('sigma',(12,15)),('delta',(.5,2)),('alpha',(8,13)),('theta',(4,7))]:
        ratio=power[:,(f>=band[0])&(f<=band[1])].sum(1)/power[:,(f>=.5)&(f<=45)].sum(1).clip(1e-12)
        result[name+'_matched']=top(ratio)
    return result


def align(events,epoch,candidates,permuted_events=None):
    if candidates is None:return {'status':'NO_POSITIVE_SUPPORT','proxies':{}}
    result={};positive=candidates['exact_positive'];k=positive.bit_count()
    for key,values in events.items():
        if key.endswith('|spindle_strict') or key.endswith('|slow_wave_relaxed') or key.endswith(('|spindle','|slow_wave','|alpha','|theta','|HF','|excursion','|flat','|clip','|extreme','|step','|low','|high','|sigma','|delta')):
            intervals=event_intervals(values,epoch);methods={}
            for name,mask in candidates.items():methods[name]={str(t):overlap_metrics(mask_intervals(mask),intervals,t) for t in (0,.1,.25)}
            shifted=shifted_intervals(intervals,10);methods['proxy_shift10s']={'0':overlap_metrics(mask_intervals(positive),shifted)}
            if permuted_events is not None and key in permuted_events:
                other=permuted_events.get(key,[]);methods['cross_channel_permuted_same_recording']={'0':overlap_metrics(mask_intervals(positive),other)}
            result[key]={'methods':methods,'event_count':len(intervals),'event_eligible':bool(intervals),'support_atoms':k,'whole_epoch_degenerate':k==15,
                         'greedy_budget_matches':candidates['greedy_own_budget'].bit_count()==k,'human_event_accuracy':False}
    return {'status':'COMPUTED','proxies':result}


def interval_concordance(first,second):
    # Greedy maximum-IoU one-to-one matching over sparse overlapping pairs.
    a=sorted(first,key=lambda r:r['start']);b=sorted(second,key=lambda r:r['start']);pairs=[];j0=0
    for i,x in enumerate(a):
        while j0<len(b) and b[j0]['end']<=x['start']:j0+=1
        j=j0
        while j<len(b) and b[j]['start']<x['end']:
            y=b[j];iou=max(0,min(x['end'],y['end'])-max(x['start'],y['start']))/(max(x['end'],y['end'])-min(x['start'],y['start']))
            if iou>=.2:pairs.append((iou,i,j))
            j+=1
    useda=set();usedb=set();matches=[]
    for iou,i,j in sorted(pairs,key=lambda r:(-r[0],r[1],r[2])):
        if i in useda or j in usedb:continue
        useda.add(i);usedb.add(j);matches.append(iou)
    n=len(matches)
    return {'first_events':len(a),'second_events':len(b),'matched':n,'F1':2*n/(len(a)+len(b)) if a or b else None,
            'mean_matched_IoU':float(np.mean(matches)) if matches else None,'method':'greedy largesteligibleIoU>=.2; one-to-one; signal-viewagreement only'}
