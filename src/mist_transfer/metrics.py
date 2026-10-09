"""Fixed-five-class metrics with explicit absent-class handling."""
from __future__ import annotations
import numpy as np


def confusion_metrics(matrix):
    cm=np.asarray(matrix,dtype=np.int64)
    if cm.shape!=(5,5) or np.any(cm<0) or cm.sum()==0:raise ValueError('nonempty five-class confusion required')
    tp=np.diag(cm);support=cm.sum(1);predicted=cm.sum(0);total=int(cm.sum())
    f1=np.divide(2*tp,support+predicted,out=np.zeros(5),where=support+predicted>0)
    recall=np.divide(tp,support,out=np.zeros(5),where=support>0)
    accuracy=float(tp.sum()/total);expected=float(np.dot(support.astype(float),predicted)/(total*total))
    return {'accuracy':accuracy,'macro_f1':float(f1.mean()),'per_stage_f1':f1.tolist(),
        'balanced_accuracy':float(recall[support>0].mean()),
        'five_class_balanced_accuracy_zero_absent':float(recall.mean()),
        'per_stage_recall':[float(v) if n else None for v,n in zip(recall,support)],
        'support':support.tolist(),'absent_truth_classes':np.flatnonzero(support==0).tolist(),
        'kappa':float((accuracy-expected)/(1-expected)) if 1-expected>1e-15 else None,
        'confusion_matrix':cm.tolist(),'n_epochs':total,
        'absent_class_policy':'fixed-five Macro-F1: undefined F1=0; balanced accuracy averages observed truth classes; undefined kappa/recall=null'}


def classification_metrics(truth,predicted,subjects,mask=None):
    y=np.asarray(truth);p=np.asarray(predicted);s=np.asarray(subjects)
    if y.ndim!=1 or p.shape!=y.shape or s.shape!=y.shape:raise ValueError('metric arrays disagree')
    m=np.asarray(mask,dtype=bool) if mask is not None else y>=0
    if m.shape!=y.shape or not np.isin(y[m],range(5)).all() or not np.isin(p[m],range(5)).all():
        raise ValueError('invalid scored labels/predictions')
    if not m.any():raise ValueError('no scored epochs; primary validation metric undefined')
    def matrix(a,b):return np.bincount(a.astype(int)*5+b.astype(int),minlength=25).reshape(5,5)
    result=confusion_metrics(matrix(y[m],p[m]));subject_metrics={}
    for sid in sorted(set(s.tolist())):
        selected=m&(s==sid)
        subject_metrics[str(sid)]=confusion_metrics(matrix(y[selected],p[selected])) if selected.any() else None
    valid={sid:met['macro_f1'] for sid,met in subject_metrics.items() if met is not None}
    if len(valid)!=len(subject_metrics):raise ValueError('validation subject has no scored epochs; do not silently drop participant')
    result.update(subject_metrics=subject_metrics,subject_macro_f1=valid,
                  mean_subject_macro_f1=float(np.mean(list(valid.values()))),unscored_epochs=int((~m).sum()))
    return result
