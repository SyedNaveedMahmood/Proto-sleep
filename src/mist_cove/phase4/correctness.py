"""Generated-data exact resume and bounded CPU/CUDA checks; no EEG loader."""
from pathlib import Path
import torch

from mist_cove.phase2.synthetic_runtime import mock_recordings,assert_exact
from mist_cove.phase2.training import configure_precision,sealed_train
from mist_evidence.runtime import TrainConfig,seed_all,load_trusted
from .source import Adapter


def exact_resume(output,method='M2',device='cpu'):
    configure_precision(device)
    train,val=mock_recordings();cfg=TrainConfig(epochs=2,patience=12,core_epochs=4,encode_batch=4,seed=123)
    def fresh():
        seed_all(123,True);return Adapter(method,{'mean_uV':0.,'std_uV':1.})
    args={'provenance':{'scope':'SYNTHETIC_PHASE_IV_RESUME','method':method},'device':device}
    output=Path(output)
    sealed_train(fresh(),[train],[val],cfg,output/'full',**args)
    try:sealed_train(fresh(),[train],[val],cfg,output/'resumed',interrupt_after_epoch=1,**args)
    except InterruptedError:pass
    else:raise AssertionError('interruption missing')
    sealed_train(fresh(),[train],[val],cfg,output/'resumed',resume=True,**args)
    a=load_trusted(output/'full/last.pt');b=load_trusted(output/'resumed/last.pt')
    for k in ['state_dict','optimizer','rng','epoch','best_score','best_epoch','stale']:assert_exact(a[k],b[k],k)
    semantic=['epoch','train_nll','val_nll','val_subject_macro_f1','val_pooled_macro_f1','val_accuracy','best_epoch','stale','optimizer_steps','last_grad_norm']
    assert_exact([{k:r[k] for k in semantic} for r in a['history']],[{k:r[k] for k in semantic} for r in b['history']])
    return {'method':method,'device':device,'model_optimizer_rng_history_exact':True,'epochs':2,'optimizer_updates':sum(r['optimizer_steps'] for r in a['history'])}
