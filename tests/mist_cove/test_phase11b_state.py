from contextlib import contextmanager
import copy
import pytest
import torch
from mist_cove.phase10.mock_models import make_mock
from mist_evidence.model import ModelConfig
from mist_cove.phase11b import historical
from mist_cove.phase11b.security import IntegrityError

@pytest.fixture
def generated_payload(monkeypatch):
    model=make_mock().network
    payload={'state_dict':copy.deepcopy(model.state_dict()),'model_config':model.cfg.dictionary(),
             'bank':{'seed':123,'config':ModelConfig(radius=0,use_crf=False).dictionary(),'waveforms':[b.waveforms.detach().clone() for b in model.banks],'amplitude_stats':(0.,1.)},
             'signature':'generated','best_epoch':28,'metrics':{}}
    @contextmanager
    def snapshot(*args,**kwargs):yield None
    monkeypatch.setattr(historical,'verify_source',lambda:{})
    monkeypatch.setattr(historical,'authenticated_receipts',lambda:({},{}))
    monkeypatch.setattr(historical,'authenticated_snapshot',snapshot)
    monkeypatch.setattr(historical,'restricted_load',lambda _:payload)
    monkeypatch.setattr(historical,'STATE_SHA',historical.named_state_digest(payload['state_dict']))
    return payload

def test_generated_exact_schema_and_no_state_mutation(generated_payload):
    model,*_,receipt=historical.authenticate_model()
    assert receipt['parameters']==34197 and receipt['state_unchanged']

@pytest.mark.parametrize('mutation',['extra_key','dtype','nonfinite','digest','configuration','bank','epoch','named_key'])
def test_malformed_state_rejected(generated_payload,monkeypatch,mutation):
    p=generated_payload;key=next(k for k,v in p['state_dict'].items() if v.numel()>2)
    if mutation=='extra_key':p['unexpected']=True
    elif mutation=='dtype':p['state_dict'][key]=p['state_dict'][key].double()
    elif mutation=='nonfinite':p['state_dict'][key].reshape(-1)[0]=float('nan')
    elif mutation=='digest':p['state_dict'][key].reshape(-1)[0]+=1
    elif mutation=='configuration':p['model_config']['radius']=1
    elif mutation=='bank':p['bank']['waveforms'][0][0,0]+=1
    elif mutation=='epoch':p['best_epoch']=27
    elif mutation=='named_key':
        p['state_dict']['wrong_name']=p['state_dict'].pop(key)
        monkeypatch.setattr(historical,'STATE_SHA',historical.named_state_digest(p['state_dict']))
    with pytest.raises(IntegrityError):historical.authenticate_model()
