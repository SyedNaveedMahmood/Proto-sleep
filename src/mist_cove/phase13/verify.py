"""Generated dependency proofs, exhaustive tables and negative controls."""
import json
import torch
from mist_cove.phase10.verification import hidden_change
from mist_cove.phase10.response_circuits import summarize_response
from .provenance import PUBLIC,isolated_rng,write_new
from .model_adequacy import WidthLedger
from .support_operator import LedgerOperator
from .resources import measured


def verify_generated():
    torch.set_num_threads(2);rows=[];comparisons=0
    with measured('generated_operator_correctness','ENGINEERING',tables=8,table_state_evaluations=262144):
        for capacity in ('S','L'):
            for routing in (False,True):
                with isolated_rng(13023):
                    model=WidthLedger(capacity,routing).eval();operator=LedgerOperator(model)
                    t=torch.arange(3000)/100
                    x=torch.stack((torch.sin(2*torch.pi*2.3*t),torch.sin(2*torch.pi*12.7*t)))[:,None]+.15*torch.randn(2,1,3000)
                    maximum=0
                    for i in range(2):
                        cache=operator.cache(x[i:i+1]);full=torch.ones(1,15,dtype=torch.bool)
                        for neutral in (0.,.01):
                            actual=operator.logits(cache,full,neutral)[0];torch.testing.assert_close(actual,cache.native,atol=1e-6,rtol=1e-6)
                            maximum=max(maximum,float((actual-cache.native).abs().max()))
                        table=operator.response_table(cache,int(cache.native.argmax()));summary=summarize_response(table)
                        rows.append({'capacity':capacity,'routing':routing,'example':i,'exact_all_rows':summary['exact_all_rows'],
                                     'BDD_nodes':summary['BDD_nonterminal_nodes'],'support_identity':operator.support_identity,'full_identity_max_abs':maximum})
                    cache=operator.cache(x[:1])
                    for mask_id in (0,1,21845,32767^128):
                        mask=((torch.tensor(mask_id)>>torch.arange(15))&1).bool()
                        for kind in ('random','dc','rms','boundary'):
                            changed=operator.cache(hidden_change(x[:1],mask,kind))
                            for neutral in (0.,.01):
                                torch.testing.assert_close(operator.logits(cache,mask[None],neutral),operator.logits(changed,mask[None],neutral),atol=1e-6,rtol=1e-6);comparisons+=1
                    # A deliberately unrestricted global path changes under a
                    # hidden DC intervention; the proper operator withholds it.
                    hidden=torch.zeros(15,dtype=torch.bool);changed=operator.cache(hidden_change(x[:1],hidden,'dc'))
                    if torch.equal(model.global_head(cache.global_features),model.global_head(changed.global_features)):
                        raise ValueError('broken-global negative control not detected')
        write_new(PUBLIC/'operator_correctness.json',{'status':'PASS','domain':'GENERATED_NO_REAL_EEG','tables':rows,
            'noninterference_comparisons':comparisons,'broken_global_controls_detected':4,'P3B_proof_reused':False,'atol':1e-6,'rtol':1e-6})
        print(json.dumps({'generated_tables':len(rows),'noninterference_comparisons':comparisons,'status':'PASS'}))
