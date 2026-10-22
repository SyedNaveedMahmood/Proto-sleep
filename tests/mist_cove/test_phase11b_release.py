import pytest
from mist_cove.phase11b.release import validate_public_text,validate_publication_path,validate_task_ledger,validate_resource_receipt

@pytest.mark.parametrize('path',['mist_transfer_runs/private.json','reports/weights.pt','../escape.json','/absolute/report.json'])
def test_private_publication_paths_rejected(path):
    with pytest.raises(ValueError):validate_publication_path(path)

def test_identifier_path_guard_and_safe_aggregate():
    for text in ['source /home/example/file','SC9999E0','subject GENERATED_PRIVATE_ID']:
        with pytest.raises(ValueError):validate_public_text(text,['GENERATED_PRIVATE_ID'])
    validate_public_text('15 TRAIN participants; aggregate F1 .7; seed123')

def test_status_cannot_claim_completion_without_evidence():
    ledger={'work_packages':[{'id':'WP'+str(i).zfill(2),'status':'COMPLETE','artifacts':['aggregate.json'],'block_reason':None} for i in range(1,37)]}
    validate_task_ledger(ledger,final=True);ledger['work_packages'][0]['artifacts']=[]
    with pytest.raises(ValueError):validate_task_ledger(ledger,final=True)

def test_resource_overrun_not_rounded_to_pass():
    receipt={'core_new_neural_fits':9,'scientific_optimizer_updates':2700,'cached_head_queries':50000000,'combined_full_and_cache_examples':250000,'total_scientific_CPU_wall_seconds':7200,'GPU_seconds':0}
    validate_resource_receipt(receipt);receipt['total_scientific_CPU_wall_seconds']+=.001
    with pytest.raises(ValueError):validate_resource_receipt(receipt)
