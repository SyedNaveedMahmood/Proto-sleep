"""Public-only Phase XV release identity and scientific-status regression."""
from mist_cove.phase15.governance import PUBLIC, read
from mist_cove.phase15.release import verify


def test_public_release_manifest_is_exact():
    assert verify() >= 40


def test_negative_gates_and_exposure_cannot_be_promoted():
    gates=read(PUBLIC/'scientific_gates.json')
    assert gates['status']=='EXPLORATORY_REUSED_ST'
    assert gates['Phase_XIV_ST_exact_minus_sigma']=='SCIENTIFIC_FAIL_UNCHANGED'
    assert gates['human_event_validity']=='NOT_TESTED'
    assert gates['new_independent_ST_confirmation']=='NOT_TESTED'


def test_public_receipts_keep_zero_training_and_approved_roles():
    usage=read(PUBLIC/'resource_receipts.json')
    access=read(PUBLIC/'data_access_receipt.json')
    assert usage['new_neural_fits']==usage['new_optimizer_updates']==0
    assert usage['new_cached_head_queries_attempted_plus_successful']<=38_000_000
    assert access['new_ST_PSG_waveform_reads']==44
    assert access['new_raw_SC_VAL_reads']==access['new_SC_RESERVED_reads']==0
