"""Metadata-only guard tests: no EEG or locked-cohort access."""
import json
import runpy
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
ns = runpy.run_path(str(ROOT / "scripts/validate_mist_cove_design.py"), run_name="design_test_import")
verify_contract = ns["verify_contract"]
config = json.loads((ROOT / "configs/mist_cove_design_v0.json").read_text())
matrix = json.loads((ROOT / "experiments/mist_cove/ablation_matrix_v0.json").read_text())

def test_design_validation():
    assert verify_contract(config, matrix)["data_accessed"] is False

@pytest.mark.parametrize("role", ["source_train_eeg","source_val_eeg","reserved_sc","sleep_telemetry","shhs","training"])
def test_forbidden_access_rejected(role):
    c = json.loads(json.dumps(config))
    c["authorized_access"][role] = True
    with pytest.raises(ValueError):
        verify_contract(c, matrix)

def test_reserved_split_change_rejected():
    c = json.loads(json.dumps(config))
    c["split"]["reserved_subjects"] = 54
    with pytest.raises(ValueError):
        verify_contract(c, matrix)

def test_gate_escalation_rejected():
    c = json.loads(json.dumps(config))
    c["gates"][1]["status"] = "AUTHORIZED"
    with pytest.raises(ValueError):
        verify_contract(c, matrix)

def test_external_proof_cannot_be_claimed():
    m = json.loads(json.dumps(matrix))
    m["cohorts"]["INDEPENDENT_EXTERNAL"] = "VERIFIED"
    with pytest.raises(ValueError):
        verify_contract(config, m)

def test_missing_ablation_rejected():
    m = json.loads(json.dumps(matrix))
    m["experiments"].pop()
    with pytest.raises(ValueError):
        verify_contract(config, m)
