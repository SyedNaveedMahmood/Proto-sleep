"""Metadata-only MIST-COVE design verifier. No source data imports or I/O."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mist_cove_design_v0.json"
MATRIX = ROOT / "experiments/mist_cove/ablation_matrix_v0.json"
EXPECTED_SPLIT = "3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84"
EXPECTED_SHA = "1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5"
RESTRICTED = ("source_train_eeg", "source_val_eeg", "reserved_sc", "sleep_telemetry", "shhs", "training")


def verify_contract(config: dict, matrix: dict) -> dict:
    """Validate explicit design-only gates without resolving any dataset path."""
    if config.get("schema_version") != "mist-cove-design-v0":
        raise ValueError("unknown design version")
    if config.get("status") != "DESIGN_ONLY_NOT_AUTHORIZED_FOR_DATA_ACCESS_OR_TRAINING":
        raise ValueError("unsafe execution status")
    allowed = config.get("authorized_access")
    if not isinstance(allowed, dict) or allowed.get("design_metadata_only") is not True:
        raise ValueError("design-only authorization missing")
    if any(allowed.get(k) is not False for k in RESTRICTED):
        raise ValueError("data or training permissions must be explicitly false")
    split = config.get("split", {})
    if split.get("fingerprint") != EXPECTED_SPLIT or split.get("file_sha256") != EXPECTED_SHA:
        raise ValueError("frozen split identity changed")
    if [split.get(k) for k in ("train_subjects", "val_subjects", "reserved_subjects")] != [15, 4, 55]:
        raise ValueError("SC subject roles changed")
    if [split.get(k) for k in ("train_recordings", "val_recordings", "reserved_recordings")] != [29, 8, 109]:
        raise ValueError("SC recording roles changed")
    inp = config.get("input", {})
    if (inp.get("channel"), inp.get("units"), inp.get("sample_rate_hz"),
        inp.get("samples"), inp.get("epoch_seconds")) != ("EEG Fpz-Cz", "uV", 100, 3000, 30):
        raise ValueError("physical signal contract changed")
    if inp.get("stages") != ["Wake", "N1", "N2", "N3", "REM"] or inp.get("unscored_label") != -1 or inp.get("full_grid_inference") is not True:
        raise ValueError("stage or inference mask policy changed")
    if config.get("candidate", {}).get("temporal_radius") != 0:
        raise ValueError("initial model must be epoch-only")
    gates = config.get("gates")
    if not isinstance(gates, list) or len(gates) != 7 or gates[0] != {"id":"D0","status":"DESIGN_ONLY","data_access":False}:
        raise ValueError("design phase gate invalid")
    if any(g.get("status") != "NEEDS_SEPARATE_AUTHORIZATION" for g in gates[1:]):
        raise ValueError("future gates cannot be authorized by this design")
    if matrix.get("schema_version") != "mist-cove-ablation-v0" or matrix.get("status") != "DESIGN_ONLY":
        raise ValueError("experiment matrix not design-only")
    if set(matrix.get("hypotheses", {})) != {"H1", "H2", "H3", "H4"}:
        raise ValueError("hypothesis inventory missing")
    exp = matrix.get("experiments", [])
    if [x.get("id") for x in exp] != [f"A{i}" for i in range(9)]:
        raise ValueError("ablation inventory missing or reordered")
    if matrix.get("cohorts") != {
        "SC_VAL":"SOURCE_SELECTION_ONLY", "SC_RESERVED":"SEALED_WITHIN_STUDY",
        "ST":"CONDITIONAL_EXPLORATORY", "INDEPENDENT_EXTERNAL":"NOT_VERIFIED"}:
        raise ValueError("cohort semantics changed")
    return {"status":"PASS_DESIGN_METADATA_ONLY", "experiments":len(exp), "hypotheses":4, "data_accessed":False, "training_performed":False}


def main() -> None:
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    m = json.loads(MATRIX.read_text(encoding="utf-8"))
    print(json.dumps(verify_contract(c, m), indent=2))


if __name__ == "__main__":
    main()
