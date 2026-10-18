"""End-to-end guarded source-runner wiring with fake proofs and mock loaders.

The fake authorization and fake full-grid count prove software paths only.
They never authorize or validate a real source experiment.
"""
import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from mist_cove.phase2 import source
from mist_cove.phase2.provenance import ROOT, digest, file_sha, implementation_snapshot
from mist_cove.phase2.synthetic_runtime import mock_recordings, synthetic_bank


def mock_launch(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs/mist_cove_source_prepared_v1.json").read_text())
    cfg_path = tmp_path / "source_config.json"; cfg_path.write_text(json.dumps(config))
    split = tmp_path / "fake_split.json"; split.write_text("FAKE SYNTHETIC SPLIT; NEVER A REAL FROZEN SPLIT")
    rows = []
    for role in ("train", "val"):
        rows.append({"recording_id": "generated_" + role, "subject_id": "generated_person_" + role,
                     "source_sha256": ("a" if role == "train" else "b") * 64, "annotation_sha256": "c" * 64,
                     "source_stat_proof": {"synthetic": True}, "annotation_stat_proof": {"synthetic": True}, "split": role})
    from mist_transfer.experiment import source_input_proof
    norm = {"mean_uV": 0., "std_uV": 1., "fit_role": "train", "label_access": False,
            "physical_epoch_count": 79984, "fit_subjects": [rows[0]["subject_id"]],
            "source_hashes": {rows[0]["recording_id"]: rows[0]["source_sha256"]}}
    norm_path = tmp_path / "normalization.json"
    norm_path.write_text(json.dumps({"statistics": norm, "input_proof": source_input_proof(rows)}))
    done_path = tmp_path / "normalization_receipt.json"
    done_path.write_text(json.dumps({"outputs": {"normalization.json": file_sha(norm_path)}}))
    correctness = tmp_path / "fake_correctness.json"; correctness.write_text('{"scope":"MOCK_ONLY"}')
    banks = {}
    for seed in (123, 456, 789):
        bank = synthetic_bank(); bank["seed"] = seed
        path = tmp_path / f"synthetic_bank_{seed}.pt"; torch.save(bank, path)
        banks[str(seed)] = {"path": str(path), "sha256": file_sha(path)}
    receipt = {"scope": "SC_TRAIN_VAL_ONLY_AUTHORIZED", "authorization_reference": "MOCK_ONLY_NOT_REAL_AUTHORIZATION",
               "config_sha256": digest(config), "implementation_sha256": implementation_snapshot()["sha256"],
               "split_fingerprint": source.SPLIT, "split_file_sha256": source.SPLIT_SHA,
               "cuda_exact_resume_passed": True, "cpu_regression_passed": True, "gpu_feasibility_passed": True,
               "normalization_path": str(norm_path), "normalization_sha256": file_sha(norm_path),
               "normalization_receipt_path": str(done_path), "normalization_receipt_sha256": file_sha(done_path),
               "banks": banks, "correctness_artifacts": [{"path": str(correctness), "sha256": file_sha(correctness)}]}
    launch = tmp_path / "fake_launch.json"; launch.write_text(json.dumps(receipt))
    mock_train, mock_val = mock_recordings()
    import mist_transfer.protocol as protocol
    import mist_transfer.source_data as source_data
    import mist_transfer.full_experiment as full
    split_parser = Mock(return_value={"split_fingerprint": source.SPLIT})
    role_loader = Mock(side_effect=lambda frozen, role: [rows[0]] if role == "train" else [rows[1]])
    record_builder = Mock(side_effect=lambda row, norm, smoke: mock_train if row["split"] == "train" else mock_val)
    bank_validator = Mock()
    grid_validator = Mock(return_value={"status": "SYNTHETIC_MOCK_COUNTS_NOT_REAL_GRID", "reserved_access": False})
    monkeypatch.setattr(protocol, "read_frozen_split", split_parser)
    monkeypatch.setattr(protocol, "load_development", role_loader)
    monkeypatch.setattr(source_data, "SourceRecording", record_builder)
    monkeypatch.setattr(source_data, "validate_bank", bank_validator)
    monkeypatch.setattr(full, "verify_grid", grid_validator)
    monkeypatch.setattr(source, "configure_precision", lambda device: None)
    monkeypatch.setattr(source, "ROOT", tmp_path)
    monkeypatch.setattr(source, "file_sha", lambda path: source.SPLIT_SHA if Path(path) == split else file_sha(path))
    return cfg_path, split, launch, receipt, norm_path, (split_parser, role_loader, record_builder, bank_validator, grid_validator)


def test_guarded_preflight_uses_only_train_val_and_no_training(tmp_path, monkeypatch):
    config, split, launch, receipt, norm, spies = mock_launch(tmp_path, monkeypatch)
    train_spy = Mock(side_effect=AssertionError("preflight must never train")); monkeypatch.setattr(source, "sealed_train", train_spy)
    result = source.run_source(config, split, tmp_path / "mist_transfer_runs/mist_cove/phase2_source_v1/run",
                               authorize_source_train_val=True, launch_receipt_path=launch)
    assert result["status"] == "PREFLIGHT_ONLY_NO_TRAINING"
    assert [call.args[1] for call in spies[1].call_args_list] == ["train", "val"]
    assert spies[2].call_count == 2 and spies[3].call_count == 3 and spies[4].call_count == 1
    assert all(call.kwargs == {"smoke": False} for call in spies[2].call_args_list)
    train_spy.assert_not_called()


def test_mock_future_comparison_wires_frozen_recipe_all_variants(tmp_path, monkeypatch):
    config, split, launch, receipt, norm, spies = mock_launch(tmp_path, monkeypatch)
    trainer = Mock(return_value=(None, {"mean_subject_macro_f1": .2}, [])); monkeypatch.setattr(source, "sealed_train", trainer)
    output = tmp_path / "mist_transfer_runs/mist_cove/phase2_source_v1/run"
    result = source.run_source(config, split, output, authorize_source_train_val=True, launch_receipt_path=launch, preflight_only=False)
    assert len(result) == 18 and trainer.call_count == 18
    assert {row["variant"] for row in result} == set(source.VARIANTS)
    for call in trainer.call_args_list:
        cfg = call.args[3]
        assert cfg.epochs == 60 and cfg.patience == 12 and cfg.lr == .0003 and cfg.weight_decay == .0001
        assert cfg.core_epochs == cfg.encode_batch == 16 and cfg.grad_clip == 5.
        assert call.kwargs["device"] == "cuda"


@pytest.mark.parametrize("corruption", ["split", "normalization", "bank", "correctness", "normalization_role", "launch_code"])
def test_mock_proof_corruption_blocks_before_training(tmp_path, monkeypatch, corruption):
    config, split, launch, receipt, norm_path, spies = mock_launch(tmp_path, monkeypatch)
    train_spy = Mock(side_effect=AssertionError("invalid proof trained")); monkeypatch.setattr(source, "sealed_train", train_spy)
    if corruption == "split": monkeypatch.setattr(source, "file_sha", lambda path: "bad" if Path(path) == split else file_sha(path))
    if corruption == "normalization": norm_path.write_text("changed")
    if corruption == "bank": Path(receipt["banks"]["123"]["path"]).write_bytes(b"never deserialize")
    if corruption == "correctness": Path(receipt["correctness_artifacts"][0]["path"]).write_text("changed")
    if corruption == "launch_code":
        receipt["implementation_sha256"] = "bad"; launch.write_text(json.dumps(receipt))
    if corruption == "normalization_role":
        saved = json.loads(norm_path.read_text()); saved["statistics"]["fit_role"] = "val"; norm_path.write_text(json.dumps(saved))
        receipt["normalization_sha256"] = file_sha(norm_path)
        done = Path(receipt["normalization_receipt_path"]); done.write_text(json.dumps({"outputs": {"normalization.json": file_sha(norm_path)}}))
        receipt["normalization_receipt_sha256"] = file_sha(done); launch.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        source.run_source(config, split, tmp_path / "mist_transfer_runs/mist_cove/phase2_source_v1/run",
                          authorize_source_train_val=True, launch_receipt_path=launch, preflight_only=False)
    train_spy.assert_not_called()
