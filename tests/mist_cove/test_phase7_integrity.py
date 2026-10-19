"""Synthetic filesystem fixtures only; never open actual historical weights."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch

from mist_cove.phase7 import integrity as gate


def put(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("scope", ["train", "val", "reserved", "test", "ST", "SHHS", None])
def test_unauthorized_scope_before_path_conversion(scope):
    class ForbiddenPath:
        def __fspath__(self):
            pytest.fail("filesystem touched before scope rejection")
    with pytest.raises(gate.IntegrityError, match="before filesystem"):
        gate.inspect(ForbiddenPath(), scope=scope)
    with pytest.raises(gate.IntegrityError, match="before filesystem"):
        gate.write_receipt(ForbiddenPath(), "anything", {}, scope=scope)


@pytest.mark.parametrize("name", ["/tmp/foreign", "../foreign", "x/../../foreign", ""])
def test_confined_input_rejects_path_escape(tmp_path, name):
    with pytest.raises(gate.IntegrityError):
        gate.confined_file(tmp_path, name)


def test_confined_input_rejects_symlink_before_target_read(tmp_path):
    (tmp_path / "linked").symlink_to(tmp_path / "nonexistent-foreign-root")
    with pytest.raises(gate.IntegrityError, match="symlink"):
        gate.confined_file(tmp_path, "linked/private")


def test_digest_mismatch_precedes_json_parsing(tmp_path):
    put(tmp_path, "bad.json", b"not even json")
    with pytest.raises(gate.IntegrityError, match="digest mismatch"):
        gate.verified_bytes(tmp_path, "bad.json", "0" * 64)


def test_observed_hash_is_not_promoted_to_historical_seal(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint deserialized")
    monkeypatch.setattr(torch, "load", forbidden)
    weight = put(tmp_path, gate.PRIVATE_RUN + "/single_123_full.pt", b"untrusted bytes")
    weight.with_suffix(".seal.json").write_text(json.dumps({"sha256": gate.sha256(weight.read_bytes())}))
    rows = gate.checkpoint_inventory(tmp_path)
    row = rows[0]
    assert row["observed_sha256"] == hashlib.sha256(b"untrusted bytes").hexdigest()
    assert row["sidecar_names_present"] == ["single_123_full.seal.json"]
    assert row["status"] == "BLOCKED" and not row["digest_is_historical_seal"]
    assert row["reason"] == "NO_TRUSTED_HISTORICAL_WEIGHT_DIGEST"
    weight.write_bytes(b"changed after inventory")
    later = gate.checkpoint_inventory(tmp_path)[0]
    assert later["observed_sha256"] != row["observed_sha256"]
    assert later["status"] == "BLOCKED"  # A stale local sidecar unlocks nothing.


def test_missing_checkpoint_not_replaced_by_cuda_correctness_model(tmp_path):
    put(tmp_path, "mist_transfer_runs/mist_cove/phase6_v1/cuda_v1/M1/generated.pt", b"wrong model")
    rows = gate.checkpoint_inventory(tmp_path)
    assert len(rows) == 6
    assert all(r["reason"] == "CHECKPOINT_MISSING" for r in rows)
    assert all(r["observed_sha256"] is None for r in rows)


def result_fixture(tmp_path):
    rows = []
    for task in gate.TASKS:
        for seed in gate.SEEDS:
            for index in range(12):
                rows.append({"task": task, "seed": seed, "stage": "VI-3",
                             "method": "full" if index == 0 else f"other-{index}",
                             "training": {"optimizer_updates": 300, "example_exposures": 4800}})
    result = {"fits": 72, "rows": rows,
              "gates": {"VI-3": {"status": "FAIL"}, "VI-4": {"optimizer_updates": 0}}}
    data = json.dumps(result).encode()
    put(tmp_path, gate.PRIVATE_RUN + "/results.json", data)
    return result, {"private_receipt_sha256": gate.sha256(data)}


def test_authenticated_metrics_are_separate_from_weight_provenance(tmp_path):
    _, execution = result_fixture(tmp_path)
    result = gate.verify_results(tmp_path, execution)
    assert result["fits"] == 72 and result["historical_robustness"] == "FAIL"
    assert result["historical_head_updates"] == 0
    assert all(r["status"] == "BLOCKED" for r in gate.checkpoint_inventory(tmp_path))


def test_private_result_tamper_rejected(tmp_path):
    _, execution = result_fixture(tmp_path)
    put(tmp_path, gate.PRIVATE_RUN + "/results.json", b"{}")
    with pytest.raises(gate.IntegrityError, match="digest mismatch"):
        gate.verify_results(tmp_path, execution)


@pytest.mark.parametrize("change", ["count", "updates", "exposure", "historical_gate", "heads", "identity"])
def test_inconsistent_historical_accounting_rejected_even_with_matching_hash(tmp_path, change):
    r, _ = result_fixture(tmp_path)
    if change == "count":
        r["fits"] = 71
    elif change == "updates":
        r["rows"][0]["training"]["optimizer_updates"] = 301
    elif change == "exposure":
        r["rows"][0]["training"]["example_exposures"] = 16
    elif change == "historical_gate":
        r["gates"]["VI-3"]["status"] = "PASS"
    elif change == "heads":
        r["gates"]["VI-4"]["optimizer_updates"] = 1
    else:
        r["rows"][0]["seed"] = 1
    data = json.dumps(r).encode()
    put(tmp_path, gate.PRIVATE_RUN + "/results.json", data)
    with pytest.raises(gate.IntegrityError):
        gate.verify_results(tmp_path, {"private_receipt_sha256": gate.sha256(data)})


def test_atomic_receipt_and_overwrite_refusal(tmp_path):
    name = gate.OUTPUT_ROOT + "/fixture/receipt.json"
    gate.write_receipt(tmp_path, name, {"status": "BLOCKED"})
    original = (tmp_path / name).read_bytes()
    with pytest.raises(FileExistsError):
        gate.write_receipt(tmp_path, name, {"status": "PASS"})
    assert (tmp_path / name).read_bytes() == original
    assert not list((tmp_path / name).parent.glob(".receipt-*"))


def test_racing_receipt_writer_is_preserved(tmp_path, monkeypatch):
    original_link = gate.os.link
    def race(source, target):
        Path(target).write_text('"other writer"')
        original_link(source, target)
    monkeypatch.setattr(gate.os, "link", race)
    name = gate.OUTPUT_ROOT + "/raced.json"
    with pytest.raises(FileExistsError):
        gate.write_receipt(tmp_path, name, {"status": "BLOCKED"})
    assert (tmp_path / name).read_text() == '"other writer"'
    assert not list((tmp_path / name).parent.glob(".receipt-*"))


@pytest.mark.parametrize("name", ["reports/public.json", "mist_transfer_runs/mist_cove/phase6_v1/old.json",
                                  gate.OUTPUT_ROOT + "/../../escape.json"])
def test_receipts_cannot_write_outside_new_ignored_root(tmp_path, name):
    with pytest.raises(gate.IntegrityError):
        gate.write_receipt(tmp_path, name, {})


@pytest.fixture
def historical_repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = "src/historical.py"
    put(tmp_path, source, b"original source\n")
    manifest = "experiments/proof.json"
    put(tmp_path, manifest, json.dumps({"files": {source: gate.sha256(b"original source\n")}}).encode())
    for name in (".github/workflows/ci.yml", "pyproject.toml", "configs/ci_cpu_constraints_v1.txt"):
        put(tmp_path, name, b"fixture\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "synthetic fixture"], cwd=tmp_path, check=True)
    monkeypatch.setattr(gate, "BASE", gate.git(tmp_path, "rev-parse", "HEAD").decode().strip())
    monkeypatch.setattr(gate, "MANIFESTS", (manifest,))
    return tmp_path, source, manifest


def test_snapshot_accepts_additions_and_keeps_original_proof(historical_repo):
    root, _, _ = historical_repo
    put(root, "src/new_integrity.py", b"new source\n")
    subprocess.run(["git", "add", "src/new_integrity.py"], cwd=root, check=True)
    proof = gate.snapshot(root)
    assert proof["status"] == "PASS"
    assert "src/historical.py" in proof["files"]


def test_mutating_source_and_manifest_together_cannot_rewrite_git_anchor(historical_repo):
    root, source, manifest = historical_repo
    put(root, source, b"altered source\n")
    put(root, manifest, json.dumps({"files": {source: gate.sha256(b"altered source\n")}}).encode())
    with pytest.raises(gate.IntegrityError, match="historical tracked files changed"):
        gate.snapshot(root)


def test_missing_private_artifacts_report_blocked_not_training(tmp_path, monkeypatch):
    put(tmp_path, gate.CONFIG, json.dumps({"scope": "SYNTHETIC_ONLY", "mode": "integrity_only",
        "base_commit": gate.BASE, "allowed_fits": 0, "allowed_optimizer_updates": 0,
        "real_data_authorized": False}).encode())
    put(tmp_path, "reports/mist_cove/phase6_v1/execution_receipt_v1.json",
        json.dumps({"private_receipt_sha256": "0" * 64}).encode())
    monkeypatch.setattr(gate, "snapshot", lambda root: {"status": "PASS"})
    monkeypatch.setattr(gate, "git", lambda *a: b"fixture\n")
    result = gate.inspect(tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["scientific_fits"] == result["optimizer_updates"] == 0
    assert result["checkpoint_deserializations"] == 0
    assert result["gates"]["VII-5"] == "NOT TESTED"
    assert "PRIVATE_RESULT_PROOF_UNAVAILABLE_OR_INVALID" in result["blockers"]


@pytest.mark.parametrize("change", [{"allowed_fits": 1}, {"allowed_optimizer_updates": 1},
                                    {"real_data_authorized": True}, {"base_commit": "main"}])
def test_config_cannot_unlock_training_before_historical_verification(tmp_path, monkeypatch, change):
    config = {"scope": "SYNTHETIC_ONLY", "mode": "integrity_only", "base_commit": gate.BASE,
              "allowed_fits": 0, "allowed_optimizer_updates": 0, "real_data_authorized": False}
    config.update(change)
    put(tmp_path, gate.CONFIG, json.dumps(config).encode())
    def forbidden(*args):
        pytest.fail("advanced past configuration rejection")
    monkeypatch.setattr(gate, "snapshot", forbidden)
    with pytest.raises(gate.IntegrityError, match="zero-fit configuration"):
        gate.inspect(tmp_path)


def test_checkpoint_inventory_rejects_symlinked_sidecar(tmp_path):
    weight = put(tmp_path, gate.PRIVATE_RUN + "/single_123_full.pt", b"untrusted")
    weight.with_suffix(".seal.json").symlink_to(tmp_path / "foreign.json")
    with pytest.raises(gate.IntegrityError, match="symlink"):
        gate.checkpoint_inventory(tmp_path)
