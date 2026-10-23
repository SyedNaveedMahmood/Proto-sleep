"""Generated-only checks for the resumption's historical and failure barriers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def setup_generated_resume(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts/mist_sleep_m1_resume.py"
    spec = importlib.util.spec_from_file_location("generated_m1_resume", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/mist_sleep_m1_resume.py"))
    public = tmp_path / "reports/mist_sleep/phase_m1_hmc_resumption_v1"
    private = tmp_path / "mist_transfer_runs/mist_sleep/phase_m1_hmc_resumption_v1"
    historical = tmp_path / "reports/mist_sleep/phase_m1_hmc_v1"
    for directory in (public, private, historical):
        directory.mkdir(parents=True)
    historical_bytes = b'{"historical":"PRIMARY_EVALUATION_BLOCKED"}\n'
    (historical / "publication_manifest.json").write_bytes(historical_bytes)
    protocol = {"generated": True}
    registration = {"historical_blocked_report": "reports/mist_sleep/phase_m1_hmc_v1",
                    "historical_publication_manifest_sha256": hashlib.sha256(historical_bytes).hexdigest(),
                    "protocol_sha256": module.digest(protocol)}
    (public / "resumption_registration.json").write_text(json.dumps(registration))
    raw = tmp_path / "generated_raw"; raw.mkdir(mode=0o700)
    (private / "configuration.private.json").write_text(json.dumps({"HMC_DATA_ROOT": str(raw),
              "prediction_root": str(tmp_path / "generated_predictions")}))
    monkeypatch.setattr(module, "require_frozen_protocol", lambda _: protocol)
    monkeypatch.setattr(module, "require_inference_readiness", lambda _: None)
    monkeypatch.setattr(module, "verified_manifest", lambda _: [])
    monkeypatch.setattr(sys, "argv", ["generated_resume", "acquire"])
    return module, public, private, historical, historical_bytes


def test_resumption_stops_on_failure_and_preserves_history(tmp_path, monkeypatch):
    module, public, private, historical, original = setup_generated_resume(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise ValueError("source checksum mismatch")
    monkeypatch.setattr(module, "acquire", fail)
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 2
    stopped = json.loads((public / "STOPPED_ON_FAILURE.json").read_text())
    assert stopped["decision"] == "PRIMARY_EVALUATION_BLOCKED"
    assert not stopped["automatic_retry"] and not stopped["automatic_deletion"]
    assert (historical / "publication_manifest.json").read_bytes() == original
    assert (private / "failure.private.json").is_file()
    with pytest.raises(RuntimeError, match="already stopped"):
        module.main()


def test_changed_historical_receipt_blocks_before_acquisition(tmp_path, monkeypatch):
    module, public, private, historical, _ = setup_generated_resume(tmp_path, monkeypatch)
    (historical / "publication_manifest.json").write_text("changed historical evidence")
    monkeypatch.setattr(module, "acquire", lambda *a, **k: pytest.fail("acquisition must not start"))
    with pytest.raises(RuntimeError, match="historical M1 blocked evidence changed"):
        module.main()
    assert not (private / "acquire_started.json").exists()
