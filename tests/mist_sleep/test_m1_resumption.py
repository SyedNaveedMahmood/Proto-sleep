"""Generated-only checks for the resumption's historical and failure barriers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import io
from urllib.request import Request

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


def load_generated_transport():
    source = Path(__file__).resolve().parents[2] / "scripts/mist_sleep_m1_curl_transport.py"
    spec = importlib.util.spec_from_file_location("generated_m1_curl_transport", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("url", ["https://example.org/recordings/SN001.edf",
    "https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.0/recordings/SN001.edf",
    "https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/../other.edf"])
def test_transport_rejects_unapproved_source_before_process(url, monkeypatch):
    module = load_generated_transport()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("no process for unapproved source"))
    with pytest.raises(ValueError): module.CurlResponse(Request(url), 30)


def test_transport_preserves_range_headers_and_opaque_bytes(monkeypatch):
    module = load_generated_transport()
    commands = []
    class GeneratedProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = io.BytesIO(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 10-14/15\r\n\r\nabcde")
            self.returncode = 0
        def wait(self): return self.returncode
        def communicate(self): return b"", b""
        def kill(self): self.returncode = -9
    monkeypatch.setattr(module.subprocess, "Popen", GeneratedProcess)
    request = Request("https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/recordings/SN001.edf",
                      headers={"Range": "bytes=10-"})
    with module.CurlResponse(request, 30) as response:
        assert response.status == 206 and response.headers.get("Content-Range") == "bytes 10-14/15"
        assert response.read(3) == b"abc" and response.read(3) == b"de" and response.read(3) == b""
    assert commands[0][1] == "--disable" and any(v.startswith("10-") for v in commands[0]) and "--location" not in commands[0]


def test_transport_nonzero_exit_cannot_look_like_success(monkeypatch):
    module = load_generated_transport()
    class GeneratedProcess:
        def __init__(self, *args, **kwargs):
            self.stdout = io.BytesIO(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-0/1\r\n\r\na")
            self.returncode = 28
        def wait(self): return self.returncode
        def communicate(self): return b"", b"generated network timeout"
        def kill(self): self.returncode = -9
    monkeypatch.setattr(module.subprocess, "Popen", GeneratedProcess)
    request = Request("https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/recordings/SN001.edf")
    with pytest.raises(OSError):
        with module.CurlResponse(request, 30) as response:
            response.read(100)


def test_bounded_prefetch_preserves_exact_byte_order(monkeypatch):
    module = load_generated_transport()
    monkeypatch.setattr(module, "CHUNK_BYTES", 4)
    class GeneratedProcess:
        def __init__(self, command, **kwargs):
            requested = command[command.index("--range")+1]
            start, end = map(int, requested.split("-")); end = min(end, 9)
            body = b"abcdefghij"[start:end+1]
            self.stdout = io.BytesIO(f"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/10\r\n\r\n".encode()+body)
            self.returncode = 0
        def wait(self): return self.returncode
        def communicate(self): return b"", b""
        def kill(self): self.returncode = -9
    monkeypatch.setattr(module.subprocess, "Popen", GeneratedProcess)
    request = Request("https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/recordings/SN001.edf")
    with module.CurlResponse(request, 30) as response:
        chunks = []
        while chunk := response.read(3): chunks.append(chunk)
    assert b"".join(chunks) == b"abcdefghij"


def test_prefetch_reserves_byte_budget_before_network(monkeypatch):
    module = load_generated_transport()
    monkeypatch.setattr(module, "_budget_available", module.CHUNK_BYTES-1)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("source byte ceiling must precede network"))
    request = Request("https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/recordings/SN001.edf")
    with pytest.raises(OSError, match="byte ceiling"):
        module.CurlResponse(request, 30)
