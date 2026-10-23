#!/usr/bin/env python3
"""Bounded header-only official-mirror audit, after the scientific Git freeze.

Store original headers privately. Their HTTPS byte hashes are NOT whole-EDF
SHA256 verification. Never request the waveform payload or scoring files.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from mist_sleep.external.edf_header import parse_header
from mist_sleep.external.hmc_manifest import MIRROR_URL, verified_manifest
from mist_sleep.external.protocol import digest
from mist_sleep.external.runner import require_frozen_protocol
from mist_sleep.external.signal_contract import channel_index


def main():
    repo = Path(__file__).resolve().parents[1]
    require_frozen_protocol(repo)
    private = repo / "mist_transfer_runs/mist_sleep/phase_m1_hmc_v1"
    target = private / "headers"
    target.mkdir(mode=0o700, exist_ok=True)
    manifest = verified_manifest(private / "hmc_metadata")
    previous_path = private / "header_audit.private.json"
    previous = {r["recording"]: r for r in json.loads(previous_path.read_text())} if previous_path.exists() else {}

    def fetch(row):
        name = f"recordings/{row['recording']}.edf"
        existing = target / (row["recording"]+".header.bin")
        if existing.is_file():
            raw = existing.read_bytes()
            if hashlib.sha256(raw).hexdigest() != previous.get(row["recording"], {}).get("header_sha256"):
                raise ValueError("private original header identity changed")
            header = parse_header(raw)
            return {"recording": row["recording"], "header_sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw), "header": header, "error": None}
        # One 256-byte range determines the exact total header size.
        def ranged(start, end):
            request = Request(MIRROR_URL + name, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=12) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise ValueError("server did not honor exact bounded header range")
                data = response.read(end-start+2)
            if len(data) != end-start+1:
                raise ValueError("unexpected header range size")
            return data
        try:
            fixed = ranged(0, 255)
            size = int(fixed[184:192])
            if not 512 <= size <= 65536:
                raise ValueError("unbounded EDF header")
            raw = fixed + ranged(256, size-1)
            header = parse_header(raw)
            (target / (row["recording"]+".header.bin")).write_bytes(raw)
            return {"recording": row["recording"], "header_sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw), "header": header, "error": None}
        except Exception as error:
            return {"recording": row["recording"], "bytes": 0, "error": type(error).__name__}
    rows = list(ThreadPoolExecutor(max_workers=4).map(fetch, manifest))
    (private / "header_audit.private.json").write_text(json.dumps(rows, indent=2)+"\n")
    summary = {"status": "HEADER_ONLY_AUDIT", "expected_recordings": 151,
               "headers_verified_over_HTTPS": sum(r["error"] is None for r in rows),
               "header_failures": sum(r["error"] is not None for r in rows),
               "stored_header_bytes": sum(r["bytes"] for r in rows),
               "source": MIRROR_URL, "private_receipt_sha256": digest(rows),
               "waveform_payload_bytes": 0, "scoring_annotation_bytes": 0,
               "whole_EDF_checksum_verified": False, "channels": {}}
    for channel in ("C4/M1", "F4/M1"):
        eligible, rates, units = 0, Counter(), Counter()
        for row in rows:
            if row["error"]:
                continue
            signals = row["header"]["signals"]
            try:
                index = channel_index([s["label"] for s in signals], channel)
            except ValueError:
                continue
            signal = signals[index]
            eligible += 1
            rates[str(signal["sample_rate"])] += 1
            units[signal["unit"]] += 1
        summary["channels"][channel] = {"present": eligible, "sample_rates": dict(rates), "units": dict(units)}
    (repo / "reports/mist_sleep/phase_m1_hmc_v1/header_audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
