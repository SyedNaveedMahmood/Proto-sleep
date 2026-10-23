#!/usr/bin/env python3
"""Versioned M1 continuation using the unchanged authenticated execution modules.

Stages are invoked separately so the public prediction seal is committed before
one-time scoring. Failures persist and prohibit further automatic execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time

from mist_sleep.external.hmc_manifest import acquire, private_root, verified_manifest, MIRROR_URL
from mist_sleep.external.prediction_seal import write_once
from mist_sleep.external.protocol import digest, file_sha256
from mist_sleep.external.runner import blind, score, require_frozen_protocol, require_inference_readiness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("acquire", "blind", "score"))
    parser.add_argument("--resume-interrupted-curl-transport", action="store_true")
    parser.add_argument("--transport-session", type=int, default=1)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    public = repo / "reports/mist_sleep/phase_m1_hmc_resumption_v1"
    private = repo / "mist_transfer_runs/mist_sleep/phase_m1_hmc_resumption_v1"
    metadata = repo / "mist_transfer_runs/mist_sleep/phase_m1_hmc_v1/hmc_metadata"
    configuration = json.loads((private / "configuration.private.json").read_text())
    os.environ["HMC_DATA_ROOT"] = configuration["HMC_DATA_ROOT"]
    root = private_root()
    prediction_root = Path(configuration["prediction_root"])
    registration = json.loads((public / "resumption_registration.json").read_text())
    if file_sha256(repo / registration["historical_blocked_report"] / "publication_manifest.json") != registration["historical_publication_manifest_sha256"]:
        raise RuntimeError("historical M1 blocked evidence changed")
    protocol = require_frozen_protocol(repo)
    require_inference_readiness(repo)
    if digest(protocol) != registration["protocol_sha256"]:
        raise RuntimeError("resumption protocol identity changed")
    if (public / "STOPPED_ON_FAILURE.json").exists():
        raise RuntimeError("resumption already stopped; automatic retry forbidden")
    if args.resume_interrupted_curl_transport:
        if args.stage != "acquire":
            raise RuntimeError("transport option only permits source acquisition")
        interruption = json.loads((private / "transport_interruption.private.json").read_text())
        if not interruption["no_integrity_or_storage_failure"]:
            raise RuntimeError("cannot resume a failed integrity or storage gate")
        if args.transport_session == 2:
            interruption2 = json.loads((private / "transport_interruption_2.private.json").read_text())
            if not interruption2["no_integrity_or_storage_failure"]:
                raise RuntimeError("cannot resume failed source acquisition")
        elif args.transport_session != 1:
            raise RuntimeError("unregistered transport session")
        from mist_sleep_m1_curl_transport import install
        install()
    started = {"stage": args.stage, "timestamp_UTC": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": digest(protocol), "automatic_deletion": False}
    stage_identity = "acquire_curl_transport" if args.resume_interrupted_curl_transport else args.stage
    if args.resume_interrupted_curl_transport and args.transport_session == 2:
        stage_identity += "_2"
    started["transport"] = "curl via original acquisition guards" if args.resume_interrupted_curl_transport else "original"
    write_once(private / f"{stage_identity}_started.json", started)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        if args.stage == "acquire":
            manifest = verified_manifest(metadata)
            result = acquire(root, manifest, source_url=MIRROR_URL)
            result.update(status="COMPLETE_OFFICIAL_SOURCE_CHECKSUMS_VERIFIED", recordings=len(manifest),
                          files=453, manifest_sha256=digest(manifest),
                          stored_source_bytes=sum((root / name).stat().st_size for row in manifest for name in row["files"]),
                          source=MIRROR_URL, annotation_semantic_reads=0, waveform_value_reads=0)
            if args.resume_interrupted_curl_transport:
                from mist_sleep_m1_curl_transport import statistics
                result["transport"] = statistics()
            output = "acquisition_receipt.json"
        elif args.stage == "blind":
            receipt = json.loads((public / "acquisition_receipt.json").read_text())
            if receipt["status"] != "COMPLETE_OFFICIAL_SOURCE_CHECKSUMS_VERIFIED":
                raise RuntimeError("complete authenticated source acquisition required")
            result = blind(repo, metadata, prediction_root)
            output = "prediction_seal.json"
        else:
            relative = "reports/mist_sleep/phase_m1_hmc_resumption_v1/prediction_seal.json"
            result = score(repo, metadata, prediction_root, relative)
            output = "external_results.json"
        write_once(public / output, result)
        resource_receipt = {**started, "status": "COMPLETE", "wall_seconds": time.monotonic()-wall,
                            "process_CPU_seconds": time.process_time()-cpu,
                            "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            "optimizer_updates": 0, "target_normalization_fits": 0}
        write_once(public / f"{args.stage}_resource_receipt.json", resource_receipt)
        print(json.dumps(resource_receipt), flush=True)
    except Exception as error:
        message = str(error)
        reason = {"unverified HMC text scoring schema; do not guess": "UNVERIFIED_REAL_HMC_TEXT_SCHEMA",
                  "waveform/scoring absolute start-time mismatch": "WAVEFORM_SCORING_START_TIME_MISMATCH",
                  "EDF+/text annotation disagreement": "EDF_TEXT_ANNOTATION_DISAGREEMENT",
                  "unsupported scoring text": "UNSUPPORTED_SCORING_TEXT",
                  "stage interval is not on the fixed 30-second grid": "SCORING_NOT_ON_FROZEN_EPOCH_GRID"}.get(message, "EXECUTION_GATE_FAILURE")
        private_failure = {**started, "exception_type": type(error).__name__, "exception_message": message}
        write_once(private / "failure.private.json", private_failure)
        failed = {**started, "status": "STOPPED_ON_FAILURE", "decision": "PRIMARY_EVALUATION_BLOCKED",
                  "reason": reason, "exception_type": type(error).__name__,
                  "wall_seconds": time.monotonic()-wall, "process_CPU_seconds": time.process_time()-cpu,
                  "unblind_claim_persisted": (prediction_root / "UNBLIND_STARTED.json").exists(),
                  "primary_effect": None, "paired_interval_95": None, "optimizer_updates": 0,
                  "target_normalization_fits": 0, "automatic_retry": False, "automatic_deletion": False}
        write_once(public / "STOPPED_ON_FAILURE.json", failed)
        print(json.dumps(failed), flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
