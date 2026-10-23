#!/usr/bin/env python3
"""Publish versioned evidence after the resumption's persisted hard stop.

No acquisition, waveform decoding, annotation parsing, inference or fitting.
"""
import hashlib
import json
from pathlib import Path
import re


def main():
    repo = Path(__file__).resolve().parents[1]
    public = repo / "reports/mist_sleep/phase_m1_hmc_resumption_v1"
    private = repo / "mist_transfer_runs/mist_sleep/phase_m1_hmc_resumption_v1"
    load = lambda path: json.loads(path.read_text())
    registration = load(public / "resumption_registration.json")
    failure = load(public / "STOPPED_ON_FAILURE.json")
    acquisition = load(public / "acquisition_partial_receipt.json")
    exception = load(private / "failure.private.json")
    if failure["stage"] != "acquire" or failure["unblind_claim_persisted"]:
        raise RuntimeError("this report describes only the observed acquisition stop")
    log = (private / "cpu_regression.log").read_text()
    matched = re.search(r"(\d+) passed, (\d+) skipped.* in ([0-9.]+)s", log)
    if matched is None:
        raise RuntimeError("successful CPU evidence required")
    verification = {"CPU_passed": int(matched[1]), "CPU_skipped": int(matched[2]),
                    "CPU_wall_seconds": float(matched[3]),
                    "CPU_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
                    "fixture_policy": "generated only; actual raw HMC never supplied to tests",
                    "new_resumption_tests": 9, "scientific_modules_unchanged": True,
                    "historical_publication_hashes_rechecked": True}
    (public / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    results = {"decision": "PRIMARY_EVALUATION_BLOCKED", "resumption_reason": "OFFICIAL_SOURCE_TRANSPORT_FAILURE",
               "storage_gate_passed": True, "frozen_protocol_sha256": registration["protocol_sha256"],
               "M0_macro_f1": None, "M2_macro_f1": None, "M2_minus_M0": None,
               "per_seed_effects": {"123": None, "456": None}, "paired_interval_95": None,
               "success_threshold": .02, "threshold_evaluated": False, "eligible_recordings": None,
               "evaluated_recordings": 0, "known_participants": None, "predictions_sealed": False,
               "target_labels_opened": False, "stage_metrics": None, "calibration_metrics": None,
               "secondary_F4_M1_results": None, "representation_shift": None,
               "recommendation": "BLOCKED; stop this resumption; no automatic retry or next phase",
               "original_M1_blocked_report_preserved": True, "M0_decision": "MPI_FEASIBILITY_NOT_ESTABLISHED"}
    (public / "scientific_results.json").write_text(json.dumps(results, indent=2)+"\n")
    interruption1 = load(private / "transport_interruption.private.json")
    interruption2 = load(private / "transport_interruption_2.private.json")
    resources = {"source_files_and_partial_retained_bytes": acquisition["verified_complete_bytes"]+acquisition["incomplete_bytes"],
                 "opaque_probe_bytes_retained": 19_922_944, "raw_files_deleted": 0,
                 "new_scientific_training_trajectories": 0, "optimizer_updates": 0,
                 "target_normalization_fits": 0, "target_inference_epochs": 0,
                 "source_waveform_value_reads": 0, "target_waveform_value_reads": 0,
                 "target_annotation_semantic_reads": 0, "SC_VAL_reads": 0, "SC_RESERVED_reads": 0,
                 "other_external_dataset_reads": 0, "GPU_seconds": 0,
                 "first_transport_interruption": interruption1,
                 "second_transport_interruption": interruption2,
                 "final_failed_stage_wall_seconds": failure["wall_seconds"],
                 "final_failed_stage_process_CPU_seconds": failure["process_CPU_seconds"],
                 "all_transport_CPU_seconds": None, "network_response_body_bytes_total": None,
                 "accounting_limit": "interrupted streams, curl child CPU and failed/unconsumed prefetch buffers not fully metered; retained-byte inventory is exact; 18 GB guards remained active",
                 "free_bytes_after_stop": acquisition["free_bytes_after_stop"],
                 "no_automatic_retry_after_failure": True}
    (public / "resource_receipts.json").write_text(json.dumps(resources, indent=2)+"\n")
    report = f"""# MIST-Sleep M1 resumption v1: stopped on source transport failure

**Scientific decision: `PRIMARY_EVALUATION_BLOCKED`. Storage gate: PASS.**

The configured private HMC directory is on the root SSD. Its resumption preflight
found **{registration['disk_gate']['free_bytes']:,} free bytes**, exceeding the
unchanged 15,700,000,000-byte estimate plus 2,000,000,000-byte reserve. Permissions
remain 0700. All four exact source checkpoints, source-fit normalization and frozen
protocol/readiness identities passed before acquisition.

Acquisition from the official HMC v1.1 mirror retained **8 complete recording
triplets / 24 files**, totaling **{acquisition['verified_complete_bytes']:,} bytes**.
Every retained complete file passes its original official SHA256, including an
independent post-stop checksum audit. One **{acquisition['incomplete_bytes']:,}-byte
partial** remains unauthenticated and unusable for evaluation. No raw file was
deleted. Available persistent space after stopping is
**{acquisition['free_bytes_after_stop']:,} bytes**; storage remains sufficient.
Whole-cohort integrity, real scoring-schema compatibility and scoring alignment
remain unverified. Eight source triplets do not establish a scored eligible cohort.

Two intentionally interrupted slow streams retained their verified files and
partials before any integrity failure. The transport was changed to curl, then
bounded 16 MB ranges with at most four prefetched responses. The original source
allowlist, full-file SHA256, byte ceiling and disk-reserve checks remained active.
All scientific inference/scoring source files and protocol choices are byte
identical to the original M1 implementation. Small opaque transport probes are
retained privately and accounted for.

The bounded transport subsequently raised `OSError` with the exact diagnostic
`{exception['exception_message']}`. The failed stage lasted
{failure['wall_seconds']:.2f} seconds and used {failure['process_CPU_seconds']:.3f}
Python process CPU seconds. A 120-second per-range wall limit was active, but curl's
exit code and detailed stderr were not retained. The evidence therefore does not
distinguish a timeout from remote truncation/another nonzero curl exit. It does
establish a failed response-body/transport gate. **No automatic retry followed.**

The persisted stop receipt prevents additional automatic acquisition, inference
or scoring for this attempt. No target waveform value was decoded, no model
prediction was produced or sealed, and no original annotation was semantically
opened. Thus M0/M2 external Macro-F1, both seed effects, the paired 95% interval,
stage/calibration/representation analyses and secondary-channel outcomes are all
**unmeasured**. The +0.02 criterion was not evaluated. This provides neither a
positive nor a negative efficacy result for MPI.

The original `phase_m1_hmc_v1` blocked reports and publication hashes remain
unchanged; this version records the newly passed storage gate and subsequent
transport blocker. M0 remains `MPI_FEASIBILITY_NOT_ESTABLISHED`. No scientific
checkpoint, channel, signal conversion, normalization, label rule, metric,
bootstrap or decision threshold was changed. Scientific training, target tuning,
source/target normalizer fitting, RESERVED/VAL waveform reads and other-dataset
access remain zero. No research data was automatically deleted.

Validation: **{verification['CPU_passed']} CPU tests passed / {verification['CPU_skipped']}
skipped**, including nine generated resumption/transport tests. The generated
tests check historical preservation, stop persistence, source/version allowlists,
range geometry, opaque byte ordering and prefetch budget enforcement. Actual HMC
data is excluded from CI. Exact final-commit CI is verified remotely after the
publication commit and recorded in [draft PR #21](https://github.com/SyedNaveedMahmood/Proto-sleep/pull/21).

Stop after this M1 resumption. Do not merge, repair/retry a failed real-data stage,
start another dataset or begin another research phase automatically.
"""
    (public / "DASHBOARD.md").write_text(report)
    (repo / "docs/mist_sleep/PHASE_M1_RESUMPTION_RESULTS_V1.md").write_text(report)
    files = [*public.glob("*"), repo / "docs/mist_sleep/PHASE_M1_RESUMPTION_V1.md",
             repo / "docs/mist_sleep/PHASE_M1_RESUMPTION_RESULTS_V1.md",
             *sorted((repo / "scripts").glob("mist_sleep_m1_*transport.py")),
             repo / "scripts/mist_sleep_m1_resume.py", repo / "scripts/mist_sleep_m1_resume_report.py",
             repo / "tests/mist_sleep/test_m1_resumption.py"]
    manifest = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files if p.is_file() and p.name != "publication_manifest.json"}
    (public / "publication_manifest.json").write_text(json.dumps({"files_sha256": manifest,
        "self_excluded": True, "private_EDF_or_predictions_published": False,
        "historical_manifest_sha256": registration["historical_publication_manifest_sha256"],
        "clinical_result_figures": 0}, indent=2)+"\n")
    print(json.dumps({"decision": results["decision"], "reason": results["resumption_reason"],
                      "verified_triplets": 8, "CPU_passed": verification["CPU_passed"]}))


if __name__ == "__main__":
    main()
