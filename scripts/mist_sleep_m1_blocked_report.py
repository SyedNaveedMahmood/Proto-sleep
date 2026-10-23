#!/usr/bin/env python3
"""Render the actual blocked M1 evidence, with null clinical outcomes.

Consumes metadata and generated validation receipts only. Never starts acquisition,
training, target inference or annotation access. No fabricated performance figures.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mist_sleep.external.protocol import digest, file_sha256, specification
from mist_sleep.external.signal_contract import physical_epochs

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "reports/mist_sleep/phase_m1_hmc_v1"
PRIVATE = REPO / "mist_transfer_runs/mist_sleep/phase_m1_hmc_v1"
DOCS = REPO / "docs/mist_sleep"


def write(name, value):
    (PUBLIC / name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def document(name, title, content):
    (DOCS / name).write_text(f"# {title}\n\n{content.strip()}\n")


def main():
    protocol = specification()
    validation = json.loads((PUBLIC/"generated_validation.json").read_text())
    metadata = json.loads((PRIVATE/"hmc_metadata/access.json").read_text())
    preflight = json.loads((PRIVATE/"preflight.private.json").read_text())
    headers = json.loads((PUBLIC/"header_audit.json").read_text())
    test_log = (PRIVATE/"cpu_regression.log").read_text()
    counts = re.search(r"(\d+) passed, (\d+) skipped.* in ([\d.]+)s", test_log)
    if counts is None:
        raise RuntimeError("successful full CPU regression evidence required")
    cpu = {"passed": int(counts[1]), "skipped": int(counts[2]), "wall_seconds": float(counts[3]),
           "runtime": "PyTorch 2.10 CPU; generated fixtures only", "log_sha256": file_sha256(PRIVATE/"cpu_regression.log")}
    if preflight["pass"]:
        raise RuntimeError("this renderer only represents the observed failed-capacity run")
    blocked = {"status": "NOT_RUN_BLOCKED", "reason": "INSUFFICIENT_PERSISTENT_DISK",
               "eligible_recordings": None, "evaluated_recordings": 0}
    write("data_access_receipt.json", {"status": "METADATA_AND_HEADER_ONLY", "dataset": "HMC", "version": "1.1",
          "metadata": metadata, "metadata_downloaded_bytes": sum(r["bytes"] for r in metadata),
          "header_downloaded_bytes": headers["stored_header_bytes"], "header_HTTP_range_requests": 302,
          "waveform_payload_downloaded_bytes": 0, "scoring_annotation_downloaded_bytes": 0,
          "target_waveform_value_reads": 0, "target_scoring_outcome_reads": 0,
          "new_source_waveform_reads": 0, "SC_VAL_reads": 0, "SC_RESERVED_reads": 0,
          "other_external_dataset_reads": 0, "configured_private_root": True,
          "known_repository_local_HMC_copy_found": False, "license": "CC BY 4.0", "DOI": "10.13026/t79q-fr32"})
    write("preprocessing_receipt.json", {"status": "GENERATED_VALIDATION_PASSED_REAL_WAVEFORMS_NOT_OPENED",
          "protocol_sha256": digest(protocol), "contract": protocol["signal"],
          "source_normalization_authenticated": True, "target_normalizer_fits": 0,
          "generated_full_input_source_behavior_exact": True, "historical_development_rescored": False,
          "whole_HMC_EDF_checksums_verified": False, "HMC_waveform_conversion_executed": False})
    write("channel_contract.json", {"primary": "C4/M1", "secondary": "F4/M1",
          "source": "Fpz-Cz", "aliases": protocol["channel_label_aliases"],
          "actual_header_coverage": headers["channels"], "montage_correction": False,
          "initial_freeze_commit": "933bcba", "pre_waveform_alias_refinement_commit": "d7947ef",
          "refinement_reason": "EDF standard electrode-separator spelling verified in headers; no waveform or score exposure"})
    for name in ("prediction_seal.json", "scoring_receipt.json"):
        write(name, {**blocked, "seal_sha256": None, "prediction_files": 0, "labels_opened": False})
    write("primary_results.json", {**blocked, "decision": "PRIMARY_EVALUATION_BLOCKED", "m0_macro_f1": None,
          "m2_macro_f1": None, "effect": None, "interval_95": None, "success_threshold": .02,
          "threshold_passed": None, "severe_collapse": None})
    write("per_seed_results.json", {**blocked, "seeds": {"123": None, "456": None}})
    for name in ("stage_metrics.json", "calibration_metrics.json", "representation_shift.json", "uncertainty.json"):
        write(name, {**blocked, "result": None, "clinical_finding": "NOT_MEASURED"})
    write("secondary_channel_results.json", {**blocked, "channel": "F4/M1", "analysis": "secondary sensitivity", "result": None})
    write("scientific_gates.json", {"decision": "PRIMARY_EVALUATION_BLOCKED", "research_recommendation": "BLOCKED",
          "M0_decision_unchanged": "MPI_FEASIBILITY_NOT_ESTABLISHED", "all_four_checkpoint_hashes": True,
          "generated_full_input_behavior_reproduced": True, "preprocessing_generated_tests": True,
          "official_metadata_verified": True, "official_source_full_files_verified": False,
          "disk_preflight": preflight, "predictions_sealed": False, "labels_opened": False,
          "primary_gain": None, "both_seed_directions": None, "CI_lower_above_zero": None,
          "zero_scientific_optimizer_updates": True, "zero_target_fits": True,
          "not_a_negative_MPI_efficacy_result": True})
    implementation = {p.relative_to(REPO).as_posix(): file_sha256(p)
                      for p in sorted((REPO/"src/mist_sleep/external").glob("*.py"))}
    private_bytes = sum(p.stat().st_size for p in PRIVATE.rglob("*") if p.is_file())
    resource = {"scientific_training_trajectories": 0, "scientific_optimizer_updates": 0,
                "target_adaptation_updates": 0, "target_inference_epochs": 0,
                "target_waveform_payload_bytes": 0, "source_waveform_reads": 0,
                "metadata_bytes": sum(r["bytes"] for r in metadata), "header_bytes": headers["stored_header_bytes"],
                "source_download_bytes": sum(r["bytes"] for r in metadata)+headers["stored_header_bytes"],
                "private_phase_disk_bytes_at_snapshot": private_bytes, "disk_preflight": preflight,
                "generated_validation": validation, "CPU_regression": cpu,
                "full_CPU_regression_observed_wall_seconds_by_run": [20.29, 20.28, 20.40],
                "cumulative_regression_process_CPU_seconds": None,
                "generated_GPU_inference_test": {"passed": 1, "wall_seconds_upper_bound": 1.17,
                    "device_active_seconds": None, "optimizer_updates": 0,
                    "limitation": "test wall time includes setup and CPU comparison; no device timer used"},
                "test_inference_calls_outside_benchmark": None,
                "accounting_limit": "448 generated benchmark epoch forwards: 384 guarded plus 64 reference; total generated regression forward/optimizer operations belong to inherited fixture tests and were not metered per operation",
                "initial_inspection_limit": "checkpoint metadata initially inspected under Torch 2.5.1 with weights_only=False after byte hashing; formal audit and all source-model validation reloaded with inherited restricted loader under Torch 2.10 CPU; no target inference used exploratory loads"}
    write("resource_receipts.json", resource)
    write("verification.json", {"CPU": cpu, "M1_generated_contract_tests": 52, "M1_CUDA_inference_passed": 1,
          "historical_source_implementation_unchanged": True, "implementation_sha256": implementation,
          "real_HMC_predictions_or_outcomes": False})
    work = [
        ("COMPLETE", "Initialized exact M0 commit and M1 branch", "Git ancestry", "History preserved", "No scientific inference"),
        ("COMPLETE", "Read all 13 proposal text pages and M0 reports", "Foundation report", "Representation consistency differs from transfer efficacy", "PDF layout binary absent"),
        ("COMPLETE", "Preserved M0 failed +0.02 readiness decision", "M0 scientific_gates.json unchanged", "M1 is separately authorized falsification", "M0 gate is not revised"),
        ("COMPLETE", "Authenticated four checkpoint bytes and restricted-loaded states", "checkpoint_receipt.json", "All exact final identities match", "Generated forward reproduction; no development waveform replay"),
        ("COMPLETE", "Implemented inference guard and state digest", "test_frozen_inference_state_and_prediction_order", "Optimizer/backward/state mutation rejected", "Guard is dedicated-process; arbitrary external Python is outside scope"),
        ("COMPLETE", "Committed protocol and pre-waveform electrode label refinement", "933bcba; d7947ef", "Models/channel/metric/threshold fixed before waveform values", "Initial slash-only matcher preserved in audit"),
        ("COMPLETE", "Verified official metadata, DOI and CC BY 4.0", "data_access_receipt.json", "Official v1.1 manifest authenticated", "Waveform payload not acquired"),
        ("COMPLETE_GATE_FAILED", "Configured private root and measured persistent capacity", "resource_receipts.json", "7.06 GB available; 17.7 GB estimated need including reserve", "Full acquisition blocked"),
        ("IMPLEMENTED_BLOCKED", "Resumable pinned-manifest official-source acquisition", "hmc_manifest.py; capacity control test", "No bulk transfer launched", "Cannot acquire full cohort at this destination"),
        ("COMPLETE", "Enumerated official 151 records and 453 source triplets", "source_manifest.json", "Removed recordings excluded", "Eligibility is not yet known"),
        ("BLOCKED", "Verified metadata SHA256; full-file checker implemented", "hmc_manifest.py; metadata hashes", "Whole waveform/scoring files not authenticated", "No full files present"),
        ("COMPLETE_WITH_LIMITATION", "Reviewed aggregate-only participant metadata", "Source and license report", "Use recording clusters; no participant independence claim", "Repeated-person linkage unverified"),
        ("COMPLETE_HEADER_ONLY", "Audited 151 actual EDF headers over exact HTTP ranges", "header_audit.json", "Both EEG derivations at 256 Hz/uV in all headers", "Header hashes are not whole-EDF authentication"),
        ("PARTIAL_BLOCKED", "Read EDF+ standard; strict generated CSV contract", "hmc_scoring.py; geometry tests", "No scoring outcomes inspected", "Real HMC text schema unverified; fails closed"),
        ("COMPLETE_GENERATED", "Implemented explicit physical calibration and source units", "signal_contract.py; calibration tests", "Generated scaling correct", "No actual waveform converted"),
        ("COMPLETE_GENERATED", "Implemented explicit FIR polyphase 25/64 resampling", "Inband/alias/impulse tests", "Generated passband and timing verified", "Clinical compatibility not inferred from tones"),
        ("COMPLETE_GENERATED", "Preserved physical t=0 and complete epoch grid", "geometry tests; alignment report", "No record boundary bridging or label trimming", "Real annotation alignment blocked"),
        ("COMPLETE_GENERATED", "Waveform-only loader disables annotation decoding", "test_generated_edf_blind_reader", "Generated EDF annotations inaccessible during prediction", "Full cohort prediction not executed"),
        ("COMPLETE_GENERATED", "Generated EDF through full-input frozen model", "test_generated_edf_blind_reader; generated_validation.json", "Finite ordered output, immutable model", "No clinical performance claim"),
        ("COMPLETE_GENERATED", "Benchmarked four authenticated models on generated input", "generated_validation.json", "Compute estimate within CPU ceiling", "Excludes real I/O; disk fails"),
        ("BLOCKED", "Blind entrypoint implemented; target inference not launched", "runner.py; primary_results.json", "Zero target predictions", "Insufficient disk"),
        ("BLOCKED", "Exclusive private seal and public hash receipt implemented", "seal tamper/one-time tests", "No real prediction seal exists", "Blind cohort inference never ran"),
        ("BLOCKED", "Strict sealed one-time scoring gate implemented", "label access and timing tests", "Zero original labels opened", "No prediction seal; real text schema also unverified"),
        ("BLOCKED", "Implemented fixed-five primary contrast", "stage_metrics.py; paired_statistics.py", "Primary effect is null, not zero", "No observed outcomes"),
        ("BLOCKED", "Implemented stage confusion/F1/recall and calibration summaries", "stage_metrics.py tests", "All target stage results unmeasured", "No labels or logits"),
        ("BLOCKED", "Implemented evaluation-only NLL/Brier/ECE/confidence", "confidence=1 calibration control", "No calibration fitted", "Adequacy unmeasured"),
        ("BLOCKED", "Both seeds fixed in paired analysis", "paired bootstrap tests", "Seed direction unmeasured", "No model-paired recording scores"),
        ("BLOCKED", "F4/M1 declared secondary; same inference contract", "channel_contract.json", "No channel selection", "No secondary predictions"),
        ("BLOCKED", "Implemented descriptive embedding/occupancy summary", "representation_audit.py", "No external mechanism result", "View stability, stage conditioning and source-distance analysis not executed"),
        ("BLOCKED", "Registered negative transfer, class collapse and confidence failures", "scientific_gates.json", "No observed failure pattern to interpret", "No clinical outcome"),
        ("BLOCKED", "Implemented recording/verified-participant clustered bootstrap", "paired unit and invalid grouping tests", "Seeds never count as patients", "No real paired units"),
        ("COMPLETE_GENERATED", "Generated integrity, timing, identity and label-shuffle controls", "falsification_controls.json; test_external_m1.py", "Analytical controls behave as expected", "Engineering evidence only"),
        ("COMPLETE_BLOCKED_FINDING", "Distinguished blocked evaluation from negative efficacy", "External results report", "PRIMARY_EVALUATION_BLOCKED", "MPI benefit or failure not measured"),
        ("COMPLETE_BLOCKED_FINDING", "Reassessed original claims against actual evidence", "Method-paper implications report", "No new transfer support", "Multiple-target, label efficiency and clinical semantics untested"),
        ("COMPLETE", "Issued BLOCKED research recommendation", "Scientific decision report", "No GO, HOLD or PIVOT efficacy inference warranted", "Storage and integrity gates prevent the test"),
        ("COMPLETE", "Preserved inherited CPU suite; added generated M1 controls", "verification.json; GitHub CI", "All local CPU and inference-only GPU checks pass", "Tests cannot establish transfer"),
        ("COMPLETE", "Produced foundation, source, model and scientific reports", "docs/mist_sleep/PHASE_M1_*", "Null outcomes and limitations explicit", "Clinical reports explain missing evidence"),
        ("COMPLETE_WITH_LIMITATION", "Generated metadata coverage, tone, cost and capacity figures", "figures/", "Only actual metadata and generated validation plotted", "Requested clinical result figures blocked, not fabricated"),
        ("COMPLETE", "Built public artifact manifest and resource accounting", "publication_manifest.json; resource_receipts.json", "No patient-linked arrays or checkpoint bytes staged", "Unmetered fixture operation counts explicitly null"),
        ("DELIVERY_PENDING", "Prepared branch and draft PR delivery", "DASHBOARD.md; final GitHub delivery receipt", "Stop after M1; do not merge", "Exact final-head CI verification occurs after push"),
    ]
    delivery_path = PUBLIC / "delivery_receipt.json"
    delivery = json.loads(delivery_path.read_text()) if delivery_path.exists() else None
    if delivery:
        work[39] = ("DELIVERED", "Pushed M1 branch and opened draft PR against exact M0 branch",
                    delivery["draft_PR"], "Scientific result remains blocked; stop after M1",
                    "Exact final-head CI evidence is recorded by GitHub check runs after publication, avoiding a self-referential commit hash")
    entries = []
    for i, (status, implementation_text, evidence, finding, limitation) in enumerate(work, 1):
        entries.append({"id": f"WP{i:02}", "status": status, "actual_implementation": implementation_text,
                        "tests": "verification.json and generated M1 suite; outcome analyses not run where blocked",
                        "evidence": evidence, "scientific_finding": finding, "limitation": limitation,
                        "resource_use": "metadata/header or generated validation only; target inference 0; see resource_receipts.json",
                        "blockers": ["INSUFFICIENT_PERSISTENT_DISK"] if "BLOCKED" in status or status in ("IMPLEMENTED_BLOCKED", "COMPLETE_GATE_FAILED") else []})
    assert len(entries) == 40
    write("progress.json", {"phase": "M1", "scientific_status": "PRIMARY_EVALUATION_BLOCKED", "work_packages": entries})
    common = "The primary outcome is **`PRIMARY_EVALUATION_BLOCKED`**. No HMC waveform payload, original scoring outcome, target prediction or clinical metric was read or produced. All clinical values are null. The Phase M0 decision remains `MPI_FEASIBILITY_NOT_ESTABLISHED`."
    document("PHASE_M1_RESEARCH_FOUNDATION_V1.md", "Phase M1 research foundation", f"""{common}

The original 13-page proposal distinguishes nuisance-invariant representation learning from stage-discriminative transfer. It predicts possible failure under strong montage and physiology shifts. Its H2/H4 transfer and stage-preservation claims require independent scored outcomes; stable embeddings alone are insufficient. M0 tested matched M0/M2 architectures, augmentation, prototype initialization and exposure. Its clean +0.00370 and bounded-view +0.00585 gains failed the +0.02 developmental readiness gate. This phase was independently authorized to falsify the existing method; it does not reinterpret that result.

All original text pages, M0 results/decision, future transfer protocol, AttnSleep reproduction audit, source/training/evaluation code and frozen protocol were reviewed. The original PDF layout binary is unavailable; the tracked complete text mirror is the available source. M0 history and all source implementation hashes are preserved. No new trajectories, source-development waveform reads, reserved reads, target fitting or alternate datasets are permitted.""")
    document("PHASE_M1_HMC_SOURCE_AND_LICENSE_V1.md", "HMC source, license and acquisition gate", f"""{common}

The [official HMC v1.1 source](https://physionet.org/content/hmc-sleep-staging/1.1/) identifies open access under [CC BY 4.0](https://physionet.org/content/hmc-sleep-staging/1.1/LICENSE.txt), DOI [10.13026/t79q-fr32](https://doi.org/10.13026/t79q-fr32). Attribution: Alvarez-Estevez D and Rijsman R (2022), Haaglanden Medisch Centrum sleep staging database, version 1.1, PhysioNet. Associated study: Alvarez-Estevez D and Rijsman RM (2021), Inter-database validation of a deep learning approach for automatic sleep scoring, PLoS ONE 16(8):e0256111. See the dataset page for its requested PhysioNet platform citation.

The original official RECORDS, SHA256SUMS.txt, LICENSE.txt and aggregate metadata total {sum(r['bytes'] for r in metadata):,} downloaded bytes. Their actual byte hashes are in data_access_receipt.json. All four metadata files were fetched from the version-pinned PhysioNet root; RECORDS and license hashes match its checksum list. It lists 151 waveform identities with matching EDF+/text files (453 files); SN014/SN064/SN135 are absent. File continuity was not assumed. No whole waveform or scoring checksum has been verified because no complete source triplet was acquired.

After the protocol commit, 151 EDF headers (347,904 bytes) were fetched using exact HTTP ranges from the official public mirror. They show literal EEG C4-M1 and EEG F4-M1 channels at 256 Hz in uV. All original header bytes and individual receipts remain private. HTTPS header hashes do not authenticate complete EDF files. Header counts do not establish scoring eligibility. Only aggregate participant metadata exists here; anonymous recording identities do not establish unique people. The planned statistical clusters therefore remain recordings, with participant linkage explicitly unknown.

The private HMC_DATA_ROOT is configured in an ignored mode-0600 environment file, and its directory is mode 0700. No local complete HMC copy was found in the known repository experiment directory. Available persistent space was {preflight['free_bytes']:,} bytes against {preflight['required_bytes']:,} estimated source bytes plus {preflight['reserve_bytes']:,} safety reserve. Bulk acquisition was aborted. The 18 GB ceiling was respected. The mounted external disk advertised by the host is inaccessible to this account; it was not used. No files were deleted or moved to free capacity.""")
    document("PHASE_M1_MODEL_AUTHENTICATION_V1.md", "Frozen source model authentication", f"""{common}

All four actual local step-1,200 checkpoint byte hashes exactly match the user-specified M0/M2 seed-123/456 identities. Formal loading uses inherited authenticated snapshots and the patched PyTorch 2.10 restricted weights-only loader, with no custom globals, unsafe fallback or optimizer-state restoration. Strict state shapes, arm, seed, final step, M0 protocol hash, 64-by-64 prototype bank, finiteness and 41,429 parameters pass. See checkpoint_receipt.json.

The ten-person source-fit normalization receipt is authenticated through the existing Phase XIII metadata chain: mean 0.2903136955214651 uV; SD 26.31764815403481 uV. No source waveform was read to recover it. For each authenticated checkpoint, generated [16,1,3000] inputs reproduce the unchanged original MISTSleep forward output exactly under patched CPU inference, with the entire state digest unchanged. Historical development performance was not recomputed. This establishes generated full-input compatibility, not new source or external accuracy.

Audit limitation: an initial exploratory metadata inspection used the older Torch 2.5.1 loader with weights_only=False after checking byte hashes. That inspection did not run target inference. Every formal authentication and source behavior check was subsequently performed through the inherited patched restricted loader. GPU correctness used a generated model only and did not deserialize source checkpoints.""")
    document("PHASE_M1_CROSS_MONTAGE_CONTRACT_V1.md", "Clinical and cross-montage contract", f"""{common}

The source is SC Fpz-Cz at 100 Hz; primary HMC is C4/M1 and secondary HMC is F4/M1 at 256 Hz. This combines cohort, clinical physiology, acquisition system, reference and electrode-location changes. Resampling is not montage correction. No missing electrode is synthesized, no polarity is selected after errors and no target mapping is fitted.

Physical EDF calibration precedes conversion to uV. Continuous per-record 25/64 rational resampling uses the explicit symmetric 1,281-tap FIR, fixed Kaiser beta 5, cutoff 1/64, float64 arithmetic and zero padding at physical record ends. The grid begins at original t=0; only the incomplete trailing epoch is discarded. Source-fit mean/SD are used directly, without target moments, BatchNorm adaptation or stage-conditioned transforms. Generated tone/impulse/anti-alias/constant-offset controls pass. Actual HMC conversion remains unexecuted.

All headers have both intended electrode pairs at 256 Hz/uV, but whole-file and annotation verification remain blocked. Provider lights-off/on clipping differs from SC complete physical recording treatment. Neither success nor failure on this particular shift would isolate its individual causal components.""")
    document("PHASE_M1_EDF_SCORING_ALIGNMENT_V1.md", "EDF scoring and physical alignment", f"""{common}

The loader disables pyEDFlib annotation reading in Stage A and permits only waveform file identities. Generated EDF+ tests forbid annotation access and verify complete 30-second/3,000-sample ordering. A strict one-time scoring claim is persisted only after content verification of the sealed predictions and committed public seal receipt. Subsequent attempts reject; interrupted scoring stays exposed.

The [EDF+ standard texts](https://www.edfplus.info/specs/edftexts.html) govern explicit Sleep stage W/N1/N2/N3/R mappings. Unknown/movement intervals map only as registered. Nonfinite, negative, fractional-grid, overlapping, repeated, contradictory and out-of-record stage intervals reject. Gaps remain unscored. Absolute waveform/scoring start times and complete EDF+/text event agreement are required; no label-dependent repair is permitted.

The real HMC scoring text schema is not verified. The implementation supports the explicitly named generated CSV fixture schema and fails closed on other formats. No real annotation has been opened to extend that parser. Full original-file verification and schema compatibility are outstanding gates, not passing tests or established clinical alignment.""")
    document("PHASE_M1_BLIND_INFERENCE_RECEIPT_V1.md", "Blind inference status", f"""{common}

No real blind prediction run began. There is no real seal, logits array, prediction vector or scoring claim. prediction_seal.json explicitly records NOT_RUN_BLOCKED and a null seal digest. Synthetic tampering, duplicate scoring and pre-seal annotation-access controls pass. The guard forbids optimizer construction/backward and verifies unchanged model parameters and buffers, eval mode and frozen gradients. This engineering evidence is not external evaluation.""")
    document("PHASE_M1_EXTERNAL_TRANSFER_RESULTS_V1.md", "External transfer result: blocked", f"""{common}

| Quantity | Observed evidence |
|---|---|
| Source checkpoint identities | 4/4 authenticated |
| Official manifest | 151 recordings / 453 source files |
| Header audit | 151/151; primary and secondary derivations present |
| Fully authenticated source triplets | 0 |
| Eligible scored recordings | Unknown |
| Evaluated recordings | 0 |
| Known unique participants | Unknown |
| M0 / M2 external Macro-F1 | Not measured / not measured |
| M2-minus-M0 effect and 95% CI | Not measured |
| Seed consistency / +0.02 gate | Not evaluated |

Disk preflight failed before waveform access. Unsupported real text-schema handling would also block any unblinding until verified. This is a resource/integrity blocker, not negative transfer, efficacy equivalence or severe clinical class collapse. No second dataset, new model, rescue scoring or additional seed was started.""")
    document("PHASE_M1_STAGE_AND_CALIBRATION_ANALYSIS_V1.md", "Stage and calibration evidence unavailable", f"""{common}

Wake/N1/N2/N3/REM recall and F1, class presence/absence counts, confusion, balanced accuracy, pooled F1, NLL, Brier, ECE, entropy, margins and wrong-confidence distributions are unmeasured. No classes may be said to improve, deteriorate or collapse. Generated analytical tests verify absent-class zero-F1, fixed stage order, shuffled/cyclic labels and confidence=1 ECE boundaries. No target calibration was fitted; passing these tests does not establish calibration adequacy.""")
    document("PHASE_M1_REPRESENTATION_SHIFT_ANALYSIS_V1.md", "Representation shift evidence unavailable", f"""{common}

No HMC embedding, prototype distribution, switching rate or stability result exists. The summary implementation provides magnitude, feature variance, mean-assignment entropy, effective occupancy and dead-unit counts. Source distances are explicitly unavailable without authenticated frozen source-fit reference moments. Target view stability and stage-conditioned analyses were not executed. M0's previously reported bounded-view stability improvement remains historical source evidence only; it cannot supply the missing transfer endpoint or validate clinical prototype semantics.""")
    document("PHASE_M1_STATISTICAL_ANALYSIS_V1.md", "Paired uncertainty contract and missing outcomes", f"""{common}

The endpoint is the equal-recording fixed-five Macro-F1 contrast M2-minus-M0, averaged over seeds 123/456. The 10,000-draw percentile bootstrap uses complete model-paired recording units and keeps the two seeds together; seed 20261023 is fixed. Epochs and training seeds never increase the patient count. If independently verified participant linkage exists before scoring, cluster on people while retaining the equal-recording endpoint. Current linkage is unknown. No real denominator, effect, confidence interval, per-seed contrast or stage missingness estimate can be calculated.

Generated controls recover a constant +0.03 paired effect and degenerate [+0.03,+0.03] interval. Duplicate recording IDs, wrong seed geometry, invalid grouping, single-cluster inference and changed bootstrap settings reject. These are analytical checks on invented units, not measured clinical uncertainty.""")
    document("PHASE_M1_METHOD_PAPER_IMPLICATIONS_V1.md", "Method-paper implications", f"""{common}

This phase adds authenticated model/source identities and engineering compatibility evidence. It adds no support for cross-study transfer, montage robustness, preserved external stage discrimination, external prototype reuse, label efficiency or multiple-target generalization. Four models have identical 41,429-parameter inference graphs; generated throughput is useful for feasibility but excludes clinical I/O and does not establish deployment efficiency.

No second independent target is scientifically justified by an unobserved result. The separately authorized HMC test remains blocked on storage/full-file/scoring-schema gates. This is not evidence warranting a method PIVOT, nor a positive GO. The original proposal's broader comparisons, multiple protected targets, clinical prototype naming and low-label adaptation remain untested. No new method or training run was started.""")
    answers = [
        "Yes: all four exact byte hashes, final-step metadata and architecture authenticated (checkpoint_receipt.json).",
        "Official v1.1 metadata and 151 headers obtained; the complete EEG dataset was not obtained.",
        "Official metadata verified; whole waveform/scoring files and original annotation content not verified.",
        "Unknown eligibility; 151 manifest entries and compatible headers; zero recordings evaluated.",
        "Unique-person/repeat linkage unavailable; recordings retained as clusters; no invented people count.",
        "Yes in all 151 actual headers: literal EEG C4-M1, 256 Hz, uV; full waveform integrity still unavailable.",
        "Generated passband, alias suppression, impulse timing and epoch geometry pass; no real conversion ran.",
        "No predictions were produced or sealed; zero original scoring outcomes opened.",
        "Authenticated source-fit scaler fixed; target normalization fits zero; target application not run.",
        "Not measured: M0 external Macro-F1 is null.", "Not measured: M2 external Macro-F1 is null.",
        "Not measured: primary effect is null.", "Not measured: both seed effects are null.",
        "Not measured: paired confidence interval is null.", "Not evaluated: +0.02 criterion neither passes nor fails.",
        "No stage improvement observed because no clinical outcome was measured.",
        "No stage deterioration observed because no clinical outcome was measured.",
        "N1 recall is unmeasured.", "REM recall is unmeasured.",
        "Target class collapse is unmeasured; no severe-failure flag inferred from missing data.",
        "Calibration adequacy is unmeasured; no calibration fitted.",
        "HMC representation stability is unmeasured; historical M0 stability does not answer this question.",
        "HMC prototype informativeness/degeneracy is unmeasured.",
        "F4/M1 is present in all headers but its secondary outcome is unmeasured.",
        "No difference exists to attribute. Matched source arms would isolate their MPI training contrast conditional on fixed seeds and shared architecture/exposure.",
        "Montage/electrode reference, clinical cohort, acquisition hardware/prefilters and provider clipping remain combined.",
        "No new independent transfer support was obtained.",
        "Transfer, clinical stage preservation, multiple-target generalization, label efficiency and clinical prototype semantics remain unsupported.",
        "No second target is justified by this unobserved outcome; do not automatically initiate one.",
        "BLOCKED: stop Phase M1. Preserve M0's failed developmental gate and do not infer method efficacy or start tuning.",
    ]
    table = "\n".join(f"| {i} | {answer} |" for i, answer in enumerate(answers,1))
    document("PHASE_M1_SCIENTIFIC_DECISION_V1.md", "Phase M1 decision: PRIMARY_EVALUATION_BLOCKED", f"""{common}

Recommendation: **BLOCKED — evaluation could not be conducted**. Approximately 7.06 GB persistent capacity fails the estimated 15.7 GB collection plus 2 GB safety reserve. All model identities and header metadata pass; full source integrity, real scoring geometry, predictions and statistical outcomes remain missing. This is not a negative or positive MPI efficacy result. No automatic continuation, training, additional dataset or merge is authorized by this result.

The table answers the assignment's thirty final scientific questions in their original order. All missing outcomes remain explicitly unmeasured.

| Question | Evidence-based answer |
|---|---|
{table}""")
    figures = PUBLIC/"figures"; figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(7,4)); ax.bar(["Official records", "Headers audited", "Whole triplets", "Evaluated"], [151,151,0,0]); ax.set_ylabel("Recording count"); ax.set_title("Actual M1 coverage; eligibility remains unknown"); fig.tight_layout(); fig.savefig(figures/"01_actual_coverage.png",dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4)); ax.bar(["Available", "Collection estimate", "Estimate + reserve"], [preflight["free_bytes"]/1e9,15.7,17.7]); ax.set_ylabel("GB (decimal)"); ax.set_title("Persistent disk gate failed before bulk acquisition"); fig.tight_layout(); fig.savefig(figures/"02_disk_gate.png",dpi=150); plt.close(fig)
    t=np.arange(7680)/256; x=20*np.sin(2*np.pi*13*t); y,_=physical_epochs(x,unit="uV",sample_rate=256)
    restored=y.ravel()*protocol["source_normalization"]["std_uV"]+protocol["source_normalization"]["mean_uV"]
    fig, ax=plt.subplots(figsize=(7,4)); ax.plot(t[256:384],x[256:384],label="Generated 256 Hz"); ax.plot(np.arange(100,150)/100,restored[100:150],".",label="Fixed 100 Hz"); ax.set_xlabel("Physical seconds"); ax.set_ylabel("uV"); ax.set_title("Generated 13 Hz conversion; no clinical waveform"); ax.legend(); fig.tight_layout(); fig.savefig(figures/"03_generated_signal_validation.png",dpi=150); plt.close(fig)
    fig, ax=plt.subplots(figsize=(7,4)); ax.bar([r["model"] for r in validation["models"]],[r["epochs_per_second"] for r in validation["models"]]); ax.set_ylabel("Generated epochs / second"); ax.set_title("Measured CPU forward cost; excludes clinical I/O"); fig.tight_layout(); fig.savefig(figures/"04_generated_cpu_throughput.png",dpi=150); plt.close(fig)
    dashboard = f"""# MIST-Sleep Phase M1 dashboard

**Scientific decision: `PRIMARY_EVALUATION_BLOCKED`. Research recommendation: BLOCKED.**

{common}

- Four source checkpoints authenticated. Original full-input forward behavior reproduced on generated inputs.
- Official v1.1 manifest: 151 recordings, 453 source files; 151 actual headers audited.
- C4/M1 and F4/M1 electrode pairs present in all headers at 256 Hz/uV.
- Persistent disk: {preflight['free_bytes']/1e9:.2f} GB free, 17.7 GB estimated need with reserve. No bulk acquisition.
- Target waveforms, labels, predictions and optimizer updates: zero. Eligible cohort and all clinical metrics: unknown/null.
- CPU regression: {cpu['passed']} passed, {cpu['skipped']} skipped; M1 generated GPU inference-only correctness: 1 passed.
- Forty work packages have individual status, implementation, tests, evidence, finding, limitation, resource and blocker records in progress.json.
- Four figures show actual metadata/capacity and explicitly generated signal/cost validation. Clinical result figures were not fabricated.

See [scientific decision](../../../docs/mist_sleep/PHASE_M1_SCIENTIFIC_DECISION_V1.md), [source audit](../../../docs/mist_sleep/PHASE_M1_HMC_SOURCE_AND_LICENSE_V1.md), [registered protocol](../../../docs/mist_sleep/PHASE_M1_FROZEN_PROTOCOL_V1.md) and [verification](verification.json).

Initial freeze: 933bcba. Header-only label syntax refinement: d7947ef; primary electrode pair unchanged. Original strict-slash audit preserved. Whole EDF checksums and real HMC text schema remain unverified. No external result, clinical deployment, label efficiency or multiple-target generalization claim is supported. Stop after M1; do not merge automatically.
"""
    if delivery:
        dashboard += f"\nDraft delivery: [PR #20]({delivery['draft_PR']}). Exact final-head CI is verified through its GitHub checks after the last publication commit; local receipts cannot contain their own final commit SHA.\n"
    (PUBLIC/"DASHBOARD.md").write_text(dashboard)
    candidates = sorted([*PUBLIC.rglob("*"), *DOCS.glob("PHASE_M1_*.md"),
                         *(REPO/"src/mist_sleep/external").glob("*.py"),
                         *(REPO/"scripts").glob("mist_sleep_m1_*.py"),
                         REPO/"tests/mist_sleep/test_external_m1.py", REPO/".github/workflows/ci.yml", REPO/"pyproject.toml"])
    safe = {p.relative_to(REPO).as_posix(): file_sha256(p) for p in candidates
            if p.is_file() and p.name != "publication_manifest.json"}
    write("publication_manifest.json", {"status": "PUBLIC_AGGREGATES_AND_GENERATED_FIXTURES_ONLY", "files_sha256": safe,
          "self_hash_excluded": True, "raw_EDF_in_Git": False, "private_labels_in_Git": False,
          "individual_prediction_vectors_in_Git": False, "checkpoint_bytes_in_Git": False,
          "patient_linked_result_rows_in_Git": False, "private_absolute_paths_in_public_reports": False,
          "target_fitted_normalizers": 0, "no_source_model_modification": True,
          "clinical_figures_produced": 0, "generated_or_metadata_figures_produced": 4})
    print(json.dumps({"decision":"PRIMARY_EVALUATION_BLOCKED", "work_packages":len(entries), "CPU":cpu}))


if __name__ == "__main__": main()
