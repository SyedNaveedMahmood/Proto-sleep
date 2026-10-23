#!/usr/bin/env python3
"""Reproduce original full-input forward behavior and measure generated CPU cost.

No raw source or target waveforms, source labels, fitting or optimizer construction.
"""
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch

from mist_cove.phase11b.security import authenticated_snapshot, restricted_load
from mist_sleep.core import MISTSleep
from mist_sleep.external.source_checkpoint import load_source
from mist_sleep.external.protocol import CHECKPOINTS, digest
from mist_sleep.external.blind_inference import predict, state_digest
from mist_sleep.external.signal_contract import physical_epochs
from mist_sleep.external.negative_controls import generated_controls


def main():
    repo = Path(__file__).resolve().parents[1]
    public = repo / "reports/mist_sleep/phase_m1_hmc_v1"
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    epochs = np.random.default_rng(20261023).normal(size=(16, 1, 3000)).astype("float32")
    rows = []
    start_cpu, start_wall = time.process_time(), time.monotonic()
    for key, expected in CHECKPOINTS.items():
        arm, seed_text = key.split("_")
        models_root = repo / "mist_transfer_runs/mist_sleep/method_recovery_v1/models"
        model, _ = load_source(models_root, arm, int(seed_text))
        path = models_root / key / "step_1200.pt"
        with authenticated_snapshot(path, expected, allowlist=[path]) as snapshot:
            payload = restricted_load(snapshot)
        reference = MISTSleep().eval().requires_grad_(False)
        reference.load_state_dict(payload["model"], strict=True)
        before = state_digest(model)
        with torch.inference_mode(): direct = reference(torch.from_numpy(epochs))["logits"].numpy()
        result = predict(model, epochs)
        if not np.array_equal(result["logits"], direct) or state_digest(model) != before:
            raise RuntimeError("original full-input forward behavior was not reproduced")
        elapsed = []
        for _ in range(5):
            t = time.monotonic(); predict(model, epochs); elapsed.append(time.monotonic()-t)
        rows.append({"model": key, "source_sha256": expected, "generated_full_input_exact_match": True,
                     "state_unchanged": True, "epochs_per_second": 16/float(np.median(elapsed))})
    # Known-tone transfer and temporal impulse checks are separate from class predictions.
    t = np.arange(7680*2)/256
    waveform = 20*np.sin(2*np.pi*13*t)
    begin = time.monotonic(); converted, onsets = physical_epochs(waveform, unit="uV", sample_rate=256)
    signal_seconds = time.monotonic()-begin
    receipt = {"kind": "GENERATED_ONLY", "models": rows, "input_shape": list(epochs.shape),
               "epoch_inferences": 4*16*7, "wrapped_inference_epochs": 4*16*6,
               "reference_inference_epochs": 4*16, "optimizer_updates": 0, "target_waveform_reads": 0,
               "source_waveform_reads": 0, "CPU_seconds": time.process_time()-start_cpu,
               "wall_seconds": time.monotonic()-start_wall,
               "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "threads": 2},
               "signal_conversion_seconds_60s_generated": signal_seconds,
               "signal_shape": list(converted.shape), "signal_onsets": onsets.tolist(),
               "estimated_cohort_epochs_per_channel": 151*7.6*120,
               "estimated_8_model_channel_pass_seconds_excluding_IO": (151*7.6*120)*2*sum(1/r["epochs_per_second"] for r in rows),
               "estimate_limitations": "uses published mean TIB, generated CPU forward timing; excludes downloads, real EDF I/O, resampling, representation-view analyses and actual eligibility",
               "historical_development_metrics_recomputed": False}
    (public / "generated_validation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    controls = generated_controls()
    (public / "falsification_controls.json").write_text(json.dumps(controls, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__": main()
