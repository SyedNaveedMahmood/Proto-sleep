# Phase IV synthetic CUDA completion addendum

This addendum supersedes only the GPU-blocked status in the [Phase IV results](PHASE_IV_RESULTS_V2.md) and original correctness receipt. Those artifacts and both blocked attempts remain unchanged. The negative H4.1/H4.2/H4.3 findings, trained synthetic results and recommendation against a real EEG launch remain unchanged.

The competing GPU queue had stopped before this attempt. No process was terminated and no administrator credential was used or saved. The frozen checker found no competing compute process before allocating CUDA memory. All inputs were generated random tensors or mock recording fixtures; no real TRAIN/VAL EEG, source descriptor, RESERVED participant, Sleep-Telemetry, SHHS or fitted real artifact was accessed.

## Measured environment and identity

Measurement checkout: `f7e43238dd913ee257766a82ee961c51277d9e97`. The source implementation inventory remains `e8f84002bd71a57259e596a547b101b5733e2d7a755b45bed706a830ff59142d`, identical to the completed synthetic comparison. Every inventory file, original public artifact hash and corrected preregistration hash was verified before publication. No model, selection rule, config, threshold or training budget changed.

Actual device: NVIDIA GeForce RTX 4080 SUPER, NVIDIA-reported total 16,376 MiB, free 15,881 MiB before allocation; driver 580.95.05. Imported Python 3.12.9, PyTorch 2.5.1 with CUDA 12.1, cuDNN 90100 and NumPy 1.26.4. Installed NumPy metadata also reports 1.26.4. Two CPU threads, deterministic FP32, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The existing reconciled CUDA environment was reused without modifying it.

## Results

All six methods passed generated batch-16 forward/backward with finite gradients, additive reconciliation within 1e-5, and exact interrupted-versus-uninterrupted two-epoch adapter training. Each exact-resume comparison covers model, optimizer, Python/NumPy/Torch RNG states, best/early-stop state and semantic validation history. Each completed logical trajectory contains six optimizer updates. Checkpoints and RNG artifacts remain private and ignored.

| Method | Five forward/backward passes (s) | Peak allocated (MiB) | Peak reserved (MiB) | Exact CUDA resume |
|---|---:|---:|---:|---|
| M0 | 0.220058 | 65.799 | 68.000 | PASS |
| M1 | 0.185072 | 69.063 | 90.000 | PASS |
| M2 | 0.013345 | 69.082 | 90.000 | PASS |
| M3 | 0.012750 | 69.082 | 90.000 | PASS |
| M2_rms | 0.013273 | 69.081 | 90.000 | PASS |
| M1_phase | 0.054387 | 69.241 | 90.000 | PASS |

These are bounded smoke measurements from the fixed sequential checker, including cold-start/kernel initialization effects, rather than a warmed matched-throughput benchmark. Method order, live tensors and allocator caches affect process peaks. Allocator figures exclude CUDA context/driver allocations and are measured around the batch-16 smoke, not the subsequent resume probes or full source training. The differences do not establish relative model speed or real EEG runtime feasibility. No CPU fallback occurred.

The eleven previously CPU-skipped CUDA regression cases also ran on generated/mock inputs: **11 passed, 0 failed, 0 errors, 0 skipped, 504 deselected in 4.29 seconds**. The earlier complete CPU result remains **504 passed / 11 CUDA skips**; these are separate executions, not a newly claimed single 515-test run. Together the executed cases cover every collected test. Six new Phase IV exact CUDA resume probes are separate from those eleven pytest cases.

Machine-readable device/environment, source inventory, per-method resources, exact-resume outcomes and private receipt/JUnit hashes are in [cuda_measurements_v1.json](../../reports/mist_cove/phase4_v1/cuda_measurements_v1.json). Private raw outputs remain under the ignored Phase IV run root. The new publication manifest binds this addendum and receipt to the unchanged original publication manifest.

## Reproduction and remaining gate

Use a CUDA-enabled interpreter with the recorded dependency versions and a free GPU. Set `CUDA_PYTHON` to that interpreter, then choose a new unused private output directory:

```bash
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$CUDA_PYTHON" scripts/check_mist_cove_phase4_cuda.py \
  --output mist_transfer_runs/mist_cove/phase4_v1/NEW_UNUSED_CUDA_OUTPUT
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$CUDA_PYTHON" -m pytest tests -k cuda -ra
```

**PASS:** synthetic CUDA engineering readiness for these bounded cases. **FAIL:** the previously frozen scientific success criteria remain failed. **NOT TESTED:** clinical morphology, real source performance of these learned candidates, physiological donor validity, full-data throughput or external transfer. **BLOCKED:** new real source or external experiments lack separate launch authorization. Synthetic CUDA completion supplies neither that authorization nor a real-data budget gate.
