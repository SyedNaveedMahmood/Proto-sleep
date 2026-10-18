#!/usr/bin/env python
"""Synthetic-only CUDA smoke, peak VRAM and exact interrupted-resume probes."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from mist_cove.phase2.adapter import build_adapter, adapter_logits
from mist_cove.phase2.provenance import implementation_snapshot, git_head, environment
from mist_cove.phase2.synthetic_runtime import exact_resume_probe, synthetic_bank
from mist_cove.phase2.training import configure_precision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "mist_transfer_runs/mist_cove/phase2_cuda_v1")
    p.add_argument("--allocator-cap-mib", type=int, default=512,
                   help="bound this synthetic process's allocator; context/driver allocations are additional")
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError("choose a fresh CUDA receipt directory")
    configure_precision("cuda")
    if a.allocator_cap_mib < 128:
        raise ValueError("allocator cap is too small for the declared batch-16 checks")
    free, total = torch.cuda.mem_get_info()
    if free < (a.allocator_cap_mib + 256) * 1024 * 1024:
        raise RuntimeError(f"CUDA BLOCKED: {free / 1024**2:.1f} MiB free after context initialization; requires {a.allocator_cap_mib + 256} MiB; no CPU fallback")
    torch.cuda.set_per_process_memory_fraction(min(a.allocator_cap_mib * 1024 * 1024 / total, 1.))
    torch.set_num_threads(2)
    initial_code = implementation_snapshot()
    started = time.perf_counter()
    results = []
    for method in ("q1", "availability", "background", "rounded_q1", "raw_context", "evidence"):
        print("CUDA synthetic probe: " + method, flush=True)
        bank = synthetic_bank() if method in {"raw_context", "evidence"} else None
        model = build_adapter(method, {"mean_uV": 0., "std_uV": 1.}, bank).cuda()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(16, 1, 3000, device="cuda")
        logits = adapter_logits(model, x)
        if logits.shape != (16, 5) or not torch.isfinite(logits).all():
            raise AssertionError("CUDA finite five-stage smoke failed")
        logits.square().mean().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        smoke_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model, x, logits
        torch.cuda.empty_cache()
        row = exact_resume_probe(a.output_dir / method, method, "cuda")
        row.update(batch16_smoke_peak_allocated_bytes=peak, parameters=smoke_parameters,
                   trainable_parameters=trainable_parameters)
        results.append(row)
    if implementation_snapshot() != initial_code:
        raise RuntimeError("code changed during CUDA checks")
    receipt = {"scope": "SYNTHETIC_ONLY_NOT_SOURCE_AUTHORIZATION", "status": "PASS", "environment": environment("cuda"),
               "implementation": initial_code, "implementation_commit": git_head(), "runs": results,
               "allocator_cap_mib": a.allocator_cap_mib, "free_memory_bytes_at_start": free,
               "resource_limit": "concurrent GPU users are untouched; this is not an exclusive throughput benchmark",
               "wall_seconds": time.perf_counter() - started, "real_data_accessed": False}
    with (a.output_dir / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False); handle.write("\n")
    print("CUDA synthetic checks PASS", flush=True)


if __name__ == "__main__":
    main()
