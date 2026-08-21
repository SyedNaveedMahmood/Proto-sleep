#!/usr/bin/env python3
"""Quick model-size / AMP sanity report without touching the dataset."""
from __future__ import annotations

import os
import time
import torch

from protosleep.config import (
    USE_AMP, NUM_PROTOTYPES, MACRO_SSL_EPOCHS, MACRO_BATCH_SIZE
)
from protosleep.attnsleep import AttnSleepBaseline
from protosleep.prototypes import ProtoAttnSleep
from protosleep.macro import MacroEncoder, MacroStageClassifier, MaskedSequenceAutoencoder


def nparams(model):
    return sum(p.numel() for p in model.parameters())


def main():
    models = {
        "AttnSleepBaseline": AttnSleepBaseline(),
        "ProtoAttnSleep": ProtoAttnSleep(),
        "MacroClassifier48": MacroStageClassifier(MacroEncoder(NUM_PROTOTYPES)),
        "PrototypeMAE48": MaskedSequenceAutoencoder(NUM_PROTOTYPES, NUM_PROTOTYPES, "distribution"),
        "LatentMAE30": MaskedSequenceAutoencoder(30, 30, "continuous"),
    }

    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("AMP:", USE_AMP)
    print()
    for name, model in models.items():
        print(f"{name:22s} {nparams(model):>10,d} params")

    print()
    print("Why macro training is fast:")
    print("  - it trains on cached per-night feature trajectories, not raw EEG")
    print("  - only ~70k parameters in the classifier and ~111k in the MAE")
    print("  - with 35 nights and batch size 2 there are ~18 batches/epoch")
    print(f"  - {MACRO_SSL_EPOCHS} SSL epochs are only ~{18 * MACRO_SSL_EPOCHS} optimizer steps")
    print("  - AMP helps on CUDA, but caching + tiny model + few steps are the main reasons")
    print("  - REUSE_EXISTING can make reruns nearly instant; check [checkpoint-hit]/[cache-hit] messages")


if __name__ == "__main__":
    main()
