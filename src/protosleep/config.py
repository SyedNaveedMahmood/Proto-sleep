from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

PROJECT_NAME = "ProtoMAE-Sleep"
PROJECT_VERSION = "2026-08-repo-v1"

DATA_DIR = os.environ.get("SLEEP_EDF_NPZ_DIR", "./data/sleep-edf-20")
OUTPUT_ROOT = Path(os.environ.get("PROTOMAE_OUT", "./protomae_sleep_runs"))

FOLD_ID = int(os.environ.get("PROTOSLEEP_FOLD", "0"))
RUN_ALL_FOLDS = False
RUN_SELF_TESTS = _env_bool("PROTOSLEEP_SELF_TESTS", True)
SELF_TEST_ONLY = _env_bool("PROTOMAE_SELF_TEST_ONLY", False)

SEED = int(os.environ.get("PROTOSLEEP_SEED", "123"))
NUM_CLASSES = 5
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
EXPECTED_FS = 100
EXPECTED_SAMPLES_PER_EPOCH = 3000
NUM_WORKERS = int(os.environ.get("PROTOSLEEP_NUM_WORKERS", "0"))
PIN_MEMORY = _env_bool("PROTOSLEEP_PIN_MEMORY", True)
USE_AMP = _env_bool("PROTOSLEEP_AMP", True)
GRAD_CLIP_NORM = 5.0

MICRO_BATCH_SIZE = int(os.environ.get("PROTOSLEEP_MICRO_BATCH", "128"))
MICRO_EPOCHS = 100
MICRO_PATIENCE = 15
MICRO_LR = 1e-3
MICRO_WEIGHT_DECAY = 1e-3

NUM_PROTOTYPES = 48
PROTO_DIM = 30
PROTO_TEMPERATURE = 0.10
PROTO_POSITION_SLICE: Optional[Tuple[int, int]] = None
PROTO_BETA_INIT = 0.20

PROTO_TRIALS = [
    dict(name="p0", lambda_commit=0.05, lambda_balance=0.01, lambda_sep=0.01, sep_margin=0.30),
    dict(name="p1", lambda_commit=0.10, lambda_balance=0.01, lambda_sep=0.01, sep_margin=0.30),
    dict(name="p2", lambda_commit=0.05, lambda_balance=0.03, lambda_sep=0.01, sep_margin=0.30),
]
RUN_PROTO_HPARAM_SEARCH = False

ENABLE_MICRO_MASK = False
MICRO_MASK_RATIO = 0.30
MICRO_MASK_SPAN_MIN = 2
MICRO_MASK_SPAN_MAX = 6
MICRO_MASK_HIDDEN = 64
MICRO_MASK_LAYERS = 2
MICRO_MASK_HEADS = 4
MICRO_MASK_LAMBDA = 0.10
MICRO_MASK_WARMUP_EPOCHS = 5

MACRO_BATCH_SIZE = int(os.environ.get("PROTOSLEEP_MACRO_BATCH", "2"))
MACRO_D_MODEL = 64
MACRO_LAYERS = 2
MACRO_HEADS = 4
MACRO_FF = 128
MACRO_DROPOUT = 0.10
MACRO_DECODER_DIM = 64
MACRO_DECODER_LAYERS = 1
MACRO_DECODER_HEADS = 4
MACRO_DECODER_FF = 128

MACRO_SSL_EPOCHS = 50
MACRO_SSL_LR = 1e-3
MACRO_SSL_WEIGHT_DECAY = 1e-4
MACRO_FT_EPOCHS = 100
MACRO_FT_PATIENCE = 15
MACRO_FT_LR = 5e-4
MACRO_FT_WEIGHT_DECAY = 1e-4

MACRO_MASK_RATIO = 0.45
MASK_SPAN_MIN = 2
MASK_SPAN_MAX = 10
TRANSITION_MASK_FRACTION = 0.30

USE_TRANSITION_GRAPH_LOSS = False
TRANSITION_GRAPH_LAMBDA = 0.10
TRANSITION_GRAPH_SMOOTHING = 1e-3

RUN_GEOMETRY_MAE_CONTROL = True
PROTO_GEOMETRY_LAMBDA = 0.25
TRANSITION_EVAL_RADIUS = 1

ATTNSLEEP_R_PERMUTE_20 = [14, 5, 4, 17, 8, 7, 19, 12, 0, 15, 16, 9, 11, 10, 3, 1, 6, 18, 2, 13]

REUSE_EXISTING = _env_bool("PROTOSLEEP_REUSE", True)
RUN_OPTIONAL_F = _env_bool("PROTOSLEEP_RUN_F", False)
RUN_OPTIONAL_G = _env_bool("PROTOSLEEP_RUN_G", False)
RUN_FINAL_TEST = _env_bool("PROTOSLEEP_FINAL_TEST", False)
ALLOW_TEST_RERUN = _env_bool("PROTOSLEEP_ALLOW_TEST_RERUN", False)
