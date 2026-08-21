from __future__ import annotations

import json
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import *
from .utils import *
from .data import *
from .attnsleep import *
from .prototypes import *
from .micro import *
from .cache import *
from .night import *
from .masking import *
from .macro import *
from .losses import *
from .train_macro import *
from .evaluation import *
from .selftest import run_self_tests


def _nparams(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _timed(label, fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    print(f"[time] {label}: {seconds:.2f}s")
    return out, seconds


def _cache_train_val(model, split, cache_dir: Path):
    train_ok = cache_complete(split["train"], cache_dir, "train")
    val_ok = cache_complete(split["val"], cache_dir, "val")

    if REUSE_EXISTING and train_ok and val_ok:
        print("[cache-hit] train/validation feature cache")
        return (
            load_feature_cache(split["train"], cache_dir, "train"),
            load_feature_cache(split["val"], cache_dir, "val"),
        )

    print("[cache-build] extracting B features for TRAIN + VALIDATION only")
    train_n = build_feature_cache(model, split["train"], cache_dir, "train")
    val_n = build_feature_cache(model, split["val"], cache_dir, "val")
    return train_n, val_n


def _load_test_cache_only_when_unlocked(model, split, cache_dir: Path):
    if not RUN_FINAL_TEST:
        return None
    if REUSE_EXISTING and cache_complete(split["test"], cache_dir, "test"):
        print("[cache-hit] final test feature cache")
        return load_feature_cache(split["test"], cache_dir, "test")
    print("[cache-build] final test cache (RUN_FINAL_TEST is enabled)")
    return build_feature_cache(model, split["test"], cache_dir, "test")


def run_fold() -> pd.DataFrame:
    print("=" * 88)
    print(f"{PROJECT_NAME} | fold={FOLD_ID} | device={DEVICE} | AMP={USE_AMP}")
    print(f"reuse={REUSE_EXISTING} | final_test={RUN_FINAL_TEST}")
    print("=" * 88)

    if RUN_SELF_TESTS:
        run_self_tests()
    if SELF_TEST_ONLY:
        print("SELF_TEST_ONLY=1; stopping before dataset access.")
        return pd.DataFrame()

    recordings = load_sleep_edf_recordings(DATA_DIR)
    fold_seed = SEED + 10000 * FOLD_ID
    seed_everything(fold_seed)
    split = split_subjects(recordings, FOLD_ID)

    fold_dir = ensure_dir(OUTPUT_ROOT / f"fold_{FOLD_ID:02d}")
    for sub in ["checkpoints", "cache", "plots", "tables", "logs"]:
        ensure_dir(fold_dir / sub)

    train_loader = make_epoch_loader(split["train"], MICRO_BATCH_SIZE, shuffle=True)
    val_loader = make_epoch_loader(split["val"], MICRO_BATCH_SIZE, shuffle=False)
    class_weights = balanced_class_weights_from_train(split["train"])

    print("\nSubjects")
    print(" train:", split["train_subjects"])
    print(" val:  ", split["val_subjects"])
    print(" test: ", split["test_subjects"], "(locked unless --final-test)")
    print("\nModel sizes")
    print(f" AttnSleep baseline: {_nparams(AttnSleepBaseline()):,} params")
    print(f" Proto-AttnSleep:    {_nparams(ProtoAttnSleep(enable_micro_mask=ENABLE_MICRO_MASK)):,} params")
    print(f" Macro classifier:   {_nparams(MacroStageClassifier(MacroEncoder(NUM_PROTOTYPES))):,} params")
    print(f" Prototype MAE:      {_nparams(MaskedSequenceAutoencoder(NUM_PROTOTYPES, NUM_PROTOTYPES, 'distribution')):,} params")

    n_train_epochs = sum(r.n_epochs for r in split["train"])
    micro_batches = int(np.ceil(n_train_epochs / MICRO_BATCH_SIZE))
    print("\nExpected optimizer-step scale")
    print(f" micro: ~{micro_batches} batches/epoch x early-stopped epochs")
    print(f" macro: {int(np.ceil(len(split['train']) / MACRO_BATCH_SIZE))} batches/epoch x {MACRO_SSL_EPOCHS} SSL epochs")
    print(" Macro training operates on cached [T,K] / [T,30] night features, not raw 3000-sample EEG.")

    VAL_RESULTS = {}
    MODELS = {}
    SSL_MODELS = {}
    ARTIFACTS = {}
    timings = {}

    # ------------------------------------------------------------------ A
    A_ckpt = fold_dir / "checkpoints" / "A_AttnSleep.pt"

    def build_A():
        if REUSE_EXISTING and A_ckpt.exists():
            print("[checkpoint-hit]", A_ckpt)
            return load_model_state(AttnSleepBaseline(), A_ckpt, DEVICE).eval()
        seed_everything(fold_seed + 1)
        model = AttnSleepBaseline()
        model.apply(init_attnsleep_weights)
        return train_micro_model(
            model, train_loader, val_loader, class_weights,
            A_ckpt, proto_cfg=None, seed=fold_seed + 1
        )["model"]

    A, timings["A_seconds"] = _timed("A AttnSleep", build_A)
    VAL_RESULTS["A_AttnSleep"] = evaluate_micro_loader(A, val_loader, DEVICE)
    MODELS["A_AttnSleep"] = A

    # ------------------------------------------------------------------ B
    default_proto_cfg = PROTO_TRIALS[0]
    B_ckpt = fold_dir / "checkpoints" / f"B_proto_{default_proto_cfg['name']}.pt"

    def build_B():
        if REUSE_EXISTING and B_ckpt.exists() and not RUN_PROTO_HPARAM_SEARCH:
            print("[checkpoint-hit]", B_ckpt)
            model = ProtoAttnSleep(enable_micro_mask=ENABLE_MICRO_MASK)
            model = load_model_state(model, B_ckpt, DEVICE).eval()
            return {"model": model, "cfg": default_proto_cfg}
        return train_or_select_proto_model(train_loader, val_loader, class_weights, fold_dir)

    resB, timings["B_seconds"] = _timed("B Proto-AttnSleep", build_B)
    B = resB["model"].to(DEVICE).eval()
    VAL_RESULTS["B_ProtoAttnSleep"] = evaluate_micro_loader(B, val_loader, DEVICE)
    MODELS["B_ProtoAttnSleep"] = B
    ARTIFACTS["proto_cfg"] = resB["cfg"]

    # --------------------------------------------------------- train/val cache
    cache_dir = fold_dir / "cache"
    (train_n, val_n), timings["cache_seconds"] = _timed(
        "feature cache train+val",
        lambda: _cache_train_val(B, split, cache_dir),
    )

    p_fwd, p_rev, transition_counts = fit_transition_graph(train_n)
    proto_vectors_np = normalized_prototype_vectors(B)
    np.savez_compressed(
        fold_dir / "tables" / "train_transition_graph.npz",
        p_fwd=p_fwd,
        p_rev=p_rev,
        counts=transition_counts,
        prototype_vectors=proto_vectors_np,
    )

    proto_diag = prototype_diagnostics(B, train_n)
    train_proto = make_night_loader(train_n, "prototype", shuffle=True)
    val_proto = make_night_loader(val_n, "prototype", shuffle=False)

    print(
        f"prototype diagnostics: effective_K={proto_diag['effective_prototypes']:.2f}/"
        f"{NUM_PROTOTYPES}, max_cos={proto_diag['max_pairwise_cosine']:.6f}"
    )

    # --------------------------------------------------------------- helper
    def load_or_train_mae_experiment(
        tag, mask_mode, train_loader_, val_loader_, input_dim,
        target_dim, target_type, ssl_seed, ft_seed,
        geometry_lambda=0.0, prototype_vectors_=None,
        p_fwd_=None, p_rev_=None,
    ):
        ssl_ckpt = fold_dir / "checkpoints" / f"{tag}_ssl.pt"
        ft_ckpt = fold_dir / "checkpoints" / f"{tag}_finetune.pt"

        mae = MaskedSequenceAutoencoder(input_dim, target_dim, target_type)
        if REUSE_EXISTING and ssl_ckpt.exists():
            print("[checkpoint-hit]", ssl_ckpt)
            mae = load_model_state(mae, ssl_ckpt, DEVICE).eval()
        else:
            seed_everything(ssl_seed)
            mae = MaskedSequenceAutoencoder(input_dim, target_dim, target_type)
            mae = pretrain_mae(
                mae, train_loader_, mask_mode, ssl_ckpt, ssl_seed,
                p_fwd_, p_rev_, prototype_vectors_, geometry_lambda,
            )["model"]

        clf = MacroStageClassifier(copy.deepcopy(mae.encoder))
        if REUSE_EXISTING and ft_ckpt.exists():
            print("[checkpoint-hit]", ft_ckpt)
            clf = load_model_state(clf, ft_ckpt, DEVICE).eval()
        else:
            clf = train_macro_classifier(
                clf, train_loader_, val_loader_, class_weights, ft_ckpt, ft_seed
            )["model"]
        return mae, clf

    # ------------------------------------------------------------------ C
    C_ckpt = fold_dir / "checkpoints" / "C_supervised_macro.pt"

    def build_C():
        if REUSE_EXISTING and C_ckpt.exists():
            print("[checkpoint-hit]", C_ckpt)
            return load_model_state(
                MacroStageClassifier(MacroEncoder(NUM_PROTOTYPES)),
                C_ckpt,
                DEVICE,
            ).eval()
        model = MacroStageClassifier(fresh_encoder(NUM_PROTOTYPES, fold_seed + 200))
        return train_macro_classifier(
            model, train_proto, val_proto, class_weights, C_ckpt, fold_seed + 200
        )["model"]

    C, timings["C_seconds"] = _timed("C supervised macro", build_C)
    VAL_RESULTS["C_SupervisedMacro"] = evaluate_macro_classifier(C, val_proto)
    MODELS["C_SupervisedMacro"] = C

    # ------------------------------------------------------------------ D
    (D_mae, D), timings["D_seconds"] = _timed(
        "D random prototype MAE",
        lambda: load_or_train_mae_experiment(
            tag="D_random_mae",
            mask_mode="random",
            train_loader_=train_proto,
            val_loader_=val_proto,
            input_dim=NUM_PROTOTYPES,
            target_dim=NUM_PROTOTYPES,
            target_type="distribution",
            ssl_seed=fold_seed + 300,
            ft_seed=fold_seed + 302,
            geometry_lambda=0.0,
            prototype_vectors_=None,
            p_fwd_=p_fwd,
            p_rev_=p_rev,
        ),
    )
    SSL_MODELS["D_RandomMAE"] = D_mae
    MODELS["D_RandomMAE"] = D
    VAL_RESULTS["D_RandomMAE"] = evaluate_macro_classifier(D, val_proto)

    # ------------------------------------------------------------------ E
    (E_mae, E), timings["E_seconds"] = _timed(
        "E transition prototype MAE",
        lambda: load_or_train_mae_experiment(
            tag="E_transition_mae",
            mask_mode="transition",
            train_loader_=train_proto,
            val_loader_=val_proto,
            input_dim=NUM_PROTOTYPES,
            target_dim=NUM_PROTOTYPES,
            target_type="distribution",
            ssl_seed=fold_seed + 300,
            ft_seed=fold_seed + 302,
            geometry_lambda=0.0,
            prototype_vectors_=None,
            p_fwd_=p_fwd,
            p_rev_=p_rev,
        ),
    )
    SSL_MODELS["E_ProtoMAE"] = E_mae
    MODELS["E_ProtoMAE"] = E
    VAL_RESULTS["E_ProtoMAE"] = evaluate_macro_classifier(E, val_proto)

    # ------------------------------------------------------------------ G
    if RUN_OPTIONAL_G:
        (G_mae, G), timings["G_seconds"] = _timed(
            "G geometry prototype MAE",
            lambda: load_or_train_mae_experiment(
                tag="G_transition_geo_mae",
                mask_mode="transition",
                train_loader_=train_proto,
                val_loader_=val_proto,
                input_dim=NUM_PROTOTYPES,
                target_dim=NUM_PROTOTYPES,
                target_type="distribution",
                ssl_seed=fold_seed + 300,
                ft_seed=fold_seed + 302,
                geometry_lambda=PROTO_GEOMETRY_LAMBDA,
                prototype_vectors_=proto_vectors_np,
                p_fwd_=p_fwd,
                p_rev_=p_rev,
            ),
        )
        SSL_MODELS["G_ProtoGeoMAE"] = G_mae
        MODELS["G_ProtoGeoMAE"] = G
        VAL_RESULTS["G_ProtoGeoMAE"] = evaluate_macro_classifier(G, val_proto)

    # ------------------------------------------------------------------ F
    if RUN_OPTIONAL_F:
        latent_mean, latent_std = fit_latent_standardizer(train_n)
        train_ls = standardized_latent_nights(train_n, latent_mean, latent_std)
        val_ls = standardized_latent_nights(val_n, latent_mean, latent_std)
        train_lat = make_night_loader(train_ls, "latent", shuffle=True)
        val_lat = make_night_loader(val_ls, "latent", shuffle=False)

        (F_mae, F_model), timings["F_seconds"] = _timed(
            "F latent MAE",
            lambda: load_or_train_mae_experiment(
                tag="F_latent_mae",
                mask_mode="random",
                train_loader_=train_lat,
                val_loader_=val_lat,
                input_dim=30,
                target_dim=30,
                target_type="continuous",
                ssl_seed=fold_seed + 400,
                ft_seed=fold_seed + 402,
            ),
        )
        SSL_MODELS["F_LatentMAE"] = F_mae
        MODELS["F_LatentMAE"] = F_model
        VAL_RESULTS["F_LatentMAE"] = evaluate_macro_classifier(F_model, val_lat)

    # ----------------------------------------------------------- validation
    val_summary = pd.DataFrame([
        {
            "model": name,
            "accuracy": met["accuracy"],
            "macro_f1": met["macro_f1"],
            "kappa": met["kappa"],
        }
        for name, met in VAL_RESULTS.items()
    ]).sort_values("macro_f1", ascending=False)

    print("\nValidation-only summary")
    print(val_summary.to_string(index=False))
    val_summary.to_csv(fold_dir / "tables" / "validation_summary.csv", index=False)

    # Timing/provenance makes "too-fast training" auditable.
    run_meta = {
        "project_version": PROJECT_VERSION,
        "fold": FOLD_ID,
        "device": str(DEVICE),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "amp": USE_AMP,
        "reuse_existing": REUSE_EXISTING,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "macro_batch_size": MACRO_BATCH_SIZE,
        "train_epochs_raw": int(n_train_epochs),
        "train_nights": int(len(train_n)),
        "model_params": {
            "attnsleep": _nparams(AttnSleepBaseline()),
            "proto_attnsleep": _nparams(ProtoAttnSleep(enable_micro_mask=ENABLE_MICRO_MASK)),
            "macro_classifier": _nparams(MacroStageClassifier(MacroEncoder(NUM_PROTOTYPES))),
            "prototype_mae": _nparams(MaskedSequenceAutoencoder(NUM_PROTOTYPES, NUM_PROTOTYPES, "distribution")),
        },
        "timings_seconds": timings,
    }
    (fold_dir / "tables" / "run_provenance.json").write_text(
        json.dumps(run_meta, indent=2)
    )

    # -------------------------------------------------------------- test lock
    if RUN_FINAL_TEST:
        test_lock = fold_dir / "TEST_EVALUATED.lock"
        saved_summary = fold_dir / "tables" / "test_summary.csv"

        if test_lock.exists() and not ALLOW_TEST_RERUN:
            print("Final-test lock exists; test metrics will NOT be recomputed:", test_lock)
        else:
            test_n = _load_test_cache_only_when_unlocked(B, split, cache_dir)
            predictions = {
                "A_AttnSleep": predict_micro_recordings(MODELS["A_AttnSleep"], split["test"]),
                "B_ProtoAttnSleep": predict_micro_recordings(MODELS["B_ProtoAttnSleep"], split["test"]),
            }
            for name in ["C_SupervisedMacro", "D_RandomMAE", "E_ProtoMAE", "G_ProtoGeoMAE"]:
                if name in MODELS:
                    predictions[name] = predict_macro_recordings(MODELS[name], test_n, "prototype")

            rows = []
            for name, pred_dict in predictions.items():
                met = pooled_metrics_from_prediction_dict(pred_dict)
                tr = transition_metrics_from_prediction_dict(pred_dict)
                row = {
                    "model": name,
                    "accuracy": met["accuracy"],
                    "macro_f1": met["macro_f1"],
                    "kappa": met["kappa"],
                    "transition_accuracy": tr["accuracy"],
                    "transition_macro_f1": tr["macro_f1"],
                    "transition_n1_f1": tr["n1_f1"],
                    "transition_n": tr["n"],
                }
                for i, stage in enumerate(STAGE_NAMES):
                    row[f"f1_{stage}"] = float(met["per_class_f1"][i])
                rows.append(row)

            test_summary = pd.DataFrame(rows).sort_values("model")
            test_summary.to_csv(saved_summary, index=False)
            test_lock.write_text(
                f"evaluated_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                f"project_version={PROJECT_VERSION}\n"
                f"fold_id={FOLD_ID}\n"
            )
            print("\nFINAL TEST")
            print(test_summary.to_string(index=False))
            print("Wrote final-test lock:", test_lock)
    else:
        print("\nFinal test skipped. Test feature cache was not built or loaded.")

    return val_summary


def main():
    run_fold()


if __name__ == "__main__":
    main()
