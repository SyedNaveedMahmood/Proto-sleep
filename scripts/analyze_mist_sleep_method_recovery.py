"""Aggregate-only analysis of sealed private Phase M0 final checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from mist_sleep.augmentation import paired_view, stress_view
from mist_sleep.experiment import ARMS, new_model, verify_protocol
from mist_sleep.metrics import fixed_five
from mist_sleep.source import PRIVATE, PUBLIC, load_full_train


def _mean_person_f1(y, logits, person):
    prediction = logits.argmax(1)
    return np.array([f1_score(y[person == p], prediction[person == p],
                              labels=np.arange(5), average="macro", zero_division=0)
                     for p in np.unique(person)])


def _bootstrap_difference(a, b, seed=91):
    difference = a-b
    rng = np.random.default_rng(seed)
    resample = rng.integers(0, len(difference), size=(10000, len(difference)))
    means = difference[resample].mean(1)
    return {"mean": float(difference.mean()),
            "participant_bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist(),
            "participant_count": len(difference)}


@torch.no_grad()
def score_state(model, wave, index, device):
    positions = np.flatnonzero((index["role"] == 1) & (index["stage"] >= 0))
    views = ("clean", "bounded", "stress")
    generator = {"bounded": torch.Generator().manual_seed(2026),
                 "stress": torch.Generator().manual_seed(2027)}
    predictions = {key: [] for key in views}
    embedding = {key: [] for key in views}
    assignment = {key: [] for key in views}
    model.eval()
    for start in range(0, len(positions), 64):
        batch = positions[start:start+64]
        x = torch.from_numpy(np.asarray(wave[batch]).copy())
        samples = {"clean": x, "bounded": paired_view(x, generator["bounded"]),
                   "stress": stress_view(x, generator["stress"])}
        for key in views:
            output = model(samples[key].to(device))
            logits = output if isinstance(output, torch.Tensor) else output["logits"]
            predictions[key].append(logits.float().cpu().numpy())
            if isinstance(output, dict):
                embedding[key].append(output["embedding"].float().cpu().numpy())
                assignment[key].append(output["q"].float().cpu().numpy())
    truth = index["stage"][positions]
    person = index["person"][positions]
    result = {}
    private_person = {}
    for key in views:
        predictions[key] = np.concatenate(predictions[key])
        result[key] = {"stage": fixed_five(truth, predictions[key], person)}
        private_person[key] = _mean_person_f1(truth, predictions[key], person)
        if key == "clean":
            continue
        original = predictions["clean"]
        current = predictions[key]
        result[key]["prediction_agreement"] = float(np.mean(original.argmax(1) == current.argmax(1)))
        result[key]["mean_confidence_change"] = float(np.mean(
            np.exp(current-current.max(1,keepdims=True)).max(1) /
            np.exp(current-current.max(1,keepdims=True)).sum(1) -
            np.exp(original-original.max(1,keepdims=True)).max(1) /
            np.exp(original-original.max(1,keepdims=True)).sum(1)))
    if embedding["clean"]:
        for key in views:
            embedding[key] = np.concatenate(embedding[key])
            assignment[key] = np.concatenate(assignment[key])
        z = embedding["clean"]
        centroids = np.stack([z[truth == stage].mean(0) for stage in range(5)])
        normal = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
        cosine = normal @ normal.T
        result["clean"]["stage_centroid_mean_offdiag_cosine"] = float(cosine[~np.eye(5,dtype=bool)].mean())
        result["clean"]["embedding_feature_variance"] = float(z.var(axis=0).mean())
        for stage in range(5):
            qmean = assignment["clean"][truth == stage].mean((0,1))
            result["clean"][f"stage_{stage}_effective_prototypes"] = float(np.exp(-(qmean*np.log(np.clip(qmean,1e-8,1))).sum()))
        for key in ("bounded", "stress"):
            alternate = embedding[key]
            p = np.clip(assignment["clean"], 1e-8, 1)
            q = np.clip(assignment[key], 1e-8, 1)
            result[key]["embedding_cosine"] = float(np.mean(np.sum(z*alternate,axis=1) /
                np.maximum(np.linalg.norm(z,axis=1)*np.linalg.norm(alternate,axis=1),1e-8)))
            result[key]["embedding_L2"] = float(np.mean(np.linalg.norm(z-alternate,axis=1)))
            result[key]["prototype_KL"] = float(np.mean(np.sum(p*(np.log(p)-np.log(q)),axis=-1)))
            result[key]["prototype_switch_fraction"] = float(np.mean(p.argmax(-1)!=q.argmax(-1)))
    return result, private_person


def main():
    verify_protocol()
    if (PUBLIC / "robustness_analysis.json").exists():
        raise FileExistsError("analysis already completed; no outcome-driven rerun")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wave, index = load_full_train()
    results, private = {}, {}
    for seed in (123,456):
        for arm in ARMS:
            name = f"{arm}_{seed}"
            train = json.loads((PUBLIC / f"{name}.json").read_text())
            checkpoint = PRIVATE / "models" / name / "step_1200.pt"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != train["trajectory"][-1]["checkpoint_sha256"]:
                raise RuntimeError("private final checkpoint digest differs")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["arm"] != arm or state["seed"] != seed or state["step"] != 1200:
                raise RuntimeError("private final checkpoint identity differs")
            model = new_model(arm)
            model.load_state_dict(state["model"], strict=True)
            model.to(device)
            results[name], private[name] = score_state(model, wave, index, device)
            print(name, results[name]["bounded"]["stage"]["mean_participant_macro_f1"], flush=True)
    comparisons = {}
    for view in ("clean", "bounded", "stress"):
        comparisons[view] = {}
        for contrast in (("M2","M0"),("M2","M1"),("M3","M2"),("M3","M1"),("M0","A0")):
            label = "-".join(contrast)
            comparisons[view][label] = {str(seed): _bootstrap_difference(
                private[f"{contrast[0]}_{seed}"][view], private[f"{contrast[1]}_{seed}"][view])
                for seed in (123,456)}
            comparisons[view][label]["two_seed_mean"] = float(np.mean([
                comparisons[view][label][str(seed)]["mean"] for seed in (123,456)]))
    payload = {"status": "COMPLETE", "development_reused": True,
               "source": "original SC TRAIN only", "seeds": [123,456],
               "views": {"bounded": "frozen paired_view; seed 2026",
                         "stress": "destructive stress_view; seed 2027"},
               "results": results, "contrasts": comparisons}
    (PUBLIC / "robustness_analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
