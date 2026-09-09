"""Auditable waveform reports for MIST-Morph transport evidence."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import torch

from .data import Recording
from .explain import score_reconstruction_close
from .model import AdditiveTemporal, STAGES, normalize_wave
from .runtime import atomic_json, write_csv
from .transport import TransportEvidenceModel


@torch.no_grad()
def export_transport_explanation(
    model: TransportEvidenceModel,
    recording: Recording,
    array_epoch: int,
    bank: dict,
    output: Path,
    device: torch.device,
    encode_batch: int = 16,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not isinstance(model, TransportEvidenceModel) or not isinstance(model.temporal, AdditiveTemporal):
        raise ValueError("transport explanation requires TransportEvidenceModel with additive emissions")
    if not 0 <= array_epoch < len(recording.y):
        raise ValueError("epoch is outside the recording")

    model.eval()
    lo, hi = next((a, b) for a, b in recording.segments() if a <= array_epoch < b)
    x = torch.from_numpy(recording.x[lo:hi]).to(device)
    feature = torch.cat([model.encode(x[i:i + encode_batch]) for i in range(0, len(x), encode_batch)])
    emission = model.emissions(feature[None])[0]
    decoded, probabilities = model.predict(emission)
    t = array_epoch - lo
    parts = model.temporal.explain_at(feature, t)
    reconstructed = parts.sum((1, 2)) + model.temporal.bias
    error = float((reconstructed - emission[t]).abs().max())
    if not score_reconstruction_close(reconstructed, emission[t]):
        raise RuntimeError(f"transport explanation does not reconstruct model score: {error}")

    chosen = int(decoded[t])
    alternatives = emission[t].clone()
    alternatives[chosen] = -float("inf")
    other = int(alternatives.argmax())
    contrast = parts[chosen] - parts[other]

    transition_delta = 0.0
    if model.crf is not None:
        crf = model.crf
        left = crf.start if t == 0 else crf.transitions[int(decoded[t - 1])]
        right = crf.end if t == len(decoded) - 1 else crf.transitions[:, int(decoded[t + 1])]
        transition_delta = float(left[chosen] - left[other] + right[chosen] - right[other])

    rows = []
    for lag_index, lag in enumerate(range(-model.cfg.radius, model.cfg.radius + 1)):
        j = t + lag
        if not 0 <= j < len(feature):
            continue
        for f, name in enumerate(model.feature_names):
            rows.append({
                "source_array_epoch": lo + j,
                "source_original_epoch": int(recording.indices[lo + j]) if recording.indices_known else None,
                "lag_epochs": lag,
                "feature": name,
                "value": float(feature[j, f]),
                "chosen_score_contribution": float(parts[chosen, lag_index, f]),
                "contrast_score_contribution": float(contrast[lag_index, f]),
            })
    rows.sort(key=lambda row: abs(row["contrast_score_contribution"]), reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "all_contributions.csv")

    details = {
        "architecture": "MIST-Morph transport v2",
        "recording": Path(recording.path).name,
        "subject": recording.subject,
        "array_epoch": array_epoch,
        "original_epoch": int(recording.indices[array_epoch]) if recording.indices_known else None,
        "truth": STAGES[int(recording.y[array_epoch])],
        "prediction": STAGES[chosen],
        "local_argmax": STAGES[int(emission[t].argmax())],
        "contrast_class": STAGES[other],
        "model_marginals_not_calibrated_confidence": probabilities[t].cpu().tolist(),
        "all_evidence_difference_without_bias": float(contrast.sum()),
        "center_epoch_evidence_difference": float(contrast[model.cfg.radius].sum()),
        "neighbor_evidence_difference": float(contrast.sum() - contrast[model.cfg.radius].sum()),
        "bias_difference": float(model.temporal.bias[chosen] - model.temporal.bias[other]),
        "crf_neighbor_transition_difference": transition_delta,
        "conditional_path_score_difference": float(emission[t, chosen] - emission[t, other]) + transition_delta,
        "score_reconstruction_max_abs_error": error,
        "transport_semantics": "Mass is normalized assignment over overlapping windows, not an event count.",
        "clinical_validation": "None. Anchor IDs are source waveforms, not spindle/K-complex diagnoses.",
        "explanation_scope": "Exact model-score accounting plus source-waveform correspondence; not physiological causality.",
        "continuity_assumed": recording.continuity_assumed,
    }

    images = []
    time_axis = np.arange(model.cfg.samples) / model.cfg.fs
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(time_axis, recording.x[array_epoch, 0], linewidth=0.8)
    ax.set(
        xlabel="Seconds within epoch",
        ylabel="Input file units",
        title=f"Observed EEG | predicted {STAGES[chosen]} | annotated {details['truth']}",
    )
    fig.tight_layout()
    fig.savefig(output / "epoch.png", dpi=140)
    plt.close(fig)
    images.append(("epoch.png", "Raw validation EEG. No waveform is generated by a decoder."))

    matches = []
    used = set()
    for row in rows:
        name = row["feature"]
        tokens = name.split("_")
        if len(tokens) != 4 or tokens[0] != "ot" or not tokens[1].startswith("s") or not tokens[2].startswith("p"):
            continue
        length = int(tokens[1][1:])
        proto = int(tokens[2][1:])
        source_epoch = int(row["source_array_epoch"])
        key = (length, proto, source_epoch)
        if key in used:
            continue
        used.add(key)

        scale_index = model.cfg.scales.index(length)
        one = torch.from_numpy(recording.x[source_epoch:source_epoch + 1]).to(device)
        _, scale_details = model.encode_with_details(one)
        info = scale_details[scale_index]
        transport = info["transport_plan"][0, :, proto]
        window = int(transport.argmax())
        start = window * model.cfg.stride
        query = torch.from_numpy(recording.x[source_epoch, 0, start:start + length]).to(device)
        anchor = model.banks[scale_index].waveforms[proto]
        q, _ = normalize_wave(query)
        p, _ = normalize_wave(anchor)

        filename = f"match_{len(matches):02d}.png"
        fig, ax = plt.subplots(figsize=(9, 2.5))
        local_time = np.arange(length) / model.cfg.fs
        ax.plot(local_time, q.cpu(), label="Observed input window", linewidth=1)
        ax.plot(local_time, p.cpu(), label="Actual training anchor", linewidth=1, alpha=0.75)
        ax.set(
            xlabel="Seconds within window",
            ylabel="Local RMS-normalized EEG",
            title=f"{name} | lag {row['lag_epochs']:+d} | contribution {row['contrast_score_contribution']:+.4f}",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)

        meta = {
            **row,
            "anchor": bank["metadata"][scale_index][proto],
            "query_start_sample": start,
            "query_stop_sample": start + length,
            "largest_window_transport_mass": float(transport[window]),
            "total_anchor_transport_mass": float(info["transport_anchor_mass"][0, proto]),
            "transport_weighted_affinity": float(info["transport_affinity"][0, proto]),
            "transport_position_moment": float(info["transport_position_moment"][0, proto]),
            "background_mass_scale": float(info["transport_background_mass"][0, 0]),
            "local_similarity_at_displayed_window": float(info["scores"][0, window, proto]),
            "observed_distance_components_shape_spectrum_envelope": info["distance_components"][0, window, proto].cpu().tolist(),
            "neural_distance": float(info["neural_distance"][0, window, proto]),
            "observed_weights": info["observed_weights"][proto].cpu().tolist(),
            "neural_fraction": float(info["neural_fraction"][proto]),
            "all_window_transport_mass_for_anchor": transport.cpu().tolist(),
            "visualization_note": "The displayed interval has the largest transported mass for this anchor. Total mass/affinity aggregate overlapping windows and are not event counts.",
        }
        matches.append(meta)
        images.append((filename, meta["visualization_note"]))
        if len(matches) >= 6:
            break

    details["transport_audit_center_epoch"] = model.transport_audit(
        torch.from_numpy(recording.x[array_epoch:array_epoch + 1]).to(device)
    )
    details["displayed_matches"] = matches
    atomic_json(details, output / "explanation.json")

    table_rows = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(r[k]))}</td>"
            for k in ("source_array_epoch", "lag_epochs", "feature", "value", "contrast_score_contribution")
        ) + "</tr>"
        for r in rows[:20]
    )
    text = f"""<!doctype html><html><head><meta charset="utf-8"><title>MIST-Morph transport evidence</title>
<style>body{{font:16px sans-serif;max-width:1000px;margin:36px auto;line-height:1.5}}td,th{{padding:6px;border:1px solid #ccc}}table{{border-collapse:collapse}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style></head><body>
<h1>MIST-Morph v2: morphology transport audit</h1>
<p><strong>Research output, not a clinical diagnosis.</strong> Anchors are actual training waveforms. Transport mass over overlapping windows is not an event count.</p>
<pre>{html.escape(json.dumps({k: v for k, v in details.items() if k != 'displayed_matches'}, indent=2))}</pre>
<h2>Transported waveform evidence</h2>
"""
    for image, caption in images:
        text += f'<img src="{image}" alt="Waveform comparison"><p>{html.escape(caption)}</p>\n'
    text += "<h2>Largest signed score differences</h2>"
    text += "<p>Compared classes: " + STAGES[chosen] + " minus " + STAGES[other] + "</p>"
    text += "<table><tr><th>Source epoch</th><th>Lag</th><th>Feature</th><th>Value</th><th>Contribution</th></tr>" + table_rows + "</table>"
    text += "<p>All terms are stored in all_contributions.csv. Transport assignment, classifier contribution, and CRF transition influence are separate quantities.</p></body></html>"
    (output / "index.html").write_text(text, encoding="utf-8")
    return details
