"""Self-contained evidence report with real query/exemplar waveforms and exact margins."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np
import torch

from .crf import EvidenceCRF
from .data import STAGES, contiguous_runs, digest_file, load_split, validate_manifest, write_json
from .model import DESCRIPTORS, normalize_crop
from .train import model_from_payload, night_logits, safe_load


def plot_uri(query, exemplar, similarity, starts, fs=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = []
    # Each chart is separate. Raw amplitudes and normalized shape are not conflated.
    for title, arrays, labels, times in (
        ("Query crop and actual training exemplar (input units)", [query, exemplar],
         ["query", "training exemplar"], [np.arange(len(query))/fs, np.arange(len(exemplar))/fs]),
        ("Locally normalized shapes, with no inferred event label", [normalize_crop(torch.tensor(query)).numpy(), normalize_crop(torch.tensor(exemplar)).numpy()],
         ["query", "training exemplar"], [np.arange(len(query))/fs, np.arange(len(exemplar))/fs]),
        ("All window similarities to this exemplar", [similarity], ["similarity"], [np.asarray(starts)/fs]),
    ):
        fig, ax = plt.subplots(figsize=(8, 2.3))
        for a, label, t in zip(arrays, labels, times):
            ax.plot(t, a, label=label, linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Seconds")
        ax.legend(fontsize=8)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return images


@torch.no_grad()
def explain_epoch(manifest: Path, checkpoint: Path, output: Path, recording: str | None = None,
                  epoch: int = 0, context_checkpoint: Path | None = None, device: str = "cpu",
                  top_k: int = 6, perturb: bool = False):
    payload = safe_load(checkpoint)
    if payload["arm"] != "evidence":
        raise ValueError("Waveform-exemplar explanations require the evidence arm. Dense controls are not intrinsically interpretable.")
    doc, _ = validate_manifest(manifest)
    if doc != payload["provenance"]["manifest"]:
        raise ValueError("Explanation split differs from training manifest")
    nights = load_split(manifest, "val")
    if any(payload["provenance"]["files"].get(n.key) != n.sha256 for n in nights):
        raise ValueError("Validation recording changed since training")
    n = nights[0] if recording is None else next((n for n in nights if n.key == recording), None)
    if n is None or not 0 <= epoch < len(n.y) or n.y[epoch] < 0:
        raise ValueError("Choose an existing labeled validation epoch")
    model = model_from_payload(payload).to(device).eval()
    x = torch.from_numpy(n.x[epoch:epoch+1]).to(device)
    result = model(x, details=True)
    logits = result["logits"][0].double().cpu()
    probabilities = logits.softmax(-1)
    context = {"left_context": 0.0, "right_context": 0.0}
    if context_checkpoint:
        cp = safe_load(context_checkpoint)
        if cp["local_sha256"] != digest_file(checkpoint):
            raise ValueError("CRF was trained with a different local model")
        crf = EvidenceCRF(bound=cp["bound"])
        crf.load_state_dict(cp["model"])
        run = next(ix for ix in contiguous_runs(n, cp["assume_contiguous"]) if epoch in ix)
        emissions = night_logits(model, n, device, 32).double()[run]
        pos = int(np.flatnonzero(run == epoch)[0])
        probabilities = crf.log_marginals(emissions)[pos].exp()
        c, r = probabilities.argsort(descending=True)[:2].tolist()
        ledger = crf.margin_ledger(emissions, pos, c, r)
        context = {k: float(ledger[k]) for k in ("left_context", "right_context")}
    else:
        c, r = probabilities.argsort(descending=True)[:2].tolist()
    contributions, bias, reconstructed = model.margin_ledger(result["features"][0], c, r)
    local_margin = float(logits[c] - logits[r])
    if abs(float(reconstructed) - local_margin) > 1e-4:
        raise RuntimeError("Local explanation completeness assertion failed")
    margin = local_margin + context["left_context"] + context["right_context"]
    if abs(margin - float(probabilities[c].log() - probabilities[r].log())) > 1e-4:
        raise RuntimeError("Context explanation completeness assertion failed")
    items = [{"feature": name, "value": float(result["features"][0, j]),
              "class_contrast_weight": float(model.classifier.weight[c, j] - model.classifier.weight[r, j]),
              "margin_contribution": float(contributions[j])}
             for j, name in enumerate(model.feature_names)]
    items.sort(key=lambda row: abs(row["margin_contribution"]), reverse=True)
    exemplar_items = [row for row in items if ".p" in row["feature"]]
    shown, seen = [], set()
    for row in exemplar_items:
        si_s, k_s, pool = row["feature"].split(".")
        si, k = int(si_s[1:]), int(k_s[1:])
        if (si, k) in seen:
            continue
        seen.add((si, k))
        detail = result["details"][si]
        j = int(detail["argmax"][0, k])
        start, length = detail["starts"][j], detail["length"]
        raw = getattr(model, f"anchors_{si}")[k, 0].cpu().numpy()
        sim = detail["similarity"][0, :, k].cpu().numpy()
        q = n.x[epoch, 0, start:start + length]
        shown.append({**row, "scale": si, "anchor": k, "query_start_sample": start,
                      "query_end_sample": start + length, "query_window_index": j,
                      "reference": model.refs[si][k], "similarities": sim.tolist(),
                      "window_start_samples": detail["starts"],
                      "measured_query": dict(zip(DESCRIPTORS, map(float, detail["descriptors"][0, j]))),
                      "measured_exemplar": dict(zip(DESCRIPTORS, map(float, detail["anchor_descriptors"][k]))),
                      "latent_distance": float(detail["latent_distance"][0, j, k]),
                      "measured_distance": float(detail["measured_distance"][0, j, k]),
                      "images": plot_uri(q, raw, sim, detail["starts"])})
        if len(shown) == top_k:
            break
    report = {"recording": n.key, "subject": n.subject, "epoch_row": epoch,
              "epoch_index": int(n.epoch_index[epoch]), "true_stage": STAGES[int(n.y[epoch])],
              "prediction": STAGES[c], "alternative": STAGES[r],
              "probabilities_uncalibrated": dict(zip(STAGES, map(float, probabilities))),
              "local_logit_margin": local_margin, "bias_margin": float(bias), **context,
              "total_log_odds": margin, "contribution_sum": float(contributions.sum()),
              "completeness_error": abs(float(contributions.sum() + bias) + sum(context.values()) - margin),
              "checkpoint_sha256": digest_file(checkpoint), "all_features": items,
              "exemplars": [{k: v for k, v in row.items() if k != "images"} for row in shown],
              "warning": "Model-faithful evidence is not proof of a biological event or causal effect. No automatic spindle/K-complex names. NPZ amplitude units are unverified.",
              "mean_pool_note": "A mean feature depends on all shown window similarities, not only the closest displayed crop."}
    if perturb and shown:
        focus = shown[0]
        s, end = focus["query_start_sample"], focus["query_end_sample"]
        length = end - s
        rng = np.random.default_rng(7001)
        starts = [s] + rng.integers(0, 3000 - length + 1, size=5).tolist()
        changes = []
        for pos in starts:
            changed = x.clone()
            left = changed[0, 0, max(0, pos-1)]
            right = changed[0, 0, min(2999, pos+length)]
            changed[0, 0, pos:pos+length] = torch.linspace(float(left), float(right), length, device=device)
            l = model(changed)["logits"][0]
            changes.append(local_margin - float(l[c] - l[r]))
        report["input_intervention"] = {"replacement": "linear interpolation; may be off distribution",
                                         "largest_absolute_evidence_window_margin_drop": changes[0],
                                         "matched_length_random_margin_drops": changes[1:],
                                         "random_starts": starts[1:], "length": length,
                                         "clinical_causality_claim": False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "explanation.json", report)
    with (output / "feature_ledger.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)
    with (output / "event_review_template.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scale", "anchor", "recording", "epoch_index", "start_sample", "end_sample", "expert_event_label", "reviewer", "montage_verified", "units_verified", "comments"])
        for si, refs in enumerate(model.refs):
            for k, ref in enumerate(refs):
                w.writerow([si, k, ref["recording"], ref["epoch_index"], ref["start_sample"], ref["end_sample"], "unreviewed", "", "", "", ""])
    esc = html.escape
    sections = ["<!doctype html><html><head><meta charset='utf-8'><title>MIST evidence report</title></head><body>",
                "<h1>MIST-Evidence: local waveform evidence and temporal context</h1>",
                f"<p>{esc(report['warning'])}</p>", f"<p>{esc(report['mean_pool_note'])}</p>",
                f"<h2>{esc(STAGES[c])} versus {esc(STAGES[r])}</h2>",
                "<pre>" + esc(json.dumps({k: report[k] for k in ('recording','epoch_row','prediction','alternative','local_logit_margin','bias_margin','left_context','right_context','total_log_odds','completeness_error')}, indent=2)) + "</pre>"]
    for row in shown:
        sections.extend([f"<h3>{esc(row['feature'])}, signed margin contribution {row['margin_contribution']:+.5f}</h3>",
                         "<pre>" + esc(json.dumps(row["reference"], indent=2)) + "</pre>",
                         f"<p>Query samples [{row['query_start_sample']}, {row['query_end_sample']}). Similarity is in the declared learned-plus-measured metric, not a clinical event match.</p>"])
        sections.extend(f"<img alt='Observed waveform or similarity curve' src='{uri}'><br>" for uri in row["images"])
    sections.append("<h2>Complete local feature ledger</h2><table border='1'><tr><th>Feature</th><th>Value</th><th>Class contrast weight</th><th>Contribution</th></tr>")
    for row in items:
        sections.append(f"<tr><td>{esc(row['feature'])}</td><td>{row['value']:.6f}</td><td>{row['class_contrast_weight']:+.6f}</td><td>{row['margin_contribution']:+.6f}</td></tr>")
    sections.append("</table></body></html>")
    (output / "explanation.html").write_text("\n".join(sections), encoding="utf-8")
    print("Evidence report: " + str(output / "explanation.html"), flush=True)
    return report
