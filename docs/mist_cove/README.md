# MIST-COVE v0 — experimental design handoff

**Status: DESIGN ONLY. Not a trained model, verified improvement, or approval to open EEG files.**

MIST-COVE = *Conditional Observability and Verified Evidence*. Its research question is whether stage decisions can preserve diagnostic evidence when measurement changes are benign, switch among legitimate witnesses when one disappears, and represent unresolvable distinctions when information is lost.

## Start here for coding agents

Read [the master coding-agent prompt](CODING_AGENT_MASTER_PROMPT.md) and the repository [AGENTS.md](../../AGENTS.md). The prompt tells the agent how to inspect the current interfaces and how to implement progressively without opening reserved subjects.

| File | Purpose |
|---|---|
| [RESEARCH_HYPOTHESES.md](RESEARCH_HYPOTHESES.md) | Mechanistic claims, identification limits, predictions, falsifiers |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | Cohorts, fair baselines, measurement interventions, metrics, frozen evaluation |
| [INTERFACES_AND_PROVENANCE.md](INTERFACES_AND_PROVENANCE.md) | Code contracts, local waveform traces, witness audit, failure conditions |
| [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | Ordered implementation phases, stage gates, acceptance checklists |
| [DECISION_RECORD_TEMPLATE.md](DECISION_RECORD_TEMPLATE.md) | Versioned preregistration and gate receipt template |
| [../../configs/mist_cove_design_v0.json](../../configs/mist_cove_design_v0.json) | Machine-readable, explicitly non-executable design contract |
| [../../experiments/mist_cove/ablation_matrix_v0.json](../../experiments/mist_cove/ablation_matrix_v0.json) | Hypothesis-to-control matrix and exact falsification conditions |
| [../../scripts/validate_mist_cove_design.py](../../scripts/validate_mist_cove_design.py) | Metadata-only, stdlib contract validator |

## Metadata-only quick check

```bash
python scripts/validate_mist_cove_design.py
pytest -q tests/transfer/test_mist_cove_design_contract.py
```

The validator reads *only* the two design JSONs; it does not inspect subject splits, EEG paths, raw files, or experiment outputs. These commands are safe to run without EEG and do not start training.

## Verified starting point (existing source-only research)

Base branch: `research/mist-transfer-v3-edf-shhs`, historical P3B report commit `15995b31821ee8db717f3d6edfa41166de65d4a3`; frozen implementation `661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23`. Four frozen variants (`attnsleep`, `evidence`, `summary_only`, `raw_context`), seeds 123/456/789, equal maximum optimizer recipe. The P3B source-VAL participant-mean macro-F1 averages are 0.312650, 0.718031, 0.512157, 0.723511 respectively. These are 4-participant *source validation* results, **not external-transfer evidence**. In particular, evidence did not beat raw_context in that comparison.

Confirmed SC scope: 74 eligible people, 146 recordings, 397,832 complete physical epochs (1,454 unscored retained). Frozen developmental division: 15 TRAIN, 4 VAL; 55 RESERVED subjects untouched. ST is conditional and exploratory; SHHS access and usability unconfirmed.

No new experiment, metric, cohort evaluation, or coding-agent-generated result is represented as completed in this design package.

## Intended next deliverable

An independently reviewed implementation PR with a minimal additive/event evidence model and synthetic tests, *not* a result on the reserved or external cohorts. Model, source-development, reserved evaluation, and genuine external-cohort gates must be separately approved and recorded.
