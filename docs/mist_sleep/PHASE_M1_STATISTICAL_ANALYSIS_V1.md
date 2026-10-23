# Paired uncertainty contract and missing outcomes

The primary outcome is **`PRIMARY_EVALUATION_BLOCKED`**. No HMC waveform payload, original scoring outcome, target prediction or clinical metric was read or produced. All clinical values are null. The Phase M0 decision remains `MPI_FEASIBILITY_NOT_ESTABLISHED`.

The endpoint is the equal-recording fixed-five Macro-F1 contrast M2-minus-M0, averaged over seeds 123/456. The 10,000-draw percentile bootstrap uses complete model-paired recording units and keeps the two seeds together; seed 20261023 is fixed. Epochs and training seeds never increase the patient count. If independently verified participant linkage exists before scoring, cluster on people while retaining the equal-recording endpoint. Current linkage is unknown. No real denominator, effect, confidence interval, per-seed contrast or stage missingness estimate can be calculated.

Generated controls recover a constant +0.03 paired effect and degenerate [+0.03,+0.03] interval. Duplicate recording IDs, wrong seed geometry, invalid grouping, single-cluster inference and changed bootstrap settings reject. These are analytical checks on invented units, not measured clinical uncertainty.
