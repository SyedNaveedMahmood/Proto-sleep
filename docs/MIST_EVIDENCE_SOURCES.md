# Prior art, sources, and novelty boundaries

Research checked on 9 September 2026. This is a focused review, not proof that no similar architecture exists. Primary papers, author repositories, and institutional publication records were used. Existing project results are identified separately from literature evidence.

## Closest prior art

| Source | What it already establishes | Consequence for this project |
|---|---|---|
| WaveSleepNet | Sleep-stage waveform prototypes, latent matching, prototype contributions, visual interpretation | "Prototype-based interpretable sleep staging" is not new |
| ProtoSleepNet, 2026 preprint | Prototype sequence-to-sequence sleep staging and whole-night prototype-grams; multimodal evaluation | "Prototypes plus full-night context" is not sufficient novelty |
| Learning Time-Series Shapelets | Learning discriminative subsequences for classification | Trainable/local waveform matching is established |
| ProtoPNet | Case-based local prototype evidence for classification | "This waveform looks like that waveform" has established methodological parents |
| Neural Additive Models | Nonlinear univariate functions combined additively | Additive nonlinear explanation is established |
| Neural CRF sleep work | Sequence-transition modeling for sleep staging | CRF smoothing/structured inference is not a contribution by itself |
| SleepMaMi, 2026 preprint | Hierarchical macro/micro modeling and hybrid self-supervised sleep representations | Generic hierarchical morphology-aware SSL is not a safe novelty claim |
| U-Sleep | Multi-cohort robust sleep staging; EEG/EOG and explicit held-out clinical cohorts | SOTA/transfer claims require much stronger data and protocol evidence than one tuned EDF-20 cohort |

The ProtoSleepNet abstract was available through its institutional record and PubMed. Its full text was not accessible through the available web fetch. Do not claim that MIST-Evidence differs from every detail of that method without a full-method review. Additional recent shapelet, prototype, and additive temporal papers also overlap individual elements. A final submission needs an updated full-text search.

## Candidate contribution, not an established novelty claim

The proposed research question concerns a constrained evidence path combining:

1. immutable real training snippets instead of freely drifting latent prototypes displayed after projection;
2. bounded local learned dissimilarity with mandatory observed shape, spectrum, and envelope contributions;
3. additive, lag-specific emission accounting with waveform evidence separated from CRF transition effects;
4. separate tests for arithmetic fidelity, event validity, model interventions, and external transfer.

The numerical lower bound on each observed contribution and the exact score decomposition are mathematical properties of this implementation. Whether the combination is novel enough, whether these constraints improve useful explanations, and whether they preserve competitive accuracy remain research questions.

No claim is made that this work is the first morphology-aware, prototype-based, explainable, or hierarchical sleep-staging model. No SOTA score is taken from an incompatible channel/pretraining/test protocol.

## References

Pei, Y., Xu, J., Yu, F., Zhang, L., & Luo, W. (2025). WaveSleepNet: An interpretable network for expert-like sleep staging. *IEEE Journal of Biomedical and Health Informatics, 29*(2), 1371-1382. https://doi.org/10.1109/JBHI.2024.3498871

Gagliardi, G., Garcia Ciudad, J., Micca, L., Kornum, B. R., Gilat, M., Alfeo, A. L., Cimino, M. G. C. A., & De Vos, M. (2026). *Prototype-based sleep micro-structure learning for explainable and robust multimodal recognition of sleep-related conditions* [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-9169987/v1

Grabocka, J., Schilling, N., Wistuba, M., & Schmidt-Thieme, L. (2014). Learning time-series shapelets. In *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 392-401). https://doi.org/10.1145/2623330.2623613

Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019). This looks like that: Deep learning for interpretable image recognition. *Advances in Neural Information Processing Systems, 32*. https://proceedings.neurips.cc/paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html

Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R., & Hinton, G. E. (2021). Neural additive models: Interpretable machine learning with neural nets. *Advances in Neural Information Processing Systems, 34*. https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html

Aggarwal, K., Khadanga, S., Joty, S. R., Kazaglis, L., & Srivastava, J. (2018). *A structured learning approach with neural conditional random fields for sleep staging* [Preprint]. arXiv. https://arxiv.org/abs/1807.09119

Park, K., Na, Y., Choi, Y., Ryu, H., Shin, H.-W., & Kim, H.-S. (2026). *SleepMaMi: A universal sleep foundation model for integrating macro- and micro-structures* [Preprint]. arXiv. https://arxiv.org/abs/2602.07628

Perslev, M., Darkner, S., Kempfner, L., Nikolic, M., Jennum, P. J., & Igel, C. (2021). U-Sleep: Resilient high-frequency sleep staging. *npj Digital Medicine, 4*, Article 72. https://doi.org/10.1038/s41746-021-00440-5

## Project evidence behind the redesign

The parent repository at commit `25f76fe80595f65b019e897de62e6efff3adb879` and the user-supplied experiment logs are the basis for the historical conclusions, not the above publications. In particular:

- MorphSpec frozen-probe folds 5-9: +0.077658 versus frozen random and +0.059716 versus frozen original MorphMAE, positive in 5/5 folds for each.
- Staged MorphSpec folds 10-14: -0.015598 versus trainable A1, positive in 0/5 folds.
- Frozen MorphSpec folds 15-19: +0.061066 versus frozen random, +0.038392 versus original frozen MorphMAE, and -0.020025 versus trainable A1.

These summaries do not constitute independent new results for MIST-Evidence, prove clinical event detection, or establish a causal fine-tuning-erasure mechanism.
