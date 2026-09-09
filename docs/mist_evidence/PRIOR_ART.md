# Closest prior work and novelty limits

Focused primary-source check: 9 September 2026. This is not an exhaustive systematic review and does not justify a 'first' or 'completely novel' claim. Publisher/index abstracts were sufficient to establish the overlap below; a full method-and-code comparison is still required before a paper novelty claim.

| Prior work | Established overlap | Consequence |
|---|---|---|
| ProtoPNet (Chen et al., 2019) | Prototype-part evidence for classification | Part-based additive evidence is not new. |
| WaveSleepNet (Pei et al., 2025) | Waveform prototypes, presence/proportion-based staging, visual explanation | 'Classify sleep from visible waveform prototypes' is already a published idea. |
| ProtoSleepNet (Gagliardi et al., 2026 preprint) | Prototype-based sequence-to-sequence sleep staging and prototype-grams | Adding a prototype sequence module is not a sufficient novelty claim. |
| SleepMaMi (Park et al., 2026 preprint) | Hierarchical micro/macro representation learning with MAE and contrastive objectives | Generic morphology + macro context + SSL is already covered. |
| UniShape (Liu et al., 2026) | Multiscale shape representations and prototype pretraining for time-series classification | Multiscale shape matching alone is not new. |
| Learning to Detect Sleep Micro-Events From Coarse Sleep Stage Annotations (WSSMED) | Weakly supervised prototype/cluster event discovery from stage labels | Coarse-label event discovery cannot be claimed as new here. Indexed record: https://pubmed.ncbi.nlm.nih.gov/41259170/ . Bibliographic details require full-record verification. |
| RAGSleepNet (2026 publisher record) | Exemplar retrieval, descriptor-driven clinical cards and sleep staging evidence | Retrieval plus physiological descriptors is also close prior art. Source: https://www.nrfhh.com/index.php/journal/article/view/683 . No full-text comparative evaluation was possible in this session. |
| ProtSleepNet (distinct name from ProtoSleepNet, publisher online record) | Gabor/wavelet features and structured waveform-prototype matching | Additional close overlap. Publisher lists a December 2026 issue; do not conflate this with Gagliardi's ProtoSleepNet. DOI: 10.1016/j.bspc.2026.111276. |

## Candidate distinction to investigate, not assert as proven novel

The proposed combination keeps literal raw waveform identities fixed while their embeddings remain live under a shared crop-local encoder, combines a declared measured/learned metric, and exports an exact competing-class evidence decomposition through an optional CRF. Its guarantees are narrow and testable: source identity, local support, no hidden residual bypass, and numerical score completeness. These invariants could support a useful systems/method contribution if the closest methods do not already provide the same contract and if accuracy/explanation experiments support its value.

CRF forward/backward and the marginal log-odds identity are standard mathematical consequences, not claimed new theorems. Clinical faithfulness and robustness do not follow from the identity.

## References

Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019). This looks like that: Deep learning for interpretable image recognition. *Advances in Neural Information Processing Systems, 32*. https://proceedings.neurips.cc/paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html

Eldele, E., Chen, Z., Liu, C., Wu, M., Kwoh, C.-K., Li, X., & Guan, C. (2021). An attention-based deep learning approach for sleep stage classification with single-channel EEG. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29*, 809-818. https://doi.org/10.1109/TNSRE.2021.3076234

Gagliardi, G., Garcia Ciudad, J., Micca, L., Kornum, B. R., Gilat, M., Alfeo, A. L., Cimino, M. G. C. A., & De Vos, M. (2026). *Prototype-based sleep micro-structure learning for explainable and robust multimodal recognition of sleep-related conditions* [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-9169987/v1

Liu, Z., Wang, Y., Li, B., Zheng, J., Eldele, E., Wu, M., & Ma, Q. (2026). A unified shape-aware foundation model for time series classification. *Proceedings of the AAAI Conference on Artificial Intelligence, 40*(28), 23972-23980. https://doi.org/10.1609/aaai.v40i28.39574

Park, K., Na, Y., Choi, Y. R., Ryu, H., Shin, H.-W., & Kim, H.-S. (2026). *SleepMaMi: A universal sleep foundation model for integrating macro- and micro-structures* (Version 2) [Preprint]. arXiv. https://arxiv.org/abs/2602.07628v2

Pei, Y., Xu, J., Yu, F., Zhang, L., & Luo, W. (2025). WaveSleepNet: An interpretable network for expert-like sleep staging. *IEEE Journal of Biomedical and Health Informatics, 29*(2), 1371-1382. https://doi.org/10.1109/JBHI.2024.3498871

Perslev, M., Darkner, S., Kempfner, L., Nikolic, M., Jennum, P. J., & Igel, C. (2021). U-Sleep: Resilient high-frequency sleep staging. *npj Digital Medicine, 4*, Article 72. https://doi.org/10.1038/s41746-021-00440-5

Sutton, C., & McCallum, A. (2012). An introduction to conditional random fields. *Foundations and Trends in Machine Learning, 4*(4), 267-373. https://doi.org/10.1561/2200000013

PyTorch Contributors. (n.d.). *Reproducibility*. PyTorch documentation. https://docs.pytorch.org/docs/stable/notes/randomness.html

Clinical event-data sources to inspect before use: DREAMS https://zenodo.org/records/2650142 ; MASS-EX annotation-only release https://doi.org/10.5281/zenodo.19087197 . The latter's epoch rationales are not automatically equivalent to sample-level event boundaries. Dataset access and licensing must be checked before import or redistribution.
