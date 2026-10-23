# MIST-Sleep — Original 13-page Research Proposal (Text Mirror)

> **Source status:** This file is a plain-text extraction of the original 13-page user-supplied PDF, not a replacement for the original PDF's layout, tables, visual rendering, or binary. **Do not silently treat this transcription as a newly revised proposal.** Preserve the original scientific wording and consult the original PDF for layout-dependent ambiguities.
>
> **Original attachment:** `MIST-Sleep_ Compute-Efficient Morphology-Invariant Transfer for Cross-Dataset EEG Sleep Staging.docx.pdf`  
> **Original file SHA-256:** `4ab676da46b3ef8a49d275d336df5c251cf09cd5aaf98dc8e735cc6a2b1fcadf`  
> **Original length:** 201,572 bytes / 13 pages  
> **Intended binary path:** `docs/mist_sleep/original_proposal/MIST-Sleep_Original_Research_Proposal.pdf`  
> **Source:** Original user-provided attachment in the ChatGPT conversation.  
> **Document scope:** Original MIST-Sleep method-paper proposal, not the subsequent MIST-COVE explanation phases.

---

## Original PDF page 1 of 13

```text
                                       Research Project Proposal


   MIST-Sleep: Compute-Efficient Morphology-Invariant Transfer for Cross-Dataset EEG Sleep
                                          Staging
1. Executive Summary
Automatic sleep staging models can perform well within a single dataset but often degrade when
transferred to new cohorts, devices, montages, or recording environments. This proposal studies whether
explicit EEG morphology invariance can reduce that transfer degradation. The project is positioned as a
compute-efficient small-model study, not as a direct replacement for large sleep foundation models. The
main contribution is MIST-Sleep, a morphology-invariant transfer framework that tests whether local
waveform structure can provide a more transferable inductive bias for sleep staging.
The proposal acknowledges the remaining reviewer-level risks before experiments begin. In particular, it
strengthens the method by adding a formal invariance statement for the MPI objective, separating MPI
from supervised contrastive learning, fixing prototype-layer design choices, adding shift-specific
predictions, making domain-adaptation and foundation-model comparisons part of the evaluation plan,
adding at least one unusual or clinical target where feasible, and adding subject-level bootstrap confidence
intervals.
2. Core Research Question
Main question: Do explicitly morphology-invariant EEG representations improve cross-dataset
transferability and label-efficient fine-tuning in automatic sleep staging by learning waveform structures
that are stable across datasets but still discriminative for sleep stages?
In simpler terms, the project asks whether a model that learns reusable EEG microstructure can generalize
better than a standard sleep-staging model when the target dataset differs in recording system, montage,
cohort, or clinical condition.
3. Literature-Based Motivation
The transferability study by Supratak and Haddawy motivates the problem: sleep-staging transfer is
strongly affected by dataset characteristics such as recording environment, channel configuration, and
subject condition (Supratak & Haddawy, 2023). This proposal extends that line of work by asking
whether a better representation, specifically a morphology-invariant one, can reduce the observed domain
transfer gap.
AttnSleep is used as the main baseline because it is a strong single-channel EEG architecture with
convolutional feature extraction and temporal context modeling (Eldele et al., 2021). SleepConFormer
motivates the use of multi-resolution representations, but the proposed study adds a stricter question:
whether morphology-preserving invariance adds value beyond generic multi-scale feature extraction
(Afshari, 2025).
TDSAN and related domain-adaptation methods show that direct source-to-target transfer is often
insufficient and that explicit feature alignment can help (Gao et al., 2023). Therefore, a domain-adaptation
baseline is mandatory in this proposal. SleepFM and SleepMaMi motivate the foundation-model
```

## Original PDF page 2 of 13

```text
comparison, but the current project is framed around label-efficient and compute-efficient small-model
transfer rather than large-scale PSG pretraining (Thapa et al., 2026; Park et al., 2026). SLEEPYLAND
motivates a fair evaluation philosophy based on harmonized preprocessing, diverse targets, and
transparent failure reporting (Dei Rossi et al., 2025).
4. Refined Gap and Novelty
The proposal no longer relies only on the statement that morphology has not been tested for
transferability. That would be an evaluation difference rather than a methodological contribution. The
refined gap is more specific: prior sleep-staging models have not formally tested whether a
morphology-preserving invariance objective can learn EEG embeddings that are stable under
physiologically safe transformations, less predictive of dataset identity, and still discriminative for sleep
stages.
The named methodological contribution is the Morphology-Preserving Invariance objective, abbreviated
as MPI. The novelty is not merely a combination of multi-scale filters, prototypes, and consistency loss.
The novelty is the explicit definition of which waveform transformations should preserve the
representation, which transformations should not be used as invariances, and how this invariance is tested
under external dataset shift.
5. Formal Rationale for MPI
Assumption: A sleep EEG epoch x can be viewed as containing two components: a morphology
component m, which includes local waveform shape, rhythm, duration, and frequency-localized structure,
and a nuisance component n, which includes amplitude scale, mild device noise, baseline drift, and other
recording-specific variation. The target label y should depend more strongly on m than on n. The domain
identity d should depend heavily on n and sometimes on cohort composition.
The sleep-theory boundary is important. This proposal does not assume that morphology is invariant
under every kind of dataset shift. It treats morphology as relatively stable under acquisition-level
nuisance, not universally stable across all populations. Acquisition-level nuisance includes amplifier
response, electrode impedance, montage differences, sampling rate, mild noise, amplitude scaling, and
preprocessing choices. MPI is designed to reduce sensitivity to these shifts when they change the
observed signal without intentionally changing the underlying sleep graphoelement. In contrast, age,
pediatric development, sleep apnea, medication, neurological disease, and severe sleep fragmentation can
change the morphology itself. In those cases, morphology is part of the target physiology rather than a
nuisance factor, and MPI is not expected to remove the shift. This distinction makes the hypothesis
clinically bounded: MIST-Sleep is expected to help most under recording-domain shift and may fail or
require adaptation under physiology-domain shift.
This boundary also clarifies the stage-level theory. N2 and N3 provide the strongest theoretical basis
because clinical sleep scoring relies on graphoelements such as spindles, K-complexes, and slow waves.
N1 is weaker because it is often a transitional stage without a single defining graphoelement. REM is also
more limited in single-channel EEG because human scoring uses a combination of low-amplitude
mixed-frequency EEG, rapid eye movements from EOG, and low chin EMG tone. Therefore, the proposal
expects strongest morphology-driven gains in N2 and N3, and treats improvements in N1, REM, and
arousal-related patterns as empirical rather than guaranteed (AASM, 2023; Silber et al., 2007).
```

## Original PDF page 3 of 13

```text
MPI is designed around the following invariance principle: for transformations t in the
morphology-preserving set T_preserve, the transformation may alter nuisance variation n but should not
intentionally change morphology m or label y. Therefore, a transferable morphology encoder f_theta
should satisfy f_theta(x) close to f_theta(t(x)) and q_theta(x) close to q_theta(t(x)), where q_theta is the
prototype assignment distribution.
Informal domain-shift statement: if source and target domains differ mainly through nuisance
transformations that are contained in or approximated by T_preserve, then reducing MPI should reduce
sensitivity to those nuisance shifts. In domain-adaptation language, this should reduce the discrepancy
between source and target morphology embeddings while preserving stage separability. This is not
claimed as a proof of universal transfer. It is a testable assumption that may fail under strong concept
shift, such as severe pathology, age-related morphology changes, or montage changes that alter the
observed waveform itself.
Training form: L_MPI = || norm(f_theta(x)) - norm(f_theta(t(x))) ||2 + KL(q_theta(x) || q_theta(t(x))). The
embedding term encourages geometric stability, while the assignment term encourages semantic stability
of the learned morphology units. Because q_theta is derived from f_theta, the two terms are related, but
they test different behaviors. The embedding term controls continuous drift, while the assignment term
controls prototype-level switching. Both will be ablated.
6. Proposed Model: MIST-Sleep
MIST-Sleep is an AttnSleep-style sleep-staging model with a morphology tokenizer, a shared prototype
layer, and the MPI objective. The final classifier predicts Wake, N1, N2, N3, and REM from
morphology-aware epoch embeddings and temporal context.
6.1 Architecture Components
1. Input and preprocessing: 30-second single-channel EEG epochs are resampled, filtered, z-score
     normalized by subject or recording, and segmented using subject-independent splits.
2. Morphology tokenizer: each epoch is divided into short overlapping windows to capture local
    waveform units rather than only whole-epoch statistics.
3. Multi-scale encoder: parallel or hierarchical temporal filters capture short transients, spindle-band or
    rhythmic patterns, and slower waves.
4. Shared prototype layer: a single prototype bank is shared across source and target datasets so that
    prototypes represent transferable recurring waveform units rather than dataset-specific units.
This shared-prototype assumption is clinically plausible for healthy adult cohorts where the same broad
waveform generators are expected to appear across datasets. It is weaker for pediatric, elderly, medicated,
or pathological cohorts, where the true morphology distribution may change. The proposal therefore treats
prototype sharing as a testable assumption rather than a guaranteed property.
5. Temporal context encoder: epoch-level embeddings are passed into an AttnSleep-style temporal
    module or lightweight transformer before classification.
6.2 Prototype Layer Specification
```

## Original PDF page 4 of 13

```text
The prototype layer is fixed before the main experiments for reproducibility. The default configuration
will use K = 64 prototypes, with a sensitivity analysis at K = 32 and K = 128. Prototypes are initialized
using k-means on morphology-token embeddings from the source training set after a short warm-up stage.
Assignments are softmax-based with temperature tau = 0.1 by default. Prototypes are shared across all
domains and updated by gradient descent with a diversity regularizer. An exponential moving average
update will be tested only as an ablation, not as the main configuration.
The prototype layer will not be described as clinical by default. A prototype will only be called
spindle-like, K-complex-like, or slow-wave-like if it passes the quantitative morphology validation rules
described later. Otherwise, it will be called a learned recurring waveform unit.
7. Loss Terms and Required Ablations
A key reviewer concern is that MPI and supervised contrastive learning may overlap. This proposal
therefore treats their disentanglement as mandatory rather than optional. MPI is transformation-based and
label-free within each sample. It asks whether the same EEG epoch keeps the same morphology
representation after a safe transformation. Supervised contrastive learning is label-based and class-level. It
asks whether samples from the same stage form coherent clusters. These are related but not identical.
 Ablation                            Purpose                              Expected interpretation
 Stage loss only                     Baseline training objective.         Shows ordinary supervised
                                                                          performance.
 Stage loss plus SupCon only         Tests class-level representation     If this matches MPI, MPI may
                                     clustering.                          be redundant.
 Stage loss plus MPI only            Tests transformation-based           If this improves transfer, MPI
                                     morphology stability.                adds information beyond class
                                                                          labels.
 Stage loss plus MPI plus            Tests whether the two objectives     Main configuration only if it
 SupCon                              are complementary.                   improves transfer or stability
                                                                          beyond either term alone.
 Embedding consistency only          Tests the continuous part of         Shows whether L2 or cosine
                                     MPI.                                 stability is sufficient.
 Prototype-assignment                Tests the discrete semantic part     Shows whether prototype
 consistency only                    of MPI.                              stability is needed beyond
                                                                          embedding stability.


8. Augmentation Policy and Shift Predictions
MPI will only use transformations that are expected to preserve waveform morphology. The
transformation set is not chosen only for empirical convenience. It is chosen based on the assumption that
many realistic recording shifts affect nuisance factors without changing the underlying sleep-stage
morphology.
```

## Original PDF page 5 of 13

```text
 Transformation group                Use in MPI training                 Reason
 Small amplitude scaling and         Allowed                             Simulates gain and baseline
 offset changes                                                          variation while preserving
                                                                         waveform shape.
 Mild Gaussian noise and             Allowed                             Simulates low-level acquisition
 baseline wander                                                         noise without destroying
                                                                         morphology.
 Small temporal shift                Allowed                             Preserves local waveform
                                                                         structure while changing
                                                                         alignment.
 Mild bandpass jitter                Restricted                          Allowed only within clinically
                                                                         reasonable bands because strong
                                                                         filtering can remove
                                                                         morphology.
 Mixup, heavy time warping,          Disallowed for MPI                  These can change or remove the
 large masking, strong frequency                                         morphology itself. They may be
 masking                                                                 used only in stress tests.


Predicted strengths: MPI should help most under amplitude, mild noise, baseline drift, and
acquisition-style shifts. It may partly help under sampling-rate or preprocessing differences if the
waveform structure is preserved after harmonization. Predicted weaknesses: MPI may not fully solve
strong montage shift, severe pathology, pediatric-to-adult shift, or cohort shifts where the relationship
between waveform morphology and sleep stage changes.
Preprocessing control: to avoid the criticism that better preprocessing alone explains the result, all models
will use the same standardized preprocessing pipeline. A preprocessing ablation will compare basic
preprocessing with a stronger normalization pipeline. If stronger preprocessing removes the gap, the paper
will report that explicitly.
9. Research Questions and Hypotheses
 RQ                                  Question                            Primary test
 RQ1                                 How large is the cross-dataset      Target-supervised Macro-F1
                                     transfer gap for standard           minus source-trained
                                     sleep-staging models?               target-tested Macro-F1.
```

## Original PDF page 6 of 13

```text
RQ                                Question                           Primary test
RQ2                               Does MIST-Sleep reduce the         External-target Macro-F1 and
                                  domain transfer gap compared       transfer-gap reduction.
                                  with standard,
                                  capacity-matched, multi-scale,
                                  DA, and foundation-model
                                  baselines?
RQ3                               Does MPI add information           MPI-only, SupCon-only, and
                                  beyond supervised contrastive      combined-objective ablations.
                                  learning?
RQ4                               Are MIST-Sleep embeddings          Domain classifier accuracy,
                                  more domain-invariant while        MMD/CORAL, linear-probe
                                  preserving stage separability?     Macro-F1, CKA, and class
                                                                     separability.
RQ5                               Does MIST-Sleep improve            Fine-tuning curves at 1%, 5%,
                                  label-efficient adaptation?        10%, 20%, and 50% target
                                                                     labels.
RQ6                               When does                          Dataset-level, subgroup-level,
                                  morphology-invariant transfer      and clinical-shift analysis.
                                  fail or cause negative transfer?


Hypothesis                        Rationale                          Falsification signal
H1. Dataset shift will reduce     Recording environment,             No meaningful transfer gap
performance.                      montage, and cohort differences    under external targets.
                                  alter EEG distributions.
H2. MIST-Sleep will reduce the    Morphology embeddings should       No external Macro-F1 or
transfer gap.                     be less sensitive to nuisance      transfer-gap improvement over
                                  variation.                         controls.
H3. MPI will add value beyond     MPI is instance-level              MPI-only and combined
SupCon.                           transformation consistency,        objectives do not improve over
                                  while SupCon is class-level        SupCon-only.
                                  clustering.
H4. MIST-Sleep will be more       Useful transfer requires           Domain invariance improves but
domain-invariant without losing   removing dataset identity while    Macro-F1 or linear-probe MF1
stage separability.               preserving stage identity.         drops.
H5. MIST-Sleep will improve       Reusable morphology should         Fine-tuning curves overlap with
low-label fine-tuning.            reduce the target-label burden.    or trail baselines.
```

## Original PDF page 7 of 13

```text
 Hypothesis                          Rationale                            Falsification signal
 H6. Negative transfer may           Some shifts change the observed      Averaged results hide failures in
 appear under strong clinical or     morphology itself, not only          specific cohorts or subgroups.
 montage shift.                      nuisance factors.


10. Dataset and Evaluation Scope
EDF-78 to EDF-20 will be retained only as a pilot and internal sanity check. The main paper-level
evidence must come from external transfer. The final dataset choice will depend on access and
preprocessing feasibility, but the proposal commits to at least three external targets if compute and data
access allow, plus one unusual or clinical target where feasible.
 Level                               Candidate datasets                   Reason for inclusion
 Pilot internal transfer             Sleep-EDF-78 to Sleep-EDF-20         Reproduces the original idea and
                                                                          checks whether the pipeline
                                                                          works.
 Main external target 1              SHHS                                 Large external PSG cohort and
                                                                          common benchmark for transfer
                                                                          evaluation.
 Main external target 2              ISRUC                                Different acquisition
                                                                          environment and useful external
                                                                          validation setting.
 Main external target 3              MASS or CFS                          Adds another external target and
                                                                          prevents overfitting conclusions
                                                                          to one target.
 Unusual or clinical target          HMC, APPLES, Dreem                   Tests whether morphology
                                     DOD-H/DOD-O, pediatric data,         invariance holds beyond
                                     or another accessible                well-trodden adult PSG
                                     clinical/wearable cohort             benchmarks. This target is
                                                                          important for negative transfer
                                                                          analysis.


If an unusual target cannot be accessed, the paper will clearly state that limitation and will not overclaim
top-tier clinical generalization. In that case, the claim will be limited to compute-efficient transfer across
common adult PSG benchmarks.
11. Baselines and Positioning
The model comparison plan is designed to prevent weak novelty claims. Each baseline removes one
possible alternative explanation for improvement.
```

## Original PDF page 8 of 13

```text
 Model                                Role                                 Why it is needed
 AttnSleep                            Main baseline                        Tests improvement over a strong
                                                                           single-channel EEG model.
 Capacity-matched AttnSleep           Parameter-count control              Tests whether gains come only
                                                                           from extra parameters.
 Generic multi-scale AttnSleep        Architecture ablation                Tests whether gains come from
                                                                           multi-scale CNNs rather than
                                                                           MPI.
 MIST-Sleep                           Proposed model                       Tests morphology tokenization,
                                                                           shared prototypes, and MPI.
 DANN, CDAN, or                       Mandatory DA comparison              Tests morphology invariance
 TDSAN-style baseline                                                      against explicit domain
                                                                           adaptation.
 SleepFM or SleepMaMi                 Foundation-model reference           Tests whether a large pretrained
 comparator if accessible                                                  representation dominates the
                                                                           small-model approach. If full
                                                                           fine-tuning is not possible, use
                                                                           embeddings, linear probing,
                                                                           adapter tuning, or clearly label it
                                                                           as contextual comparison.


The paper will explicitly own its positioning: MIST-Sleep is intended as a label-efficient and
compute-efficient small-model approach. If a fine-tuned foundation model is much stronger, the
contribution becomes efficiency, interpretability, and transfer under fixed resource constraints rather than
raw state-of-the-art dominance.
12. Experimental Settings
1. Within-dataset training: train and test each model within each dataset using subject-independent splits.
2. Direct transfer: train on source and evaluate directly on target without target labels.
3. Limited-label fine-tuning: fine-tune using 1%, 5%, 10%, 20%, and 50% of target training subjects,
    never target test subjects.
4. Domain-adaptation comparison: train a reproducible DA baseline with labeled source and unlabeled
    target data.
5. Foundation-model comparison: evaluate SleepFM or SleepMaMi under a feasible linear-probe, adapter,
    or fine-tuning setup if accessible.
6. Representation analysis: report domain classifier accuracy, MMD or CORAL, linear-probe Macro-F1,
    CKA, class separability, and visualization.
```

## Original PDF page 9 of 13

```text
7. Negative transfer analysis: report target-level and subgroup-level failures, especially under clinical,
    age, pathology, or wearable shift.
13. Decision Rules and Claim Hierarchy
To avoid unfalsifiable interpretation, the paper will separate confirmatory claims from exploratory
observations. Confirmatory claims are tested with fixed thresholds and corrected statistics. Exploratory
claims are reported as hypotheses for future work.
 Claim type                 Claim                       Support criterion           Failure or caution
                                                                                    criterion
 Confirmatory               Transfer benefit            MIST-Sleep improves         Gains are below
                                                        average external-target     threshold or occur only
                                                        Macro-F1 by at least 2      in one target.
                                                        absolute points or
                                                        reduces transfer gap by
                                                        at least 10% relative to
                                                        AttnSleep, with no
                                                        meaningful drop in
                                                        target-supervised
                                                        performance.
 Confirmatory               MPI adds value              MPI-only or                 SupCon-only matches
                                                        MPI-plus-SupCon             MPI across settings.
                                                        improves over
                                                        SupCon-only in
                                                        external transfer or
                                                        label efficiency.
 Confirmatory               Stage robustness            Macro-F1 improves           Accuracy improves but
                                                        and at least two            minority-stage F1 falls.
                                                        difficult or
                                                        morphology-relevant
                                                        stages improve without
                                                        hiding N1 degradation.
 Confirmatory               Representation              Domain classifier           Embeddings become
                            mechanism                   accuracy decreases and      domain-invariant only
                                                        MMD/CORAL                   by losing stage
                                                        decreases while             information.
                                                        linear-probe Macro-F1
                                                        remains within 1 point
                                                        of the baseline or
                                                        improves.
```

## Original PDF page 10 of 13

```text
 Claim type                 Claim                      Support criterion          Failure or caution
                                                                                  criterion
 Confirmatory               Label efficiency           MIST-Sleep reaches         Fine-tuning curves
                                                       90% of                     overlap with baselines.
                                                       target-supervised
                                                       Macro-F1 with fewer
                                                       target labels or has
                                                       higher area under the
                                                       label-efficiency curve.
 Exploratory                Clinical morphology        Prototype validation       Prototypes are reported
                            interpretation             supports spindle-like,     only as learned
                                                       K-complex-like, or         recurring waveform
                                                       slow-wave-like labels.     units.


14. Quantitative Morphology Validation
Nearest-neighbor visualization alone is not enough. The validation plan therefore includes quantitative
rules. These rules are not used to supervise training unless explicitly stated. They are used to interpret
learned prototypes after training.
 Candidate interpretation            Quantitative check                  Reporting rule
 Spindle-like                        Dominant power in                   Report percentage of
                                     approximately 11 to 16 Hz with      high-activation segments
                                     duration roughly 0.5 to 2.0         satisfying this criterion.
                                     seconds in high-activation
                                     segments.
 Slow-wave-like                      Dominant low-frequency              Report spectral ratio and
                                     activity in the delta range with    time-domain amplitude/duration
                                     large-amplitude slow                statistics.
                                     morphology.
 K-complex-like                      Sharp high-amplitude transient      Report waveform-shape
                                     with a duration around or above     statistics and
                                     0.5 seconds and a clear             manual/algorithmic agreement if
                                     negative-positive complex when      available.
                                     visible.
 Cross-dataset prototype stability   Compare per-prototype               Report whether prototypes
                                     activation distributions across     activate similarly across source
                                     datasets using Jensen-Shannon       and target domains.
                                     divergence and prototype-level
                                     CKA.
```

## Original PDF page 11 of 13

```text
If the criteria do not support clinical naming, the paper will not force clinical labels. This preserves
credibility and prevents the interpretability claim from collapsing if prototypes are useful but not
clinically aligned.
15. Statistical Analysis
Final claims will use five random seeds where compute allows. Pilot runs may use fewer seeds only for
development. Results will report mean, standard deviation, and subject-level bootstrap 95% confidence
intervals. The bootstrap will resample subjects, not epochs, because subject-level uncertainty is more
relevant in medical AI.
Paired tests will be used across matched subjects, folds, or transfer settings. Wilcoxon signed-rank tests
will be preferred when normality is unclear. Holm-Bonferroni correction will be applied only to
confirmatory comparisons. Exploratory comparisons will be labeled clearly to avoid loss of power and
overclaiming.
16. Negative Transfer and Failure Analysis
The proposal explicitly expects that morphology invariance may fail when the source-target difference is
physiological rather than only acquisition-related. For example, strong pathology, pediatric EEG, elderly
cohorts, severe apnea, medication effects, or wearable montages may change the observed waveform
morphology itself rather than only nuisance factors. If MIST-Sleep helps standard adult PSG transfer but
fails under these shifts, this will be reported as a clinically meaningful negative result rather than hidden
as an implementation failure.
The failure analysis will report which shifts are helped and which shifts are not helped. This includes
amplitude/noise shift, sampling-rate or preprocessing shift, montage shift, age or cohort shift, and
pathology shift. This makes the contribution clinically safer and more honest.
17. Timeline
 Phase                               Tasks                               Output
 Month 1                             Finalize datasets, preprocessing,   Verified pipeline and pilot
                                     splits, and baseline AttnSleep      results.
                                     pipeline. Run EDF pilot.
 Month 2                             Implement MIST-Sleep,               Stable model with internal
                                     prototype layer, MPI, and loss      ablations.
                                     ablations.
 Month 3                             Run external transfer on SHHS,      External transfer and
                                     ISRUC, MASS or CFS, and the         negative-transfer tables.
                                     unusual target if feasible.
```

## Original PDF page 12 of 13

```text
 Phase                              Tasks                              Output
 Month 4                            Run low-label fine-tuning, DA      Main experimental results.
                                    baseline, foundation-model
                                    comparator if accessible, and
                                    representation analysis.
 Month 5                            Run five-seed final experiments,   Supervisor-ready journal
                                    bootstrap CIs, morphology          manuscript.
                                    validation, and manuscript
                                    writing.




18. Final Refined Project Statement
This project will test whether a compute-efficient morphology-invariant EEG model can reduce
cross-dataset sleep-staging degradation while requiring fewer target labels. The proposed method,
MIST-Sleep, is built around a formal Morphology-Preserving Invariance objective, a shared morphology
prototype layer, and a rigorous evaluation protocol with capacity-matched, multi-scale,
domain-adaptation, and foundation-model comparisons. The study is designed to be falsifiable, because it
defines decision rules before training, reports negative transfer, and validates morphology quantitatively
rather than relying only on visual examples.
References
American Academy of Sleep Medicine. (2023). The AASM manual for the scoring of sleep and
      associated events: Rules, terminology and technical specifications, version 3. American Academy
      of Sleep Medicine.
Silber, M. H., Ancoli-Israel, S., Bonnet, M. H., Chokroverty, S., Grigg-Damberger, M. M., Hirshkowitz,
        M., Kapen, S., Keenan, S. A., Kryger, M. H., Penzel, T., Pressman, M. R., & Iber, C. (2007). The
        visual scoring of sleep in adults. Journal of Clinical Sleep Medicine, 3(2), 121-131.
Afshari, M. (2025). SleepConFormer: Contrastive temporal modeling of sleep with multi-resolution EEG
       representation learning. SSRN preprint.
Dei Rossi, A. D., et al. (2025). SLEEPYLAND: Trust begins with fair evaluation of automatic sleep
      staging models. npj Digital Medicine.
Eldele, E., Chen, Z., Liu, C., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2021). An attention-based deep
        learning approach for sleep stage classification with single-channel EEG. IEEE Transactions on
        Neural Systems and Rehabilitation Engineering, 29, 809-818.
        https://doi.org/10.1109/TNSRE.2021.3076234
Gao, D. R., Li, J., Wang, M. Q., et al. (2023). Automatic sleep staging of single-channel EEG based on
      domain adversarial neural networks and domain self-attention. Frontiers in Neuroscience, 17,
      1143495.
```

## Original PDF page 13 of 13

```text
Park, K., Na, Y., Choi, Y., Ryu, H., Shin, H. W., & Kim, H. S. (2026). SleepMaMi: A universal sleep
       foundation model for integrating macro- and micro-structures. arXiv:2602.07628.
Supratak, A., & Haddawy, P. (2023). Quantifying the impact of data characteristics on the transferability
       of sleep stage scoring models. Artificial Intelligence in Medicine, 139, 102540.
Thapa, R., et al. (2026). A multimodal sleep foundation model for disease prediction. Nature Medicine.
```

