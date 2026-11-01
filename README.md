# Class-Incremental Continual Learning for TinyML CubeSat Intrusion Detection

A from-scratch (pure NumPy, no PyTorch/TensorFlow) implementation and full
experimental evaluation of **class-incremental continual learning** for
onboard, memory-constrained intrusion detection on
CubeSats. The **MLP is the primary backbone** -- it directly processes the
tabular telemetry feature vector. A compact GRU and 1-D CNN are evaluated as
**architectural stress tests**, not as genuine temporal sequence models (the
inputs are tabular feature vectors, not raw telemetry sequences). Nine
continual-learning configurations (replay, online EWC, LwF, BiC-lite,
iCaRL-lite, combined replay+EWC, naive fine-tuning, and two offline joint
references) are compared under the strict **class-incremental**
protocol: no task ID is ever available at inference time, and every sample
must be classified by a single, dynamically-growing shared softmax head.

**Scope and claims.** This repository is a resource-aware, CubeSat-first
benchmark and negative-result study, not a claim that the combined `full`
method wins. The paper reports two co-headline estimands: the random-split IID
benchmark and the fair forward-temporal deployment-facing estimate. The
cold-start temporal arm is a support-control diagnostic; feature-provenance
results are a separate reliance probe. Neither temporal arm is a deployment
guarantee.

This repository contains the full codebase; running `python3 run_all.py`
reproduces every number and figure referenced below directly from the
provided CSV data (no placeholder or hand-typed results anywhere in the
pipeline). The pipeline enforces **train → validation tuning → freeze
hyperparameters → final test**: no test data is ever used for
hyperparameter selection.

---

## 1. Problem setting and operational concept

CubeSats running NASA's cFS on COTS hardware face evolving cyber threats
(command flooding, data injection, defence impairment, storage exhaustion,
network-layer DDoS, etc.). Deployed IDS models are typically **static**:
once launched, their weights are frozen, and adapting to a newly discovered
attack requires re-uploading the full firmware image over a narrow
5-15 minute uplink window.

The system modelled here instead assumes a **human-in-the-loop, edge
adaptation** workflow:

1. **Space** -- the onboard IDS flags anomalous telemetry and downlinks it.
2. **Earth** -- a human analyst labels the new attack type.
3. **Earth -> Space** -- only a *small* labelled CSV (kilobytes, not
   megabytes) is uplinked.
4. **Space** -- the model runs a short continual-learning update onboard,
   incorporating the new attack type **without forgetting** the attack
   types learned so far.

This shifts the adaptation burden onto the edge hardware, eliminates the
need for full firmware retransmission, and keeps a human in the loop for
safety-critical security decisions.

### Why class-incremental, not task-incremental

A **task-incremental** protocol (task ID known at inference time, one
private output head per task) is the easier of the two standard continual-
learning protocols: the model is only ever asked to discriminate within a
small, known subset of classes. It is a poor model of an onboard IDS,
though, because a real classifier has no oracle telling it in advance
"this packet is either normal or a command-flood, nothing else" -- it must
decide among *everything it has ever learned* at once.

This repository therefore implements the harder, more realistic
**class-incremental** protocol: a single shared classification head grows
by a few output columns every time a new attack type is introduced, and at
evaluation time the model must pick the correct class out of *all* classes
learned so far, with no side information about which task a sample came
from. This exposes the well-known **class-incremental collapse** /
"task-recency bias" failure mode -- softmax cross-entropy on a new class's
samples actively pushes down the logits of every previously-learned class,
even without any weight drift -- which a task-incremental evaluation would
never surface. Section 7 shows this failure mode occurring, and the CMMU's
(partial) mitigation of it, directly in the computed results.

---

## 2. Datasets

| Dataset | Role | Files | Rows | Features | Classes |
|---|---|---|---|---|---|
| **CuCD-ID** | Primary | `data/CuCD-ID/noised_dataset.csv` | 22,465 | 22 (telemetry) | 5 (`normal`, `command_flooding`, `data_injection`, `defence_impairment`, `storage_exhaustion`) |
| **STIN** | Secondary / cross-domain stress test | `data/STIN/SAT20.csv` + `TER20.csv` (concatenated) | 209,564 | 30 (network flow) | 9 distinct attack types (`Syn_DDoS`, `UDP_DDoS`, `LDAP_DDoS`, `MSSQL_DDoS`, `Botnet`, `Portmap_DDoS`, `Web Attack`, `Backdoor`, `NetBIOS_DDoS`) |
| **SCAD** | External 4-class CIL validation (local only) | `data/space-cyber-anomaly-detection-dataset-main/data/raw/{baseline,attack_rwa,attack_rwb,attack_rwc}/` | 612k scenario rows | 80 leakage-controlled telemetry features | `normal`, `RWA`, `RWB`, `RWC` |

**Provenance and scope.** CuCD-ID is the labelled NOS3/cFS CubeSat
telemetry dataset of Fayyaz et al., *Data in Brief* (2026),
doi:10.1016/j.dib.2026.112598 (data: Mendeley
doi:10.17632/7n2d42pm3n.3). It was generated in a high-fidelity
software-in-the-loop digital twin; **all telemetry is simulated, not
captured from an orbiting spacecraft**. STIN (SAT20/TER20) is the
satellite/terrestrial flow dataset of Li et al., *IEEE Access* (2020),
doi:10.1109/ACCESS.2020.3041641 (github.com/kun9717/STIN-data-set). STIN
has no benign/normal class in the provided files, so it is used purely
as a larger-scale, higher-class-count **cross-domain stress test** to check
whether findings generalize beyond the primary dataset. A coarse STIN/MLP
sweep provides an initial hyperparameter prior only; **final
method × architecture hyperparameters are tuned on CuCD-ID validation data
and frozen** (`results/best_hparams_per_method_arch.json`) before any test
evaluation -- the primary dataset's test numbers are never used for tuning.

**External SCAD validation.** `experiments/run_scad.py` evaluates the
scenario-disjoint 3-task SCAD sequence (`normal+RWA`, `RWB`, `RWC`). The
split is over 120 trajectory groups, not files, so all attack intensities
and the matching baseline of a trajectory stay in one split. Features are
80 leakage-controlled telemetry signals, centred on each scenario's first
300 attack-free rows; time, GPS/orbit/position, scenario identifiers and
attack parameters are dropped. Run it with
`python3 experiments/run_scad.py --archs MLP --tag mlp_primary` (add
`--uplink-n 700 --tag mlp_up700` for the budget sensitivity). Results are
recorded in `DATALOG.md`; the raw dataset is not redistributed.

**Leakage caveat.** CuCD-ID includes a circular time field, CCSDS packet
identifiers, and 20-second sliding-window aggregates. The headline IID
random split can share overlapping windows between train and test, so those
numbers are benchmark estimates, not deployment guarantees. The
`leakage_ablation` experiment retrains models after removing each feature
group. The fair random-past temporal protocol is the deployment-facing
co-headline estimate; the cold-start protocol is a support control, and the
no-leaky variant is a feature-reliance probe. Neither turns the IID number
into a leakage-free estimate.

### Task construction (class-incremental protocol)

Every task introduces a set of classes that have **never appeared in any
earlier task** -- classes are never repeated. There is a single shared
classifier head; when a task is processed, `model.expand_head()` appends
one new output column per new class, and every future prediction (on any
task's test data) goes through the full, current-width head. Full logic is
in `src/data_utils.py` and `src/train.py`.

* **CuCD-ID -> 4 tasks, 5 classes total**:
  `T0_launch = {normal, command_flooding}` (2 new classes -- a valid
  classifier needs at least 2 classes to start from), then
  `T1 = {data_injection}`, `T2 = {defence_impairment}`,
  `T3 = {storage_exhaustion}` (1 new class each).
* **STIN -> 3 tasks, 9 classes total**: `T0 = {Syn_DDoS, UDP_DDoS,
  LDAP_DDoS}`, `T1 = {MSSQL_DDoS, Botnet, Portmap_DDoS}`,
  `T2 = {Web Attack, Backdoor, NetBIOS_DDoS}` (3 new classes per task).

**Uplink-budget-constrained training sets.** Reflecting the "small labelled
CSV" operational constraint, task 0 of each dataset is trained on a larger
pre-launch corpus (700 samples/class), while every *subsequent* task is
trained on only 150 samples/class -- a realistically small uplink payload.
Test sets are independent, held-out, and **not** subject to this budget
(they represent a ground-truth evaluation archive, never uplinked). Section
4 of `results/RESULTS_SUMMARY.md` reports the exact computed uplink payload
size in KB for every task: adaptation payload (raw float32 and packed int8)
**plus** the onboard calibration payload consumed by BiC fitting and
val-balanced Fisher estimation, with the combined total as the reported
uplink budget. Payload figures are **packed telemetry payload estimates**
(rows × features plus label bytes), excluding CSV text formatting,
quantization scale/zero-point parameters, and packet/protocol framing
overhead.

A feature normalizer (mean/std) is fit **only** on task 0's training data of
each dataset and then applied unchanged to every later task -- modelling
that the onboard preprocessing configuration is fixed at launch and cannot
be recalibrated using data from attacks that haven't been seen yet.

**Order-independent label handling.** Each `Task`'s labels are stored
*locally* (0..n_new_classes-1); `src/train.py` resolves them to the model's
actual global head-column indices dynamically, based on the order tasks are
*actually processed in*. This is what lets the task-order-sensitivity
ablation safely replay the very same `TaskSequence` in a different order
without corrupting label alignment (see `experiments/run_task_order.py`).

---

## 3. Architectures

All three backbones are implemented from scratch in NumPy
(`src/models.py`) with hand-derived forward *and* backward passes
(verified against finite-difference gradients in `tests/test_gradients.py`,
including growing-head gradient checks) and **one** shared, dynamically
growing classification head (`expand_head(n_new_classes)` appends new
output columns while leaving existing ones untouched).

| Model | Trunk | Role |
|---|---|---|
| **CubeSat MLP** | `Linear(F,32) -> ReLU -> Linear(32,16) -> ReLU` | **Primary backbone** -- directly processes the tabular telemetry feature vector. |
| **CubeSat GRU** | Single-layer GRU, hidden size 10 | **Architectural stress test** -- feature vector reshaped into a pseudo-sequence (e.g. CuCD-ID's 22 features -> 11 timesteps x 2 features) and processed with full manual BPTT. Not a genuine temporal model. |
| **CubeSat 1-D CNN** | `Conv1D(1->6,k=3) -> ReLU -> Conv1D(6->10,k=3) -> ReLU -> GlobalAvgPool` | **Architectural stress test** -- feature vector treated as a length-F, 1-channel signal; conv implemented via an explicit (correct, gradient-checked) im2col-style loop. Not genuine sequence modelling. |

A feature-order permutation study (`experiments/run_feature_order.py`)
supports this framing only as an exploratory check: the MLP has **no imposed
feature-order inductive bias**, but retraining under a permutation is not
mathematically invariant. The study uses one data split and two training
seeds (96 runs total), so it does not estimate a stable permutation effect.
The CNN and GRU move more materially because convolution/recurrence depend on
the imposed ordering. Read the generated feature-order table for the actual
spread rather than assuming permutation invariance.

All configurations fit within the **assumed 60 KB TinyML persistent-state
budget** imposed as a conservative study constraint (not a universal
CubeSat hardware limit) -- see the generated memory ledger for the current
largest configuration. Peak update RAM (FP32 weights, gradients, Adam
states, **batch-32** activations, replay/teacher, and retained
cumulative calibration state for EWC/full/BiC) is reported separately and is
substantially larger. The cumulative calibration pool is update-time state,
not deployed inference state; keeping it while fitting EWC/BiC/Fisher adds
roughly 70.3 KB for CuCD-ID (750 calibration rows x 22 features x 4 bytes plus labels, as built), so persistent and
update footprints are never conflated.

---

## 4. CMMU -- Continual Memory Management Unit (`src/cmmu.py`)

Two independent, combinable sub-mechanisms:

* **A) Experience Replay Buffer** -- fixed-capacity reservoir-sampling
  exemplar buffer. A fraction of every training minibatch after task 0 is
  drawn from the buffer, interleaving old-class examples with new-class
  examples. Deployed footprint is computed as int8-quantised bytes.
* **B) Online EWC** -- "Online" Elastic Weight Consolidation (Schwarz et
  al., 2018): a *single* running diagonal Fisher-information estimate and
  a *single* anchor-parameter snapshot are kept (O(1) memory in the number
  of tasks learned, unlike vanilla per-task EWC, which is essential under
  a fixed onboard memory budget). The Fisher is estimated from a
  class-balanced mix of the current task's training data and the cumulative
  calibration pool of all tasks seen so far (not the current task alone),
  accumulated as `F_new = gamma * F_old + F_current` with per-architecture
  EWC-only tuned weighting (frozen configs: MLP `gamma = 0.5`, GRU
  `gamma = 1.0`, CNN `gamma = 0.5`). Here `gamma` is treated as a
  validation-tuned weighting parameter rather than a conventional decay
  factor; the combined `full` configuration is tuned separately.
  Under the
  class-incremental protocol, the shared head is just as susceptible to
  forgetting as the trunk (the logit-competition effect described above),
  so **EWC here protects the whole model, head included** -- unlike a
  task-incremental design where disjoint per-task heads would need no such
  protection. Because the head grows in width over time, the Fisher/anchor
  arrays are padded to the new shape as needed; newly-added output columns
  start with zero Fisher (unconstrained) until the task that introduced
  them is itself consolidated.

Nine configurations are evaluated: `finetune`, `replay` (A only), `ewc`
(B only), `full` (A+B combined), `lwf` (distillation only, no memory),
`bic-lite` (herding + balanced replay + distillation + post-hoc bias
correction), `icarl-lite` (embedding-space herding + balanced replay +
distillation, softmax classifier), `icarl_ncm` (the same frozen iCaRL-lite
training runs evaluated with a cosine Nearest-Class-Mean head fitted only on
stored exemplars), `joint` (legacy cumulative warm-start, non-deployable),
and `joint_scratch` (from-scratch retraining on accumulated data, an offline
joint-training reference -- not a strict upper bound, since warm-start joint
can exceed it). The `-lite` suffix marks resource-conscious variants,
not exact reproductions of the original papers.

**Validation/calibration accounting (Option A).** The validation pool is
split into two **disjoint** halves per task/class: calibration (BiC fitting
and the EWC Fisher mix) and selection (hyperparameter scoring only). Because
the onboard update is defined to reproduce these results, calibration samples
are **counted inside the reported uplink budget**: payload figures are
adaptation data + calibration data (`uplink_KB_total_with_calib` in
`fewshot.csv`, §4 table totals). The selection half is an experimental
scoring artifact and is excluded from the onboard budget. The alternative --
a calibration-free onboard update -- is out of scope.

---

## 5. Metrics

Standard continual-learning metrics computed from the full per-task
accuracy matrix `R[i,j]` = accuracy on task `j`'s classes (decided among
*all* classes known at that point, via the single shared head) right after
finishing training on task `i` (`src/metrics.py`):

* **ACC** -- mean accuracy over all tasks after learning the final task.
* **Macro-F1 (primary)** -- class-balanced mean F1 over all classes seen so
  far, computed from the final head with BiC correction applied when fitted.
* **Balanced Accuracy** -- mean per-class recall.
* **BWT** -- backward transfer (Lopez-Paz & Ranzato, 2017); negative =
  forgetting.
* **FM** -- forgetting measure (Chaudhry et al., 2018): mean drop from each
  task's peak accuracy to its final accuracy.

Plus persistent deployment footprint (int8 model + int8 replay + fp16 EWC
state) reported **separately** from estimated peak update RAM (FP32
weights, gradients, Adam states, activations, batches, replay minibatch,
teacher where applicable, temporary buffers), per-sample and per-update MAC
estimates (the shared head's compute cost grows as more classes are
learned), and a **per-class precision / recall / F1 / confusion-matrix**
breakdown (`confusion_matrix`, `per_class_prf1` in `src/metrics.py`)
computed by `experiments/run_per_class_analysis.py`. The task-level metrics
above only *imply* that specific classes get forgotten; the per-class
breakdown makes it directly visible (e.g. an old class's F1 collapsing to
0.000) and is the more "visceral" evidence a CL paper reviewer typically
expects to see alongside the aggregate numbers.

Headline method comparisons use **exact paired permutation tests with Holm
correction**, reported with mean difference, 95% CI, and Cohen's d
(`experiments/run_stats.py`) -- never p-values alone. The pre-specified
family is **216 all-pairwise comparisons per metric** (all unordered method
pairs within each dataset x architecture), with Holm applied across the full
family. Current confirmatory results: **162/216 ACC** and **164/216
Macro-F1** comparisons significant. Matched-replay and EWC-lambda controls
are reported separately (`results/stats_baseline_controls.csv`; 7/30 survive
Holm). These tests are exploratory, not winner-selected.

---

## 6. Experiments run

1. **`experiments/run_hparam_sweep.py`** -- a 160-point grid search
   (learning rate x EWC lambda x replay-buffer capacity x replay fraction)
   over the **STIN** dataset using the MLP backbone as a fast proxy and
   `method="full"`, scored on the **selection half** of validation
   (`ACC_select - max(FM_select, 0)`, `val_frac=0.20`, disjoint from the
   calibration half). The winner is a coarse prior only
   (`results/best_hparams.json`).
2. **`experiments/run_validation_tune.py`** -- per-method x per-architecture
   validation tuning on **CuCD-ID** (9 methods x 3 architectures x 3
   seeds), scored only on the selection half. Winners are frozen to
   `results/best_hparams_per_method_arch.json` and used, unchanged, by
   every subsequent experiment -- test data is never used for selection.
3. **`experiments/run_main.py`** -- the full matrix: 2 datasets x 3
   architectures x 9 methods (`finetune`/`replay`/`ewc`/`full`/`lwf`/
   `bic`/`icarl`/`joint`/`joint_scratch`) x 15 seeds = **810 runs**, each
   producing a full accuracy matrix, CL + Macro-F1/Balanced-Accuracy
   metrics, detection metrics (CuCD-ID), training time, persistent vs peak
   memory accounting and MACs estimates.
   * **NCM evaluation** (`run_icarl_ncm.py`) -- reruns the frozen iCaRL
     configurations with embedding-space herding and a post-hoc cosine NCM
     classifier fitted only on the stored exemplars; 90 runs, paired against
     every softmax-head baseline on identical seeds. This changes herding and
     classifier together, so it is not a classifier-only comparison. Outcome:
     NCM is not a free win; BiC remains the best deployable softmax head.
   * **NCM sanity/control** (`check_ncm_sanity.py`) -- 108 matched runs spanning
     input-herding softmax, embedding-herding softmax, NCM fitted on the buffer,
     NCM fitted on all training data, and a joint all-data model. Outcome:
     NCM-all approximately NCM-buffer (max gap 0.0065 ACC), so buffer capacity
     is not the limitation; the NCM gap is classifier/feature-space and is
     negative/mixed, not a superiority claim.
   * **XAI study** (`run_xai.py`) -- from-scratch KernelSHAP and exact-input-
     gradient IG on identical stratified test rows, comparing BiC,
     iCaRL-lite+NCM and offline Scratch. Main metrics are Top5Ratio,
     deletion AUC and insertion AUC on correctly classified rows. The background
     is a real training medoid (the zero vector is degenerate for cosine/NCM),
     with a class-balanced mean-of-class-medoids background as a sensitivity
     check. IG uses 512 steps and KernelSHAP uses 256 sampled coalitions; its
     efficiency constraint makes completeness exact by construction, so
     completeness is a sanity check rather than accuracy evidence. Agreement
     and faithfulness are reported together; agreement is not proof of
     faithfulness.
4. **Phase-2 evidence** -- purged temporal evaluation
   (`run_purged_temporal.py`: IID vs forward-temporal with a 0.02 rad
   TimeRadians purge), read-only temporal leakage/validity audit
   (`run_temporal_leakage_audit.py`: nearest-train distances, class
   occupancy, natural blocks, early/late drift; no model training),
   few-shot uplink sweep (`run_fewshot.py`: K =
   5..300 under fixed and scaled calibration protocols), exploratory feature-order permutation
   (`run_feature_order.py`), disentangled task-order study
   (`run_task_order.py`: two-class-T0-first primary, single-class-first
   degenerate runs reported separately), ablations
   (`run_ablation.py`: buffer capacity, EWC lambda, legacy task order),
   per-class analysis (`run_per_class_analysis.py`: frozen configs, BiC
   correction applied), exact permutation statistics on **ACC and
   Macro-F1** (`run_stats.py`), robustness + quantization + memory ledger
   (`run_robustness.py`: simulated INT8 round-trip `int8_sim`), and a
   feature-provenance leakage ablation (`run_leakage_ablation.py`: 6 unique
   variants x 3 methods x 15 seeds = 270 runs, including drop-one,
   all-provenance-drop, and provenance-only contrasts).
5. **`experiments/check_consistency.py`** -- verifies main vs few-shot
   (K=150, intersecting seeds) hyperparameter and seed consistency.
6. **`analysis/plots.py`** -- generates every figure in `figures/`
   (35 PNGs, Macro-F1 primary, NCM vs softmax baselines plus the NCM control,
   main/correct-only XAI metrics, feature-group attribution, SHAP/IG agreement
   and faithfulness, five-protocol temporal bracket, temporal
   leakage/validity audit, leakage ablation).
7. **`experiments/create_manifest.py`** -- writes
   `results/final_experiment_manifest.json` (machine-readable provenance).
8. **`experiments/generate_summary_report.py`** -- writes
   `results/RESULTS_SUMMARY.md`, `results/XAI_SUPPLEMENT.md`, and
   `v6_6_project_report_final.md`, fully
   auto-generated reports pulling every number directly out of the
   CSV/JSON result files.

### Run order and expected outputs (for running on your own machine)

```bash
pip install -r requirements.txt
python3 run_all.py
```

or step by step (`python3 run_all.py --no-phase2` skips Phase-2):

```bash
python3 tests/test_gradients.py                    # sanity check
python3 tests/test_ncm.py                           # NCM unit test
python3 tests/test_xai.py                           # SHAP/IG unit test
python3 tests/test_feature_groups.py                # exact 22/30-feature partitions
python3 experiments/run_hparam_sweep.py             # STIN coarse prior (validation-scored)
python3 experiments/run_validation_tune.py          # per-method/arch tuning (validation, frozen)
python3 experiments/run_main.py                     # 810-run matrix (test evaluation only)
python3 experiments/run_scad.py --archs MLP --tag mlp_primary  # external SCAD 135-run matrix
python3 experiments/run_icarl_ncm.py                # 90-run NCM evaluation
python3 experiments/check_ncm_sanity.py             # 108-run NCM control
python3 experiments/run_xai.py                      # 36-model SHAP vs IG + faithfulness (2 backgrounds)
python3 experiments/run_purged_temporal.py          # IID vs purged-temporal
python3 experiments/run_temporal_leakage_audit.py   # read-only temporal validity audit
python3 experiments/run_fewshot.py                  # uplink sweep
python3 experiments/run_feature_order.py            # exploratory feature-order permutation
python3 experiments/run_task_order.py               # disentangled task order
python3 experiments/run_ablation.py                 # buffer / lambda / order ablations
python3 experiments/run_per_class_analysis.py       # per-class F1 + confusion matrices
python3 experiments/run_stats.py                    # ACC + Macro-F1 permutation stats + Holm
python3 experiments/run_robustness.py               # robustness + simulated INT8
python3 experiments/run_leakage_ablation.py         # 270-run retrained feature-group ablation
python3 experiments/check_consistency.py            # main-vs-fewshot verification
python3 analysis/plots.py                           # figures
python3 experiments/create_manifest.py              # reproducibility manifest
python3 experiments/generate_summary_report.py      # report
```

`python3 run_all.py` runs all of the above **in this exact order** and
stops immediately if any step fails. The order matters: everything after
validation tuning reads `results/best_hparams_per_method_arch.json`; the
plotting, manifest, and report steps read the CSV/JSON files written by
every step before them.

Every step writes its outputs directly to `results/` (CSV/JSON) and, for
the plotting step, `figures/` (PNG) -- both directories already exist in
this repository (with the outputs from our own run included) and will
simply be overwritten in place; nothing needs to be created by hand. After
a full run you should have:

| File | Written by |
|---|---|
| `results/best_hparams.json`, `results/hparam_sweep_stin.csv` | sweep |
| `results/best_hparams_per_method_arch.json`, `results/validation_tuning_results.csv` | validation tuning |
| `results/main_results_runs.csv`, `results/main_results_summary.csv`, `results/accuracy_matrices.json` | main matrix |
| `results/temporal_purged_compare.csv` | cold-start + fair temporal |
| `results/buac_summary.csv`, `results/buac_{mlp,gru,cnn1d}_*.csv` | BUAC admission control |
| `results/temporal_leakage_audit*.csv`, `results/temporal_block_coverage.csv`, `results/temporal_class_occupancy.csv`, `results/temporal_drift_*.csv`, `results/temporal_leakage_audit.json` | temporal leakage/validity audit |
| `results/fewshot.csv` | few-shot uplink sweep |
| `results/feature_order_stability.csv` | exploratory feature-order permutation |
| `results/task_order_disentangled.csv` | task-order study |
| `results/ablation_buffer_size.csv`, `results/ablation_ewc_lambda.csv`, `results/ablation_task_order.csv` | ablations |
| `results/per_class_f1_runs.csv`, `results/per_class_f1_summary.csv`, `results/confusion_matrices.json` | per-class analysis |
| `results/stats_permutation.csv`, `results/stats_permutation_macrof1.csv` | permutation stats (ACC + Macro-F1) |
| `results/robustness.csv`, `results/memory_ledger.csv`, `results/activation_memory_ledger.csv` | robustness |
| `results/icarl_ncm_runs.csv`, `results/icarl_ncm_summary.csv`, `results/icarl_ncm_vs_baselines.csv`, `results/accuracy_matrices_ncm.json` | NCM evaluation |
| `results/ncm_sanity_runs.csv`, `results/ncm_sanity_summary.csv`, `results/ncm_sanity_vs_main.csv`, `results/ncm_sanity_vs_main_runs.csv`, `results/ncm_sanity_per_class.csv`, `results/accuracy_matrices_ncm_sanity.json` | NCM control |
| `results/xai_metrics.csv`, `results/xai_summary.csv`, `results/xai_metrics_by_correctness.csv`, `results/xai_summary_by_correctness.csv`, `results/xai_group_summary.csv`, `results/xai_feature_summary.csv`, `results/xai_baseline_sensitivity.csv` | XAI |
| `results/leakage_ablation.csv`, `results/leakage_ablation_summary.csv`, `results/leakage_group_effects.csv` | leakage/group ablation |
| `results/final_experiment_manifest.json` | manifest |
| 35 PNG files under `figures/` | plots |
| `results/baseline_controls.csv`, `results/stats_baseline_controls.csv` | equal-buffer / EWC-lambda controls |
| `results/RESULTS_SUMMARY.md`, `results/XAI_SUPPLEMENT.md`, `v6_6_project_report_final.md` | report |

Nothing needs network access at run time (the two datasets are already
included under `data/`); everything is deterministic given the fixed seeds
in `experiments/config.py` (main matrix: seeds 0-14; few-shot: 0-4;
validation tuning: 0-2), so re-running the whole pipeline should
reproduce every number in `results/RESULTS_SUMMARY.md` and every figure
exactly.

(Full pipeline: about 4 hours on 4 CPU cores with `CL_JOBS=4`; the
confirmatory run measured 3h59m compute / 4h07m wall, dominated by validation
tuning, the 810-run matrix, and ablations. No GPU required.)

---

## 7. Headline results

Every result value is written by the experiment pipeline; none is edited by
hand. See `results/RESULTS_SUMMARY.md` and the auto-generated
`v6_6_project_report_final.md` for the current tables, confidence intervals,
paired tests, ablations, and figures. `results/main_results_summary.csv` is
the underlying source for the headline matrix.

The supported qualitative findings are:

- Class-incremental naive fine-tuning collapses on the sequential benchmark;
  EWC/LwF alone do not rescue it.
- Replay and replay-based methods substantially reduce forgetting, but their
  ranking depends on architecture and tuning; report the per-backbone matrix,
  not a single pooled winner.
- The combined EWC+replay+BiC (`full`) configuration is **not** a uniform
  improvement over replay. On CuCD-ID it is below replay for all three
  architectures; treat it as a baseline, not the proposed winner.
- BiC-lite and iCaRL-lite are competitive with replay on several backbones,
  while the offline joint references define the ceiling.
- The STIN cross-domain stress test has a much lower ceiling than CuCD-ID.
- Architecture matters: MLP is the primary backbone, with GRU/CNN treated as
  stress tests rather than assumed equivalents.
- The IID split is a benchmark estimand. Cold-start and fair random-past
  temporal evaluation and feature-provenance ablations are separate
  reliance/robustness estimands and are not deployment accuracy claims.
- Update admission is only non-trivial under the fair temporal protocol.
  Cold-start rejects all updates; learned BUAC is near the 5% unsafe target on
  MLP but exceeds it on GRU/CNN, and the fixed rule can exceed it on CNN.
  BUAC is stage-wise/offline, not a closed-loop deployment claim. The current
  manuscript keeps BUAC as a secondary diagnostic and adds a scenario-disjoint
  SCAD external stress test. SCAD is a negative/boundary result: the 150-shot
  joint ceiling is 0.563 and GRU/CNN are near chance, so it is not presented as
  cross-dataset confirmation.
- Persistent model memory fits the assumed budget, but estimated peak update
  RAM is materially larger and is reported separately.

Task-order variation, per-class collapse, robustness, leakage, power, and
significance details are all generated from the corresponding CSV/JSON files;
consult those files rather than copying numbers into prose.

---

## 8. Repository structure

```
.
|-- data/
|   |-- CuCD-ID/noised_dataset.csv, label_schema.json
|   `-- STIN/SAT20.csv, TER20.csv
|-- src/
|   |-- data_utils.py     # loading, class-incremental split construction (IID + temporal + purged-temporal)
|   |-- models.py         # CubeSatMLP / CubeSatGRU / CubeSat1DCNN (shared growing head, manual fwd/bwd)
|   |-- optim.py          # Adam optimizer (shape-growth-aware momentum padding)
|   |-- losses.py         # softmax cross-entropy
|   |-- cmmu.py           # ReplayBuffer + OnlineEWC (shape-growth-aware, bounded herding)
|   |-- train.py          # class-incremental training loop (9 methods, dynamic label offsetting, V matrix, compute counters)
|   |-- inference.py      # shared logits dispatch (softmax / BiC / NCM)
|   |-- ncm.py            # cosine Nearest-Class-Mean head fitted on stored exemplars
|   |-- feature_groups.py # disjoint CuCD-ID/STIN provenance groups
|   |-- xai.py            # KernelSHAP, Integrated Gradients, completeness, group metrics
|   |-- metrics.py        # ACC / BWT / FM / Macro-F1 / BalancedAccuracy, persistent vs peak memory, MACs, per-class F1
|   `-- utils.py          # seeding
|-- experiments/
|   |-- config.py                      # 9 methods, 15 seeds, per-arch model kwargs
|   |-- run_hparam_sweep.py            # STIN coarse prior (validation-scored)
|   |-- run_validation_tune.py         # per-method/arch validation tuning (frozen configs)
|   |-- run_main.py                    # 810-run matrix (test evaluation only)
|   |-- run_icarl_ncm.py               # iCaRL-lite + NCM evaluation
|   |-- check_ncm_sanity.py            # 108-run NCM buffer/herding/classifier control
|   |-- run_xai.py                     # KernelSHAP vs IG + faithfulness + groups
|   |-- run_purged_temporal.py         # IID vs purged vs no-leaky-features temporal study
|   |-- run_temporal_leakage_audit.py  # read-only temporal leakage/validity audit
|   |-- run_fewshot.py                 # uplink sweep (frozen configs, K varies)
|   |-- run_feature_order.py           # exploratory feature-permutation check
|   |-- run_task_order.py              # disentangled task-order study
|   |-- run_ablation.py                # buffer / lambda / order ablations
|   |-- run_per_class_analysis.py      # per-class F1 + confusion matrices
|   |-- run_stats.py                   # exact permutation + Holm + effect sizes
|   |-- run_robustness.py              # robustness + simulated INT8
|   |-- run_leakage_ablation.py        # retrained feature-group leakage ablation
|   |-- check_consistency.py           # main-vs-fewshot verification
|   |-- create_manifest.py             # reproducibility manifest
|   `-- generate_summary_report.py     # auto-generated report
|-- analysis/plots.py
|-- tests/test_gradients.py, test_ncm.py, test_xai.py, test_feature_groups.py, test_buac.py, test_temporal_sampling.py
|-- results/               # CSV/JSON outputs + manifest + reports (generated)
|-- figures/                # PNG figures (generated, 35 files)
|-- run_all.py
|-- requirements.txt
`-- README.md
```

---

## 9. Limitations and threats to validity

* **No hardware-in-the-loop measurement.** Latency and memory are
  host-estimated (per-sample/per-update MAC estimates, analytic byte
  counts); no measurement was taken on actual Cortex-M-class silicon, no
  INT8 kernels were executed (quantization is a simulated round-trip
  proxy), and wall-clock NumPy timings are not reported as latency
  figures. All such quantities carry `SIM_` prefixes or "estimated"
  qualifiers where applicable.
* **STIN has no benign class** in the provided extract, so it cannot be
  used to evaluate normal-vs-attack detection directly, only inter-attack
  class-incremental discrimination; it is used as a cross-domain stress
  test only (the coarse STIN sweep is a prior, not the final tuning). Its
  joint/offline accuracy ceiling is notably lower than CuCD-ID's,
  indicating the 9-class discrimination problem is
  intrinsically harder at this feature set and scale -- not merely a
  continual-learning artefact.
* **GRU/CNN are architectural stress tests, not temporal models.** Both
  operate on reshaped tabular feature vectors, not genuine sequences. The
  exploratory feature-order study moves more for GRU/CNN, but it does not
  establish that the MLP is permutation-invariant (one split, two training
  seeds). GRU is also one of the two architectures (with CNN1D)
  most sensitive to EWC lambda -- confirmed on **both** CuCD-ID and STIN
  independently (Section 7 / `results/RESULTS_SUMMARY.md` Section 5.2) --
  reported as a finding, not adjusted away.
* **CNN1D underperforms on CuCD-ID** relative to MLP/GRU even after
  per-method validation tuning (see the generated main-results matrix), and degrades
  under feature permutation; the residual gap (optimization + structural
  sensitivity) is reported as an open question rather than resolved.
* **Small task sample counts** (deliberately, to respect the KB-scale
  uplink constraint) mean per-seed variance can be non-trivial for some
  configurations (see the std columns in `results/main_results_runs.csv`
  and the buffer-size/task-order ablations); all headline numbers are
  reported as means over 15 seeds with 95% CIs and effect sizes for key
  comparisons.
* **Empirical Fisher from the cumulative calibration pool.** Online EWC
  here uses the empirical Fisher (squared gradient of the true-label
  log-likelihood, estimated from a class-balanced mix of current-task
  training data and the cumulative calibration pool, with the running
  estimate retained) rather than the exact expected Fisher over old-task
  data; EWC-only per-architecture tuned decay applies (MLP 0.5 / GRU 1.0 /
  CNN 0.5).
* **Calibration and selection are disjoint.** BiC/Fisher use the calibration
  half; hyperparameter scoring uses the selection half. Validation is
  therefore not untouched by model fitting, but test data is. Calibration
  samples are counted inside the reported uplink budget; selection samples
  are experimental-only and excluded (see Section 2).
* **Task-order effect is non-trivial and unexplained in mechanism.** The
  disentangled study reports budget-mode means and per-order spread in the
  generated task-order table; individual runs vary more still. On primary
  orders, `fixed_position` and `moves_with_task` are identical by construction
  because every primary order starts with T0, so they are not independent
  evidence. Whether the remaining variation is class difficulty or order
  interaction is flagged as future work rather than resolved here.
* **Temporal evaluation separates cold-start support from drift.** The
  purged protocol reduces window overlap (0.02 rad TimeRadians purge) but the
  data also exhibits mission-time drift and a T0-frozen scaler. The read-only
  audit (`run_temporal_leakage_audit.py`) shows dense nearest-time adjacency
  in the random split, but only five natural time blocks and class-incomplete
  coverage; it is a validity boundary, not a leakage-free correction. Across
  3 architectures x 15 seeds x 7 deployable methods, mean ACC is **0.613
  (IID)**, **0.201 (cold-start temporal)**, **0.502 (fair random-past
  temporal)**, and **0.430 (fair + no time/ID/window features)**. The
  cold-start arm trains only on the earliest head of each class block; even
  `joint_scratch` falls to **0.128 (MLP)** cold-start versus **0.902** under
  the fair random-past protocol. A matched 70/10/20 control (same 20% test
  block and random validation) leaves MLP `joint_scratch` at **0.106**
  cold-start versus **0.870** fair; BiC is **0.171** versus **0.894**, so
  test/validation block choice is not the driver. The cold-start gap is therefore
  mostly a support artifact, not pure drift or forgetting. The fair arm still
  leaves a real gap, and `finetune` remains near chance; IID and temporal
  numbers are separate regimes, and IID is not a deployment guarantee.
* **No true scenario-level validation is possible.** CuCD-ID has no
  scenario/run identifier in the released CSV. A grouped time-block proxy is
  a sensitivity estimand only.
* **Hyperparameters transfer across datasets and tuning/evaluation seeds
  overlap.** All final configs are tuned on CuCD-ID validation; STIN is a
  cross-dataset transfer test. Tuning seeds 0-2 overlap confirmatory seeds
  0-14. Selection is validation-only (no test leakage), but the seeds are not
  fully independent.
* **NCM is a negative/mixed result, not a classifier-isolation claim.** The
  original NCM run changes herding and classifier together. The matched control
  shows NCM-all approximately NCM-buffer, so buffer size is not the limitation;
  the remaining gap is classifier/feature-space. Do not claim NCM superiority.
* **XAI baseline and subset choices matter.** Attributions use a primary
  training-medoid background because zero is degenerate for cosine/NCM, with a
  class-balanced medoid-mean sensitivity check. IG uses 512 steps; KernelSHAP
  uses 256 sampled coalitions and its completeness is exact by construction.
  Main-text metrics are correct-only Top5Ratio/deletion/insertion, with all-row
  and error-row detail in the supplement. Under the primary background the top
  feature is not universally `TimeRadians` (CuCD-ID/GRU BiC and iCaRL+NCM rank
  `UniqueMessageIDsInWindow` first); under the sensitivity background GRU
  rankings can shift to `TimeRadians` or `MemoryAnonMB`. Attribution is not
  causal proof.
* **Feature-order evidence is exploratory.** The study uses one split and two
  training seeds; it does not establish permutation invariance.
