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

**Leakage caveat.** CuCD-ID includes a circular time field, CCSDS packet
identifiers, and 20-second sliding-window aggregates. The headline IID
random split can share overlapping windows between train and test, so those
numbers are benchmark estimates, not deployment guarantees. The
`leakage_ablation` experiment retrains models after removing each feature
group, and the temporal-purged protocol is reported separately; neither
turns the IID number into a leakage-free estimate.

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
without corrupting label alignment (see `experiments/run_ablation.py`).

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
confirms this framing: the MLP is stable under random feature-column
permutation, while the CNN and GRU change materially -- the
convolution/recurrence depend on the imposed feature ordering, whereas the
MLP does not. See the generated feature-order table for current values.

All configurations fit within the **assumed 60 KB TinyML persistent-state
budget** imposed as a conservative study constraint (not a universal
CubeSat hardware limit) -- see the generated memory ledger for the current
largest configuration. Peak update RAM (FP32 weights, gradients, Adam
states, activations, batches, teacher) is reported separately and is
substantially larger -- persistent footprint and update RAM are different
quantities and are never conflated in this report.

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
  a fixed onboard memory budget). The Fisher is estimated from the
  **current task's training/validation samples** (online new-task Fisher
  variant -- previous Fisher information is retained through the running
  consolidation, not re-estimated from old data), accumulated as
  `F_new = gamma * F_old + F_current` with per-architecture tuned weighting
  (frozen configs: MLP `gamma = 0.5`, GRU `gamma = 2.0`, CNN `gamma = 0.5`).
  Here `gamma` is treated as a validation-tuned weighting parameter rather
  than a conventional decay factor: GRU's `gamma = 2.0` amplifies the
  retained Fisher instead of decaying it, and was kept because validation
  selected it. Restricting the grid to `gamma <= 1` would be a stricter EWC
  formulation but would require rerunning the EWC-related experiments.
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
correction), `icarl-lite` (herding + balanced replay + distillation,
softmax classifier -- embedding-space herding and NCM evaluation omitted),
`joint` (legacy cumulative warm-start, non-deployable), and
`joint_scratch` (from-scratch retraining on accumulated data, an offline
joint-training reference -- not a strict upper bound, since warm-start
joint can exceed it). The `-lite` suffix marks resource-conscious variants,
not exact reproductions of the original papers.

**Validation/calibration accounting (Option A).** BiC bias correction is fit on
held-out validation data and the EWC Fisher mixes in validation samples.
Because the onboard update is defined to reproduce these results, those
calibration samples are **counted inside the reported uplink budget**:
every payload figure is adaptation data + calibration data
(`uplink_KB_total_with_calib` in `fewshot.csv`, §4 table totals). The
alternative -- a calibration-free onboard update -- is out of scope.

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
(`experiments/run_stats.py`) -- never p-values alone.

---

## 6. Experiments run

1. **`experiments/run_hparam_sweep.py`** -- a 160-point grid search
   (learning rate x EWC lambda x replay-buffer capacity x replay fraction)
   over the **STIN** dataset using the MLP backbone as a fast proxy and
   `method="full"`, scored on **validation** (`ACC_val - FM_val`,
   `val_frac=0.10`). The winner is a coarse prior only
   (`results/best_hparams.json`).
2. **`experiments/run_validation_tune.py`** -- per-method x per-architecture
   validation tuning on **CuCD-ID** (9 methods x 3 architectures x 3
   seeds), scored on validation. Winners are frozen to
   `results/best_hparams_per_method_arch.json` and used, unchanged, by
   every subsequent experiment -- test data is never used for selection.
3. **`experiments/run_main.py`** -- the full matrix: 2 datasets x 3
   architectures x 9 methods (`finetune`/`replay`/`ewc`/`full`/`lwf`/
   `bic`/`icarl`/`joint`/`joint_scratch`) x 10 seeds = **540 runs**, each
   producing a full accuracy matrix, CL + Macro-F1/Balanced-Accuracy
   metrics, detection metrics (CuCD-ID), training time, persistent vs peak
   memory accounting and MACs estimates.
4. **Phase-2 evidence** -- purged temporal evaluation
   (`run_purged_temporal.py`: IID vs forward-temporal with a 0.02 rad
   TimeRadians purge), few-shot uplink sweep (`run_fewshot.py`: K =
   5..300, frozen configs, only K varies), feature-order stability
   (`run_feature_order.py`), disentangled task-order study
   (`run_task_order.py`: `fixed_position` primary), ablations
   (`run_ablation.py`: buffer capacity, EWC lambda, legacy task order),
   per-class analysis (`run_per_class_analysis.py`: frozen configs, BiC
   correction applied), exact permutation statistics on **ACC and
   Macro-F1** (`run_stats.py`), robustness + quantization + memory ledger
   (`run_robustness.py`: simulated INT8 round-trip `int8_sim`), and a
   feature-provenance leakage ablation (`run_leakage_ablation.py`:
   retrain without time/identifier/window groups).
5. **`experiments/check_consistency.py`** -- verifies main vs few-shot
   (K=150, intersecting seeds) hyperparameter and seed consistency.
6. **`analysis/plots.py`** -- generates every figure in `figures/`
   (21 PNGs, Macro-F1 primary, dual-protocol temporal bracket, leakage ablation).
7. **`experiments/create_manifest.py`** -- writes
   `results/final_experiment_manifest.json` (machine-readable provenance).
8. **`experiments/generate_summary_report.py`** -- writes
   `results/RESULTS_SUMMARY.md` and `v5_4_project_report_final.md`, fully
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
python3 experiments/run_hparam_sweep.py             # STIN coarse prior (validation-scored)
python3 experiments/run_validation_tune.py          # per-method/arch tuning (validation, frozen)
python3 experiments/run_main.py                     # 540-run matrix (test evaluation only)
python3 experiments/run_purged_temporal.py          # IID vs purged-temporal
python3 experiments/run_fewshot.py                  # uplink sweep
python3 experiments/run_feature_order.py            # feature-order stability
python3 experiments/run_task_order.py               # disentangled task order
python3 experiments/run_ablation.py                 # buffer / lambda / order ablations
python3 experiments/run_per_class_analysis.py       # per-class F1 + confusion matrices
python3 experiments/run_stats.py                    # ACC + Macro-F1 permutation stats + Holm
python3 experiments/run_robustness.py               # robustness + simulated INT8
python3 experiments/run_leakage_ablation.py         # retrained feature-group leakage ablation
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
| `results/temporal_purged_compare.csv` | purged temporal |
| `results/fewshot.csv` | few-shot uplink sweep |
| `results/feature_order_stability.csv` | feature-order stability |
| `results/task_order_disentangled.csv` | task-order study |
| `results/ablation_buffer_size.csv`, `results/ablation_ewc_lambda.csv`, `results/ablation_task_order.csv` | ablations |
| `results/per_class_f1_runs.csv`, `results/per_class_f1_summary.csv`, `results/confusion_matrices.json` | per-class analysis |
| `results/stats_permutation.csv`, `results/stats_permutation_macrof1.csv` | permutation stats (ACC + Macro-F1) |
| `results/robustness.csv`, `results/memory_ledger.csv` | robustness |
| `results/leakage_ablation.csv`, `results/leakage_ablation_summary.csv` | leakage ablation |
| `results/final_experiment_manifest.json` | manifest |
| 21 PNG files under `figures/` | plots |
| `results/RESULTS_SUMMARY.md`, `v5_4_project_report_final.md` | report |

Nothing needs network access at run time (the two datasets are already
included under `data/`); everything is deterministic given the fixed seeds
in `experiments/config.py` (main matrix: seeds 0-9; few-shot: 0-4;
validation tuning: 0-2), so re-running the whole pipeline should
reproduce every number in `results/RESULTS_SUMMARY.md` and every figure
exactly.

(Full pipeline: ~10-14 hours on a single CPU core, dominated by validation
tuning and the 540-run matrix; no GPU required.)

---

## 7. Headline results

Every result value is written by the experiment pipeline; none is edited by
hand. See `results/RESULTS_SUMMARY.md` and the auto-generated
`v5_4_project_report_final.md` for the current tables, confidence intervals,
paired tests, ablations, and figures. `results/main_results_summary.csv` is
the underlying source for the headline matrix.

The stable qualitative findings are:

- Class-incremental naive fine-tuning collapses on the sequential benchmark;
  EWC/LwF alone do not rescue it.
- Replay and replay-based methods substantially reduce forgetting, but their
  ranking depends on architecture and tuning; report the per-backbone matrix,
  not a single pooled winner.
- BiC-lite and iCaRL-lite are competitive with replay on several backbones,
  while the offline joint references define the ceiling.
- The STIN cross-domain stress test has a much lower ceiling than CuCD-ID.
- Architecture matters: MLP is the primary backbone, with GRU/CNN treated as
  stress tests rather than assumed equivalents.
- The IID split is a benchmark estimand. Temporal-purged evaluation and
  feature-provenance ablations are separate reliance/robustness estimands and
  are not deployment accuracy claims.
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
|   |-- metrics.py        # ACC / BWT / FM / Macro-F1 / BalancedAccuracy, persistent vs peak memory, MACs, per-class F1
|   `-- utils.py          # seeding
|-- experiments/
|   |-- config.py                      # 9 methods, 10 seeds, per-arch model kwargs
|   |-- run_hparam_sweep.py            # STIN coarse prior (validation-scored)
|   |-- run_validation_tune.py         # per-method/arch validation tuning (frozen configs)
|   |-- run_main.py                    # 540-run matrix (test evaluation only)
|   |-- run_purged_temporal.py         # IID vs purged-temporal dual protocol
|   |-- run_fewshot.py                 # uplink sweep (frozen configs, K varies)
|   |-- run_feature_order.py           # feature-permutation stability
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
|-- tests/test_gradients.py
|-- results/               # CSV/JSON outputs + manifest + RESULTS_SUMMARY.md (generated)
|-- figures/                # PNG figures (generated, 20 files)
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
  operate on reshaped tabular feature vectors, not genuine sequences; the
  feature-order study shows they depend on the imposed ordering while the
  MLP does not. GRU is also one of the two architectures (with CNN1D)
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
  reported as means over 10 seeds with 95% CIs and effect sizes for key
  comparisons.
* **Empirical, new-task Fisher.** Online EWC here uses the empirical
  Fisher (squared gradient of the true-label log-likelihood, estimated
  from current-task train/val samples with the running estimate retained)
  rather than the exact expected Fisher over old-task data; per-architecture
  tuned decay applies (MLP 0.5 / GRU 2.0 / CNN 0.5).
* **Validation data does double duty.** BiC calibration and the Fisher
  mix consume validation samples that also feed the `V` matrix, so
  validation metrics are not untouched; the test set is. Those calibration
  samples are counted inside the reported uplink budget (see Section 2).
* **Task-order effect is non-trivial and unexplained in mechanism.** The
  disentangled study reports budget-mode means and per-order spread in the
  generated task-order table; individual runs vary more still. Whether the
  remainder is class difficulty or order interaction is flagged as future
  work rather than resolved here.
* **Temporal evaluation is a stress estimand, not ground truth.** The
  purged protocol reduces window overlap (0.02 rad TimeRadians purge) but
  the data also exhibits mission-time drift and a T0-frozen scaler; IID
  and temporal numbers are reported as separate regimes.
