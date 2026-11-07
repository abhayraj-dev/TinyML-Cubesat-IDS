# DATALOG — data & experiment provenance (cubesat-cl-ids)

> **Historical log.** Entries below record the run that produced them; numeric
> claims in older entries (especially the 2026-09-21 K~50 knee and temporal
> figures) are superseded by the current `results/` CSVs and
> `results/RESULTS_SUMMARY.md`. Do not cite an old entry without checking the
> current manifest, which derives run counts/grids from the executable code.

Purpose: log **what data/split/config produced which results**, so the journal reproducibility checklist can be answered. Append-only: newest entries on top.

## 2026-11-07 — v6.7 first-differencing provenance controls
- **Protocol:** within each class's time-ordered recording, every row is differenced against its predecessor and the first row is dropped. Four feature sets: `raw` (message length + 8 memory gauges), `raw_window` (raw + 9 window aggregates), `resource_state` (8 gauges + 9 window aggregates; time, length and identifiers removed), `raw_no_prov` (differenced raw only; time, identifiers and window aggregates removed).
- **Matrix:** 4 feature sets × 2 protocols (`temporal_purged_fair`, `iid`) × 6 strategies × 15 seeds = **1,080 runs**.
- **Fair-temporal (mean ACC):** raw replay/BiC/iCaRL 0.822/0.857/0.833, joint-scratch 0.916; 17-feature 0.547/0.551/0.474, joint 0.692; no-provenance 0.504/0.547/0.450, joint 0.643; memory-free 0.250 throughout.
- **Tests:** every exemplar/joint drop vs the original fair run is significant after Holm (`p<0.001`; Replay+EWC `p=0.000610`). IID ordering is preserved under both controls (BiC-lite, replay, iCaRL-lite).
- **Caveat:** first-differencing removes constant recording/session offsets, not slowly varying signatures; the controls adapt but do not replicate Singh et al.'s normal-only two-sided calibration. Use 0.45–0.55 as an empirical range, not a leakage-free estimate.
- **Outputs:** `experiments/first_difference/results/first_difference_{runs,summary,comparison,tests}.csv`, `first_difference_setup.json`, `first_difference_report.md`, `first_difference_takeaways.md`; report section 7.13 and `v6_7_project_report_final.md`.

## 2026-10-28 — SCAD scenario-disjoint external validation (MLP primary)
- **Loader/protocol:** `src/data_utils.py::load_scad` reads SCAD with 80 leakage-controlled telemetry features (time, GPS/UTC, orbit/position, scenario IDs, `AttackIntensity`, `TimeDelay`, `Variant*`, `Run_Batch*` and labels dropped). Features are centred on each scenario's first 300 attack-free rows; only post-baseline rows are retained. `build_scad_tasks` uses one global 70/10/20 split over **120 trajectory groups** (month + offset), so all intensities and the matching baseline run of a trajectory stay in the same split and test trajectories are never trained on. Sequence: **T0 normal+RWA, T1 RWB, T2 RWC**.
- **Budgets/split:** 84/12/24 trajectory groups; T0 train/val/test = **1,400 / 600 / 2,400**; T1 and T2 = **150 / 300 / 1,200**. Val is split 50/50 into calibration/selection by the existing `Task` code.
- **Primary MLP result (15 seeds, frozen hparams, 150-shot uplink):** mean ACC **joint_scratch 0.563±0.040, joint 0.545±0.035, iCaRL 0.506±0.032, BiC 0.498±0.042, replay 0.462±0.024, full 0.434±0.035, EWC 0.334, LwF 0.333, finetune 0.333**. Random = 0.25. BiC beats replay on ACC (12/15 seeds, one-sided sign p=0.0176) and has the best retention among rehearsal methods (BWT -0.198 vs replay -0.396); iCaRL beats replay 15/15 (p=3.1e-5). The joint reference (0.563) is well below the CuCD/STIN joint reference, confirming SCAD is a harder distribution.
- **Cross-architecture 150-shot (15 seeds):** GRU best ACC iCaRL **0.370**, replay 0.348, joint_scratch 0.355; CNN1D best replay **0.338**, BiC 0.291, joint_scratch 0.244. Under the 150-shot budget GRU/CNN are near or below chance on this task; do not present them as positive SCAD transfer.
- **Uplink-budget sensitivity (MLP, 15 seeds, 700-shot uplink):** joint_scratch **0.662±0.041**, joint 0.642±0.033, replay 0.585±0.037, iCaRL 0.563±0.023, BiC 0.539±0.047. The BiC advantage reverses at 700 shots (replay beats BiC on 13/15 seeds), so the 150-shot protocol is the relevant operating point for the paper's claim. A 1-seed GRU/CNN 700-shot check raises joint_scratch to GRU 0.456 / CNN 0.332, confirming the 150-shot collapse is budget-driven for GRU but not fully for CNN.
- **Outputs:** `results/scad_runs_mlp_primary.csv`, `results/scad_summary_mlp_primary.csv`, `results/scad_accuracy_matrices_mlp_primary.json`, `results/scad_splits_mlp_primary.json`; `results/scad_*_mlp_up700.*`; `results/scad_*_gru_cnn15_group.*`; runner `experiments/run_scad.py`; test `tests/test_scad.py`.
- **Caveats:** SCAD has no peer-reviewed DOI (CC BY 4.0 GitHub dataset); two `-50` scenarios are truncated (RWB 208 attack rows, RWC 39). Scenario-disjoint splitting is stricter than the paper's IID CuCD protocol, so SCAD numbers are not directly comparable to the main table. GRU/CNN negative transfer should be reported honestly or SCAD should be restricted to the primary MLP architecture with the budget sensitivity as supplementary evidence.

## 2026-10-28 — v6.6 local external-dataset intake and verification (no model runs)
- **SCAD verified locally:** `data/space-cyber-anomaly-detection-dataset-main/` contains all 1,920 scenarios and `all_metadata.csv` (1,920 rows). Class/scenario counts are 120 baseline, 600 RWA, 600 RWB and 600 RWC. Across all files: **72,240 baseline-only rows**, **180,000 RWA rows**, **179,908 RWB rows** and **179,739 RWC rows**. Attack files contain 302 pre-attack baseline rows followed by attack rows. Two `-50` intensity scenarios are truncated (`12_144.0_-50` RWB has 208 attack rows; `9_108.0_-50` RWC has 39); all others have 300. Use `data_file` as the scenario key, not `Variant_name`, which repeats across classes.
- **AegisSat verified locally:** `data/AegisSat-AD.csv` is 190,028,590 bytes, **137,965 rows x 176 columns**, with **28 run IDs**, **3,526 attack rows** and **134,439 normal rows**. Every attack row has both `attacks.cpuhightarget.duration` and `target`; `_id` is unique and `run_id` is complete. Runs 24-27 are normal-only; the other 24 runs contain attacks. This supports a grouped binary check, not class-incremental evaluation.
- **UNSW-IoTSAT is not present:** `data/UNSW-IoTSAT-main/` contains code, README and a 1-byte `Scripts` placeholder only; there are no CSV, JSON, ZIP or Parquet data files. The README points to a UNSW SharePoint folder that requires Microsoft login. The paper is published at `doi:10.1016/j.csa.2026.100133`, but the dataset cannot be used until the authors grant access or provide a mirror.
- **CUCDID_All verified:** raw `consolidated_dataset_raw.csv` is **25,000 x 31** with 5,000 rows/class; augmented `noised_dataset.csv` is **22,465 x 23** and is byte-identical to `data/CuCD-ID/noised_dataset.csv` (SHA-256 `656fc2f2...cfae3c68`). No new scenario/run identifiers are present.
- **External-validation decision:** use SCAD as the primary external **4-class class-incremental** benchmark (scenario-disjoint splits, leakage-controlled features); use AegisSat only as a secondary **binary physical-testbed** check (run-disjoint splits); treat UNSW-IoTSAT as contingent on data access. Do not claim AegisSat validates continual learning.
- **Repository policy:** raw external datasets are excluded from git via `.gitignore`; this entry records their local verification without redistributing them. No external-benchmark model runs have been executed yet.

## 2026-10-27 — Temporal protocol root cause, fair random-past rerun, and BUAC fair protocol
- **Root cause of the old temporal collapse:** the purged builder trained only on the earliest head of each class's past block (`train_pool[:n_per_class]`) while IID drew a random sample. The collapse is therefore mostly a **cold-start support artifact**, not pure drift or catastrophic forgetting: even `joint_scratch` failed cold-start. Code now exposes `train_sample_mode` and `val_mode`; `val_mode="random_past"` keeps test locked in the future, purges it, and draws train/validation from the purged past.
- **Fair protocol:** 70/10/20 train/val/test, 0.02 rad TimeRadians purge, random-past train and validation. This also avoids empty validation for narrow classes; the old contiguous block can still be empty and is retained only as the cold-start baseline.
- **Temporal rerun:** `results/temporal_purged_compare.csv`, **2,025 runs** = 3 architectures x 15 seeds x 9 methods x 5 protocols (IID, cold-start, fair, cold no-leaky, fair no-leaky). Mean deployable ACC (7 methods): **0.6131 IID**, **0.2011 cold-start**, **0.5018 fair**, **0.4295 fair no-leaky**. MLP `joint_scratch` **0.128 cold-start -> 0.902 fair**; MLP BiC **0.260 -> 0.838**. Fair still leaves a real gap (`finetune` near chance) and no-leaky drops further, so temporal drift and time/ID/window reliance remain separate limitations.
- **Diagnosis artifacts:** `results/temporal_sampling_diagnosis.csv` (3 seeds: MLP BiC earliest 0.241, random-past 70 0.894; joint_scratch 0.107 -> 0.870) and `results/temporal_sampling_probes.csv` (ridge on fair 70: 0.827 all features, 0.389 no-leaky). Matched 70/10/20 control (same 20% test block and random validation): MLP `joint_scratch` 0.106 cold-start vs 0.870 fair; BiC 0.171 vs 0.894, so test/validation block choice is not the main driver. Tests cover sampling, determinism, future-test separation, purge, and non-empty random-past validation.
- **BUAC:** added `temporal_purged_fair` to `experiments/run_buac.py`; **810 runs / 3,240 stage rows**. On fair held-out updates, learned BUAC safe/unsafe acceptance is MLP **0.949/0.078**, GRU **0.560/0.169**, CNN1D **0.636/0.132**. Cold-start remains reject-all, so the old fixed rule was behaviorally trivial there. IID-only gate transfer is worse (unsafe 0.078-0.215); the fixed rule exceeds 5% unsafe on CNN (0.176) because it lacks threshold calibration. MLP budget sensitivity: at 64 KB memory the peak-training RAM gate accepts nothing; 128 KB accepts 0.108 safe / 0 unsafe; 256 KB accepts 0.882 / 0.023.
- **Caveats:** BUAC is stage-wise/offline, not closed-loop; rejecting an update does not branch later training, and ground-assisted utility/coverage is not evaluated. `r_fwd_flag` is a protocol proxy, not a physical risk sensor. Do not cite the old “temporal-purged collapse” entry as pure drift evidence.
- **Outputs:** `results/temporal_purged_compare*.csv`, `results/temporal_sampling_diagnosis.csv`, `results/temporal_sampling_probes.csv`, `results/buac_summary.csv`, `results/buac_{mlp,gru,cnn1d}_*.csv`, updated `results/RESULTS_SUMMARY.md`, `v6_6_project_report_final.md` (supersedes the v6.5 report for update-integrity), `figures/temporal_bracket.png`, `results/final_experiment_manifest.json`.
- **Path A+ manuscript:** CubeSat-first draft in `cubesat_paper/main.tex` with `bib.tex` (42 refs), 3 figures and 7 tables; `figures/temporal_mlp_protocols.png` added for the fair/cold/no-leaky comparison. The draft compiles to a 25-page 12pt preprint via Tectonic. Author names, affiliations and CRediT text remain placeholders; no new training was run.

## 2026-10-26 — v6.5 leakage dedup + XAI metric/background corrections (completed; supersedes the 315-run entry)
- **Artifact identity:** package/report version **v6.5**; current report is
  `v6_5_project_report_final.md`. Historical v6.4/v6.3 entries remain below.
- **Leakage ablation rerun:** removed the `only_raw_telemetry` alias of
  `no_time_ids_window` from `experiments/run_leakage_ablation.py`; reran 6
  unique variants x 3 methods x 15 seeds = **270 runs** (261 s, 4 workers).
  `results/leakage_ablation.csv` has 270 rows. Headline group effects are
  unchanged: `no_time_ids_window` is the strongest BiC/replay drop (ACC 0.9153,
  -0.0771; 0.9229, -0.0666), while `only_time_ids_window` retains most accuracy
  (0.9843 BiC, -0.0080).
- **XAI metric correction:** `src/xai.py` now emits `baseline_prob`,
  `insertion_gain = inserted_prob - baseline_prob`, and
  `insertion_gap_vs_full = inserted_prob - base_prob`. The former is a genuine
  gain; the latter is the old negative-by-construction contrast and remains
  supplementary. `tests/test_xai.py` checks both identities with zero and
  non-zero backgrounds; it passes.
- **XAI sensitivity rerun:** `experiments/run_xai.py` now evaluates both the
  primary training medoid and a class-balanced mean-of-class-medoids
  background. 36 models = **72 baseline evaluations** (173 s). Added
  `results/xai_baseline_sensitivity.csv`; mean |delta Top5Ratio| = 0.0343
  (max 0.0986), mean |delta IG/SHAP deletion AUC| = 0.0115/0.0140. Worst mean
  relative IG completeness error remains 0.0044; KernelSHAP efficiency is exact
  by construction, not an accuracy claim.
- **Wording fixes:** `src/xai.py` docstring no longer says “zero baseline”
  without qualification; report/supplement state 256 sampled KernelSHAP
  coalitions and the KKT efficiency caveat. Under the primary background,
  CuCD-ID/GRU BiC and iCaRL+NCM rank `UniqueMessageIDsInWindow` first, whereas
  MLP cells and GRU/joint_scratch rank `TimeRadians` first. Under the
  class-balanced sensitivity background, GRU rankings can shift to
  `TimeRadians` or `MemoryAnonMB`, so the stable reading is time/window
  aggregates, not one feature.
- **Outputs regenerated:** leakage/XAI CSVs, `XAI_SUPPLEMENT.md`,
  `RESULTS_SUMMARY.md`, `v6_5_project_report_final.md`, 33 figures, and
  `final_experiment_manifest.json`.
- **Verification:** all nine `tests/test_*.py` scripts pass (`test_gradients.py`
  included); `compileall` succeeds.

## 2026-10-26 — v6.4 NCM control, XAI revision, and 315-run feature ablation (superseded by v6.5)
- **Artifact identity:** package/report version set to **v6.4**; current report
  is `v6_4_project_report_final.md`. This snapshot added the NCM control and
  XAI/leakage extensions; v6.5 corrects the duplicate alias, insertion-metric
  label, and baseline sensitivity. The v6.3 temporal-validity audit remains
  below as history.
- **NCM audit:** added `experiments/check_ncm_sanity.py` and ran 108
  iCaRL/control runs (90 iCaRL + 18 joint). Existing `icarl_ncm` ACC is
  task-balanced; the new runner matches it exactly on the overlapping cells
  (max |delta| 0.0 for ACC and Macro-F1) and also records overall sample ACC.
  NCM-all minus NCM-buffer has max |delta| 0.0065 ACC, so buffer capacity is
  not the limitation. NCM-all is below embedding-herding softmax in 6/6
  iCaRL cells; the joint head-only contrast is near parity on GRU (+0.0001
  CuCD-ID, +0.0094 STIN) but prefers softmax on CuCD-ID MLP/CNN1D and STIN
  MLP/CNN1D. Report this as a classifier/feature-space result, not an NCM
  superiority claim.
- **Outputs:** `results/ncm_sanity_{runs,summary,vs_main,vs_main_runs,per_class}.csv`
  and `results/accuracy_matrices_ncm_sanity.json`.
- **XAI revision:** `run_xai.py` now uses a real training-medoid background
  (zero is degenerate for cosine/NCM), defaults to 512 IG steps, and emits
  correctness subsets, completeness diagnostics, group attribution, and group
  deletion/insertion. Full rerun: 36 runs. Worst mean relative IG completeness
  error is 0.0044 (SHAP ~1e-15). Correct-only Top5Ratio is highest for
  CuCD-ID/MLP/joint_scratch (0.808); NCM Top5Ratio is lower, so agreement and
  faithfulness are reported together.
- **Outputs:** `results/xai_metrics.csv`, `xai_summary.csv`,
  `xai_metrics_by_correctness.csv`, `xai_summary_by_correctness.csv`,
  `xai_group_attribution_{runs,summary}.csv`,
  `xai_group_deletion_{runs,summary}.csv`, `xai_group_summary.csv`,
  `xai_examples.csv`, `xai_feature_{importance,summary}.csv`, and the generated
  `results/XAI_SUPPLEMENT.md`.
- **Feature ablation:** extended `run_leakage_ablation.py` to 7 variants
  (drop-one, all-provenance-drop, provenance-only, raw-only) x 3 methods x 15
  seeds = 315 retrained runs. Added `results/leakage_group_effects.csv`.
  The raw-only variant was an exact alias of all-provenance-drop; v6.5
  removes it and reports 270 unique runs.
  `no_time_ids_window` / `only_raw_telemetry` remain the strongest drop
  (CuCD-ID/MLP: ACC 0.9153, -0.0771 vs all for BiC; replay 0.9229, -0.0666),
  while `only_time_ids_window` retains most accuracy (0.9843 BiC). The
  provenance-bearing fields therefore carry most of the random-split advantage;
  this is reliance, not proof that every field is illegitimate.
- **Docs/report:** corrected the old "isolates the classifier change" claim in
  `run_icarl_ncm.py`, `README.md`, `PAPER_PLAN.md`, `PROJECT_CONTEXT.md`, and
  the report generator. Added NCM control, correct-only XAI, completeness,
  feature-group sections, and updated `run_all.py`, manifest, figures (33), and
  reports. Main XAI metrics are correct-only Top5Ratio/deletion/insertion;
  all/error rows stay in the supplement.
- **Verification:** `test_ncm.py`, `test_xai.py`, and `test_feature_groups.py`
  pass; `test_gradients.py` remains part of the full suite. Reports regenerated
  from CSV/JSON only.

## 2026-10-23 — v6.3 final packaging + temporal leakage/validity audit (read-only; no retraining)
- **Artifact identity:** package/report version set to **v6.3**; current report
  is `v6_3_project_report_final.md`. Historical v5.x/v6.2 entries remain below.
- **Purpose:** quantify the random split's temporal adjacency and document why
  a true scenario split is unavailable, without training models or changing
  any confirmatory result.
- **Split replay:** `experiments/run_temporal_leakage_audit.py` replays the
  exact `build_cucd_id_tasks` random per-class splits for seeds 0-14, advances
  the shared RNG with the same per-task `rng.permutation` call, and asserts
  that the actual scaled train/test/val rows match the task builder (not just
  row counts).
- **Nearest-train distance:** measured on circular `TimeRadians` against the
  training rows actually drawn (700/class for T0, 150/class for T1-T3), not
  the full pre-draw train pool. Pooled over 15 splits, 98.6% of test rows are
  within 0.02 rad of a drawn training row. Per class: normal 100.0%,
  command_flooding 100.0%, data_injection 100.0%, defence_impairment 94.6%,
  storage_exhaustion 98.6%; medians 2.3e-5 to 4.8e-3 rad.
- **Natural blocks:** 5 blocks with gaps > 0.02 rad after unwrapping circular
  time. Normal, command_flooding, data_injection, and storage_exhaustion each
  occupy one block; defence occupies 4. Leave-one-block-out is therefore
  class-incomplete and not a valid four-task scenario estimate.
- **Feature drift:** within-class median-time split, pooled within-phase SD,
  `TimeRadians` excluded. Mean |z| per class 0.20-0.82; peak |z| 4.87 on
  `SlidingWindowMeanIntervalSec` (data_injection). The temporal-purged gap is
  a joint leakage + drift + scaler/prior-shift estimand.
- **Outputs:** `results/temporal_leakage_audit.{csv,json}`,
  `temporal_leakage_audit_summary.csv`, `temporal_leakage_histogram.csv`,
  `temporal_class_occupancy.csv`, `temporal_block_coverage.csv`,
  `temporal_class_block_ranges.csv`, `temporal_drift_by_class.csv`,
  `temporal_drift_features.csv`, and `figures/temporal_leakage_audit.png`.
- **No retraining.** Confirmatory CSVs and run counts are unchanged; only
  diagnostic outputs, reports, figure, docs, and manifest were updated.

## 2026-10-23 — v6.2 packaging + claim reframing (no rerun)
- **Artifact identity:** package/report version set to **v6.2**; current report
  is `v6_2_project_report_final.md`. Historical v5.x entries remain below.
- **Packaging:** removed the empty brace-expansion directory
  `{src,experiments,tests,data`, Python caches, and the stale untracked generic
  bundle.
- **Frozen-hparam guard:** strict loader now validates coverage and required
  keys for all 3 architectures x 9 methods; no frozen values changed.
- **Claim boundary:** report/README reframed as a benchmark and negative-result
  study. No CSV/JSON result values were regenerated; only reports/manifest and
  documentation changed.

## 2026-10-22 — Confirmatory 15-seed rerun (completed; supersedes the earlier same-day 540-run entry)
- **Split protocol:** validation is 20% of each per-class pool, then split 50/50
  into disjoint calibration and selection halves (150/150 per class at the
  cap). BiC and the EWC Fisher use calibration only; all tuning scores use
  selection only. The split is keyed by stable class identity so task-order
  permutations preserve the same calibration/selection rows.
- **Seeds/stats:** main matrix expanded from 10 to 15 seeds (810 runs). The
  fixed all-pairwise family is 216 comparisons per metric, Holm-corrected
  across the full family: **162/216 ACC** and **164/216 Macro-F1**
  comparisons survive at p_holm<0.05. All stats are pre-specified and
  exploratory, not selected after seeing results.
- **Few-shot:** 420 runs covering `fixed` (150 calib + 150 select per class)
  and `scaled` (K calib + K select per class) protocols; calibration payload
  is charged only to methods that consume it.
- **Task order:** 144 runs; two-class-T0-first permutations are primary and
  single-class-first permutations are reported separately as degenerate
  no-op updates. Current primary per-order mean spread is 0.0146.
- **Temporal/leakage:** 270-run three-protocol temporal study and 225-run
  feature-provenance leakage ablation. The current `no_window` numbers
  supersede older temporal/leakage entries. Mean deployable-method ACC falls
  from 0.6768 (IID) to 0.2240 (temporal-purged) and 0.1895 with
  time/ID/window groups removed.
- **Memory:** largest persistent `full` state is 18.53 KB; largest
  host-estimated peak update RAM is 348.34 KB. The retained-update column is
  the cumulative calibration half, not deployed inference state.
- **Controls:** 450 matched-replay/EWC-lambda runs; exact paired Holm results
  in `results/stats_baseline_controls.csv` (7/30 significant).
- **Execution:** four worker processes with one BLAS thread each. Wall time
  4h07m including an ~8 min restart after fixing a missing `time` import;
  compute time 3h59m. Main-vs-few-shot consistency passes
  (`k150_max_abs_acc_diff=0.0`).
- **Full pipeline regenerated:** 810 main runs, 756 tuning rows (2268
  seed-runs), figures, manifest, `RESULTS_SUMMARY.md`, and
  `v5_4_project_report_final.md`. Current counts are recorded in
  `results/final_experiment_manifest.json`. Do not mix these numbers with
  older rows.

## 2026-10-22 — Priority-1/2 correctness fixes + full rerun (completed; superseded by the 15-seed entry above)
- **BiC composition:** `BiasCorrector.fit_stage` now applies the current
  corrector to validation logits before fitting the next stage; BiC was
  re-tuned separately and all BiC/EWC/full rows rerun. Added
  `tests/test_bic.py`.
- **Task order:** isolated canonical splits/scaler with 12 class permutations
  x 2 budget modes (144 runs). Budget-mode gap 0.0015, per-order mean spread
  0.0824; legacy confounded task-order headline retired.
- **Temporal/leakage:** 270-run three-protocol temporal study and 150-run
  feature-provenance rerun. Current temporal and `no_window` numbers
  supersede the 2026-10-21 temporal/no_window entries.
- **Memory:** peak RAM now includes batch-32 activations and retained
  cumulative-validation state for EWC/full/BiC; activation and full ledgers
  are separate. Largest persistent state 24.17 KB; retained validation
  70.31 KB (CuCD-ID) / 168.75 KB (STIN); largest host-estimated peak
  348.34 KB.
- **Priority-2 controls:** 300 matched-replay/EWC-lambda runs; exact paired
  Holm results in `results/stats_baseline_controls.csv` (0/30 significant).
- **Stats:** exact p-value floor fixed; no zero p-values. Main/few-shot
  consistency passes (`k150_max_abs_acc_diff=0.0`).
- Full pipeline regenerated: 540 main runs, 756 tuning rows (2268
  seed-runs), figures, manifest, `RESULTS_SUMMARY.md`, and
  `v5_4_project_report_final.md`. Do not mix these numbers with older rows.

## 2026-10-21 — v5.4 Q2/Q3 correctness rerun (completed)
- **Data provenance:** CuCD-ID is simulated NOS3/cFS software-in-the-loop
  telemetry (Fayyaz et al., Data in Brief 2026,
  doi:10.1016/j.dib.2026.112598; Mendeley doi:10.17632/7n2d42pm3n.3);
  STIN is Li et al., IEEE Access 2020, doi:10.1109/ACCESS.2020.3041641.
- **Ordering:** temporal/purged splits now use a largest-circular-gap rotation
  + `np.unwrap` on `TimeRadians`, so the 0→2π wrap is chronological within
  each class block.
- **EWC:** Fisher is drawn from a class-balanced cumulative old+new
  validation pool; frozen configs are regenerated for all 9 methods.
- **New artifact:** `leakage_ablation.csv` / `_summary.csv` retrain
  CuCD-ID/MLP with time, CCSDS-ID, and 20-s-window groups removed.
- **Stats:** separate ACC and Macro-F1 exact permutation/Holm files;
  STIN comparisons added.
- Full pipeline rerun on this machine; all results, figures, manifest, and
  reports were regenerated. Current run counts, metrics, tests, ablations,
  and limitations are recorded in `results/final_experiment_manifest.json`,
  `results/RESULTS_SUMMARY.md`, and the auto-generated
  `v5_4_project_report_final.md`.
- Consistency now writes `results/consistency_check.json`; the manifest reads
  that status instead of asserting a hardcoded pass. The check accepts all 9
  tuned methods (joint/joint_scratch included).
- Historical `v5_3_project_report_final.md` is retained with a superseded
  banner; cite the generated v5.4 report.

## 2026-09-21 — v5.3 report corrections (text-only, no reruns)
- All `v5_3_project_report_final.md` numbers re-verified against on-disk CSVs
  (all match; two 0.1pp rounding fixes applied); all §2 base-paper cells
  verified against Base_Paper.pdf (seeds/AUC+PAUC/few-shot rows corrected).
- Contradictory wording fixed: STIN role, proposed-CMMU, strict-upper-bound,
  hard-60KB, task-order spread, few-shot seeds, herding scope, §5.11 buffer
  claim; peak-RAM line and CNN significance row added.
- One comment fix (`cmmu.py` budget typo). No training/CSV changes.

## 2026-09-21 — Fail-loud frozen hyperparameters (no reruns)
- All experiment scripts now abort with an actionable error if
  `best_hparams_per_method_arch.json` is missing (strict loader in
  `config.py`); legacy silent fallbacks deleted. Verified missing-file error
  path + normal path + consistency ALL CHECKS PASSED. No CSVs touched.

## 2026-09-21 — v5.1 cleanup (docs/figures/quarantine wiring; NO training reruns)
- PROJECT_CONTEXT.md rewritten to v5.1 (540 runs, frozen configs, roles, protocols).
- `run_arch_tune.py` added to `run_all.py` as labeled exploratory step.
- Figures use 95% CI error bars (1.96·SE); 20 PNGs regenerated.
- Val-slice comments → Option A; tuning stores seed-means; frozen JSON keys
  relabeled to `val_metrics_last_seed` + note (values untouched).
- Sweep/legacy wording cleaned; report + manifest + consistency regenerated.
- Consistency ALL CHECKS PASSED. No training numerics changed: all 540 +
  Phase-2 training CSVs remain valid.

## 2026-09-21 — v5 fixes: uplink Option A + gamma wording (no 540 rerun)
- Uplink budget now includes onboard calibration data (BiC/Fisher val):
  `Task.uplink_calib_bytes_*` added; `fewshot.csv` rerun (210 runs, same
  seeds/configs — ACC/BWT/FM/Macro-F1 unchanged, only new KB columns);
  report §4/§7.1 show adaptation + calibration + totals (K=150: 10.1KB
  adaptation + 10.1KB calib = 20.2KB total; calib dominates at low K).
- Gamma record corrected explicitly here: final frozen EWC configs are
  MLP γ=0.5 / GRU γ=2.0 / CNN γ=0.5 (validation-tuned weighting, not decay;
  GRU 2.0 amplifies by selection). `train.py`/`cmmu.py` `gamma=1.0` values
  are fallback defaults only, always overridden by frozen configs. No prior
  DATALOG entry stated otherwise for final configs.
- Consistency ALL CHECKS PASSED; report regenerated. No other CSV touched.

## 2026-09-21 — v4 review hardening (stats/terminology/quarantine; NO training reruns)
- `run_stats.py` rewritten to exact 2^n enumeration; `stats_permutation.csv`
  regenerated (all p-values dyadic k/1024; significance pattern unchanged:
  MLP replay/full vs finetune, GRU icarl vs full, CNN bic vs replay significant
  at Holm 0.05; icarl-vs-replay ties stand).
- Report regenerated (34066 chars): §7.7 exact p-values, §7.4 purged table,
  §1 frozen-config table, §3 fixed KB units, §5.3 legacy superseded note,
  per-arch frozen gamma line, Macro-F1 primary tables.
- 20 figures regenerated (joint_scratch legend "offline ref.", task-order title).
- Manifest regenerated ("offline joint-training reference" + legacy-quarantine
  section + estimated-MAC wording). Legacy CSVs quarantined to results/legacy/.
- Consistency: ALL CHECKS PASSED. No training-code numerics changed, so 540 +
  Phase-2 CSVs remain valid; only stats/report/figures/manifest regenerated.
- Addendum (same day): `arch_tune.csv` rerun (54 runs, V-scored, frozen base;
  it predated its own fix and is not in `run_all.py`); `arch_tune.png` +
  report §7.2 regenerated. Verified stale-artifact audit: every other CSV
  postdates the code that wrote it — no further reruns required.

## 2026-09-20 — Publication hardening round 2 FULL RERUN COMPLETED
- Code: robustness + per-class + task-order routed to frozen per-method/arch configs; arch_tune quarantined to validation-scored exploratory; bounded herding (buffer + incoming batch only); teacher returned with performed train/teacher-sample counters; peak RAM uses per-step replay minibatch + real teacher; MAC/update from performed counts; Macro-F1 BiC-corrected in main + few-shot; report/plots stale readers fixed (stats_permutation, temporal_purged_compare, int8_sim, memory KB units, Macro-F1 panels, 10-seed labels); purge empty-split guards + honest assert docstring; consistency checker fixed (CuCD-ID filter, intersecting seeds, PLAN methods); run_all extended (task-order, consistency, manifest).
- Rerun: gradient checks pass; sweep + validation tuning re-ran (fresh frozen file); main 540; purged-temporal 180; few-shot 210; feature-order 96; task-order 144; ablations; per-class (frozen); stats (9 headline, Holm); robustness (frozen, int8_sim with corrector); consistency ALL CHECKS PASSED (K=150 diffs 0.0000 on intersecting seeds); 20 figures; manifest; report (33805 chars).
- Fresh headlines (CuCD-ID/MLP, 10-seed means): bic 0.956/0.963, icarl 0.941/0.943, replay 0.919/0.928, full 0.863/0.873, collapse 0.250; temporal-purged replay 0.919→0.210, scratch 0.994→0.159; task-order spread 0.0495; persistent max 24.17 KB (40.3% assumed 60 KB), peak max ~127 KB host-estimated.
- Wording: README + PAPER_PLAN rewritten to code; report prose qualified (assumed budget, packed payload, simulated INT8, legacy-order superseded, EWC variant).
- Tag: `q2-hardening-round2` (superset of `q2-submission-v1` outputs; all CSVs/figures/manifest/report regenerated, nothing hand-edited).

## 2026-09-20 — Q2-Submission hardening FULL RERUN COMPLETED (all 8 fixes, ~1,700 runs)
- **BiC fix:** Stage-local correction applied to `n_old:n_old+n_new` only; `n_new` stored per stage. All BiC numbers regenerated.
- **Train/Val/Test:** Sweep uses `val_frac=0.10` on STIN; hparams selected on `ACC_val - FM_val`; `train_continual` returns `V` matrix.
- **Validation tuning (NEW):** `run_validation_tune.py` → 7 methods × 3 archs × 3 val seeds on CuCD-ID → `best_hparams_per_method_arch.json` frozen.
- **Purged temporal (NEW):** `build_cucd_id_tasks_purged_temporal()` with 0.02 rad (~22s equivalent) purge gaps on `TimeRadians`; STIN keeps `id`-temporal (no purge). `run_purged_temporal.py` → dual-protocol `temporal_purged_compare.csv` (MLP × 9 methods × 10 seeds × 2 protocols).
- **Resource accounting:** Persistent (INT8 model + INT8 replay + FP16 EWC) + Peak RAM (FP32 weights + gradients + Adam 2×params + activations + batch + replay_batch + teacher + temp); MAC/update.
- **Main matrix:** 10 seeds (540 runs) with frozen per-method/arch hparams → `main_results_runs.csv` (new columns: MacroF1, BalancedAccuracy, persistent_*, peak_total_KB, macs_per_update).
- **Few-shot (ENHANCED):** 5 seeds, inherits frozen hparams, K=5..300 → `fewshot.csv` with MacroF1/BWT/FM/uplink_KB_total.
- **Feature-order (NEW):** 4 perms × MLP/GRU/CNN × 4 methods × 2 seeds → `feature_order_stability.csv` (MLP stable, GRU/CNN sensitive).
- **Stats upgrade:** `run_stats.py` → exact paired permutation test + Holm + Cohen's d + 95% CI on headline comparisons.
- **INT8 wording fix:** Corruption renamed `int8` → `int8_sim`; docstring: "simulated post-training INT8 quantization accuracy proxy; NOT actual INT8 hardware inference".
- **Temporal purge naming fix:** Parameter `purge_seconds` → `purge_time_rad` (0.02 rad); documented as ~22s equivalent on TimeRadians.
- **Consistency check:** `check_consistency.py` verifies main vs few-shot hyperparameter match, seed counts, config provenance.
- **Reproducibility manifest:** `create_manifest.py` → `results/final_experiment_manifest.json` with git commit, env, datasets, protocols, frozen hparams, outputs, limitations.
- **Figures/Report:** 20 figures (CI bars, Pareto, dual protocol) + `RESULTS_SUMMARY.md` with Macro-F1 primary tables.
- All outputs tagged `q2-submission-v1` for traceability.

## 2026-09-19 — Q2-Submission hardening full rerun (all 8 fixes, ~1,700 runs)
- **BiC fix:** Stage-local correction applied to `n_old:n_old+n_new` only; `n_new` stored per stage. All BiC numbers regenerated.
- **Train/Val/Test:** Sweep uses `val_frac=0.10` on STIN; hparams selected on `ACC_val - FM_val`; `train_continual` returns `V` matrix.
- **Validation tuning (NEW):** `run_validation_tune.py` → 7 methods × 3 archs × 3 val seeds on CuCD-ID → `best_hparams_per_method_arch.json` frozen.
- **Purged temporal (NEW):** `build_cucd_id_tasks_purged_temporal()` with 0.02 rad (~22s equivalent) purge gaps on `TimeRadians`; STIN keeps `id`-temporal (no purge). `run_purged_temporal.py` → dual-protocol `temporal_purged_compare.csv` (MLP × 9 methods × 10 seeds × 2 protocols).
- **Resource accounting:** Persistent (INT8 model + INT8 replay + FP16 EWC) + Peak RAM (FP32 weights + gradients + Adam 2×params + activations + batch + replay_batch + teacher + temp); MAC/update.
- **Main matrix:** 10 seeds (540 runs) with frozen per-method/arch hparams → `main_results_runs.csv` (new columns: MacroF1, BalancedAccuracy, persistent_*, peak_total_KB, macs_per_update).
- **Few-shot (ENHANCED):** 5 seeds, inherits frozen hparams, K=5..300 → `fewshot.csv` with MacroF1/BWT/FM/uplink_KB_total.
- **Feature-order (NEW):** 4 perms × MLP/GRU/CNN × 4 methods × 2 seeds → `feature_order_stability.csv` (MLP stable, GRU/CNN sensitive).
- **Stats upgrade:** `run_stats.py` → exact paired permutation test + Holm + Cohen's d + 95% CI on headline comparisons.
- **Figures/Report:** 20 figures (CI bars, Pareto, dual protocol) + `RESULTS_SUMMARY.md` with Macro-F1 primary tables.
- All outputs tagged `q2-submission-v1` for traceability.

## 2026-09-18 — Phase 3 integration (no new trainings; reporting + mechanism only)
- `analysis/plots.py` → 20 figures (16 refreshed for 9 methods/5 seeds + `fewshot_pareto`, `arch_tune`, `temporal_bracket`, `robustness` new). `RESULTS_SUMMARY.md` regenerated with §7.1–7.7 (DET method-column bug fixed post-generation).
- `config.ARCH_TUNED_HPARAMS={CNN1D: lr=0.01, epochs=50}` wired into run_main + per-class via `hparams_for()`. ⚠️ ON-DISK matrix/per-class CSVs remain FROZEN-hparam (CNN lr=0.003/25ep); tuned CNN lives only in `arch_tune.csv` until next full rerun. Tag after rerun: `phase3-archtunedCNN`.
- `run_all.py`: CORE/PHASE2/REPORT + `--no-phase2` (~30–45 min core vs ~1.5–2.5 h full); `sys.executable` fix (system `python3` lacks numpy → old script failed step 1).

## 2026-09-18 — Phase 2e robustness/quant/memory (`results/robustness.csv`: 12 trainings MLP × replay,icarl,bic,scratch × 3 seeds + 7-corruption eval; `results/memory_ledger.csv` analytic)
- Pooled ACC: gauss_0.5 ≈ clean (−0.02); drop_0.3 −0.19; `nowindow` −0.25..−0.34 (replay 0.965→0.633) — a third of accuracy depends on aggregate window features: disclose alongside leakage limit. int8 PTQ: no loss (0.958–0.998).
- Activations (batch=1, analytic): MLP 0.395KB / GRU 3.03KB / CNN 2.81KB fwd (2× train est.); all-configs deployed total <20KB of 60KB budget. All SIM_-prefixed = host-simulated, not MCU-measured.
- Order effect does NOT replicate under val protocol: fixed_position replay 0.975±0.026, icarl 0.983±0.017; moves_with_task 0.978/0.981. Spread ~0.01 vs legacy 0.064. normal-first slightly worse (0.966 vs 0.985). Legacy headline was an artifact of no-val + `full`/3-seed noise (and excluded outlier seed 3 here). Paper must retract/replace the old order claim with this null result.

## 2026-09-18 — Phase 2d temporal-vs-random (`results/temporal_compare.csv`, 40 runs: MLP × replay,icarl,bic,scratch × random|temporal × 5 seeds)
- Random→temporal collapse for ALL methods incl. offline scratch: replay 0.923→0.202, icarl 0.967→0.288, bic 0.904→0.271, scratch 0.997→0.140. Diagnostic (scratch/temporal/seed0): train resubstitution 1.000 vs test 0.155; T0 self-accuracy 0.856→0.396 across tasks; new-task test ≈0. Cause: within-class mission-time drift — each attack occupies its own TimeRadians phase (e.g. storage_exhaustion spans 0.001–6.28 with MemoryAnonMB 19→74 early-vs-late); early-train vs late-test = attack-intensity extrapolation, and the T0-frozen scaler mismatches. So temporal-blocks ≠ leakage-free i.i.d. eval.
- Paper position: headline stays random-split (i.i.d. within-scenario) WITH explicit sliding-window-leakage limitation; temporal reported as forward-extrapolation stress bound (truth between the two). A true group/scenario split needs scenario IDs the CSV lacks — flagged as data-collection future work, not fixable by splitting alone.
- CNN is optimization-starved at frozen hparams: replay 0.484 (lr .003/25) → 0.718 (lr .01/50); icarl 0.463 → 0.856; scratch 0.841 → 0.961 (lr .01/50). Residual CL gap at best tune ≈ 0.10 (scratch 0.961 vs icarl 0.856). Conclusion: per-arch tune mandatory before any "CNN fails at CIL" claim; CNN viable but 2× epoch cost. Paper must report tuned CNN (lr=0.01/50ep) alongside frozen-hparam numbers, never silently swap.
- Flag (a) from Phase-1 rerun resolved as optimization, not CL failure.

## 2026-09-18 — Phase 2b few-shot uplink (`results/fewshot.csv`, 182 runs: CuCD-ID, T0=700 fixed, K{5,10,25,50,100,150,300}, val_frac 0.10)
- MLP mean ACC: icarl K=5: 0.732 (vs replay 0.509, bic 0.682) → K=50: 0.902 → K=150: 0.967 (reproduces matrix) → K=300: 0.990. Knee at K≈50 for MLP/icarl; replay/full need K≥100–150 to match. GRU needs K≥150 (icarl 0.736, replay 0.571) and only converges at K=300 (0.807/0.864). Uplink KB per K in `uplink_KB_total` column for the Pareto plot.
- RQ4 answer draft: minimum viable uplink ≈ 50 samples/class for MLP+iCaRL, ≥150 for GRU; 5-shot works only with herding+distill+BiC (icarl/bic), never plain replay.
- K=150 column consistency-checks the Phase-1 matrix (MLP replay 0.923, icarl 0.967, bic 0.904 — match).
- Protocol: random split + `val_frac=0.10`, frozen STIN/MLP/full hparams (`lr=0.003, lambda=100, buf=256, frac=0.9`), 5 seeds, 9 methods. Ablations also rerun at 5 seeds (buffer 180 rows, lambda 210 rows); per-class + stats + figures + `RESULTS_SUMMARY.md` regenerated.
- CuCD-ID ACC: MLP `icarl 0.967 > replay 0.923 ≈ full 0.921 > bic 0.904 >> collapse 0.250` (icarl-vs-replay p=0.345 n.s.; full-vs-replay p=0.50 n.s.); GRU `icarl 0.759 > replay 0.633 ≈ full 0.610 ≈ bic 0.604` (icarl-vs-replay p=0.043); CNN `bic 0.597 > replay 0.484 ≈ icarl 0.482 > full 0.404` (bic-vs-all p=0.043; icarl-vs-replay p=0.89 n.s.). Bounds: `joint_scratch` tops on MLP (0.997) / GRU (0.984) but **loses to warm `joint` on CNN** (0.858 vs 0.898) — CNN optimization flag.
- STIN ACC: MLP `joint 0.661 ≈ scratch 0.657 ≈ bic 0.655 ≈ icarl 0.651 ≈ full 0.648 > replay 0.622` (bic-vs-joint p=0.89 n.s. — at ceiling); GRU `joint 0.664 > replay 0.571 ≈ scratch 0.576 > icarl 0.523`; CNN `joint 0.535 ≈ icarl 0.524 (p=0.69 n.s.) > replay 0.500 ≈ bic 0.495`, scratch collapses to 0.395 — same CNN-from-scratch flag, stronger.
- Detection (CuCD-ID, 1-p(normal)): AUROC ~0.999–1.0 for all rehearsal methods; collapsed `finetune/ewc/lwf` still 0.91–0.99 AUROC but FPR95 exposed (MLP ~0.45, GRU ~0.66–0.84) — classification-vs-detection dissociation confirmed at scale.
- Open anomalies for Phase 2: (a) CNN `scratch<joint` on both datasets; (b) `bic` best on CNN/CuCD-ID but 4th on MLP; (c) leakage still unquantified (random-only); (d) task-order still 3 confounded perms; (e) no few-shot / corruption / quant evidence yet.

## 2026-09-17 — Phase 0 code (protocol fixes, old results still valid as baseline)
- `src/data_utils.py`: `Task.X_val/y_val` + `TaskSequence.split_mode/val_frac`; legacy builders unchanged when `val_frac=0.0` (verified: same 1400/150 train + same test sizes); new `build_*_tasks_temporal()` (CuCD-ID sorted by `TimeRadians`, contiguous train|val|test-future blocks).
- `src/train.py`: added `joint_scratch` (re-init each boundary = strict upper bound), `BiasCorrector` hook (`bias_correction` flag), `replay_balance` (`uniform` default preserves old numbers). Smoke (MLP, 2 epochs, seed 0, val_frac 0.10): `joint(warm) 0.979 > joint_scratch 0.926` — warm-start converges faster at low epochs, which is exactly why the old `joint` must not be called a strict bound.
- `src/cmmu.py`: added `sample_balanced()`, `select_herding()`, val-balanced Fisher mix. Gradient checks still pass; smoke `full+bic+balanced` runs end-to-end.
- Comparability: all CSVs under `results/` are still the pre-fix random-split, no-val baseline. New-split reruns must be tagged `phase0-temporal+val` and NOT mixed with old numbers in one table.

## 2026-09-17 — Phase 1 validation (CuCD-ID / MLP / 25 epochs / 5 seeds, frozen STIN-tuned hparams)
- `joint_scratch 0.997 > joint(warm) 0.995` — bound ordering now correct (was inverted at 2-epoch smoke).
- Continual ranking: `icarl 0.967 ± 0.036 > replay 0.923 ± 0.083 ≈ full 0.921 ± 0.086 > bic 0.904 ± 0.083 >> lwf/finetune/ewc 0.250` (collapse). Seed 3 outlier for replay/full (~0.75) justifies 5-seed + Wilcoxon reporting.
- `lwf` collapsing like finetune replicates literature (memory-free KD insufficient in CIL) — report as negative result, not a bug. `icarl`-lite leading is the headline to stress-test on GRU/CNN + STIN via full `run_all.py`.
- Detection (1-p(normal)): AUROC ~1.0 for all rehearsal methods, 0.914 collapsed / 0.954 lwf — classification-vs-detection dissociation holds at convergence.

## 2026-09-17 — Phase 1 code (SOTA baselines + detection + stats; results NOT yet rerun)
- Methods `lwf/bic/icarl/joint_scratch` added (`src/train.py`); KD additive fix (`CE + lam*KD`) after smoke caught convex-combo bug. 3-epoch MLP/CuCD-ID smoke: `bic 0.612 > full 0.415 ≈ replay 0.413 > icarl 0.326 > lwf 0.291`; detection AUROC ~1.0 even under ACC collapse, FPR95 separates (replay 0.001 vs lwf 0.738).
- Matrix definition changed: `METHODS` 5→9, `SEEDS` 3→5, `run_main` builds with `val_frac=0.10`, adds `DET_*` columns (CuCD-ID only), new `stats_wilcoxon.csv`.
- Comparability: existing `results/main_results_*.csv`, `per_class_*`, `ablation_*` are **Phase-0-baseline** (random split, no val, 3 seeds, 5 methods). Full rerun via `run_all.py` will overwrite them with Phase-1 protocol (270-run matrix). Tag new numbers `phase1-9methods-5seeds-val010`.

## 2026-09-17 — Phase 0 baseline (pre-fix, reproducing repo state)
- Datasets: `data/CuCD-ID/noised_dataset.csv` (22,465 rows, 22 feats + `Label` 0..4, mapping in `label_schema.json`), `data/STIN/SAT20.csv+TER20.csv` (209,564 rows, 30 feats, 9 attack classes, no benign).
- Splits: `build_cucd_id_tasks(seed, test_frac=0.30, initial 700/cls, uplink 150/cls)`, `build_stin_tasks(seed, test_frac=0.20, max_test 1200/cls)`; **random row-shuffle per class, no validation split, scaler fit on T0-train only**. Known limitation: CuCD-ID `SlidingWindow*/InWindow` features correlate neighbouring rows → train/test leakage inflates absolute ACC (relative method ranking unaffected).
- Hparams frozen from STIN/MLP/full sweep: `lr=0.003, ewc_lambda=100, buffer=256, replay_frac=0.9` (`results/best_hparams.json`).
- Headline numbers (mean/3 seeds, `main_results_summary.csv`): CuCD-ID/MLP `finetune 0.250 / ewc 0.250 / replay 0.931 / full 0.931 / joint(warm-start) 0.996`; STIN/MLP `full 0.661 > joint(warm-start) 0.634` (low separability ceiling, within noise).
- `joint` in these CSVs = **cumulative warm-start** (continues weights), NOT from-scratch. True `joint_scratch` to be added in Phase 0.2; do not cite old `joint` as strict upper bound.
- Task-order ablation (`ablation_task_order.csv`): budgets move WITH the task (`[3,2,1,0]=0.995` vs canonical `0.931`) → confounded; disentangled rerun planned.

## How to log new runs (convention)
- `[DATE] — [code tag: e.g. phase0-temporal+val] | builder + seed(s) + test/val fracs + budgets + split_mode | hparams source | methods | outputs touched | note on comparability vs above baseline`
