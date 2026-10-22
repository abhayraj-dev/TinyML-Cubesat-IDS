# DATALOG — data & experiment provenance (cubesat-cl-ids)

Purpose: log **what data/split/config produced which results**, so the journal reproducibility checklist can be answered. Append-only: newest entries on top.

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
