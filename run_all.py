#!/usr/bin/env python3
"""
Runs the complete experimental pipeline end-to-end for the confirmatory study:

  CORE (Phase 1 - validation-based tuning + main matrix):
  1. Gradient checks (sanity)                    tests/test_gradients.py
  2. Hyperparameter sweep on STIN (val-based)    experiments/run_hparam_sweep.py
  3. Per-method/arch validation tuning (NEW)     experiments/run_validation_tune.py
  4. Main experiment matrix (15 seeds)           experiments/run_main.py
  4b. iCaRL-lite + NCM evaluation (15 seeds)     experiments/run_icarl_ncm.py
  4c. NCM sanity/control (108 runs)              experiments/check_ncm_sanity.py
  4d. XAI study: SHAP vs IG + groups             experiments/run_xai.py

  PHASE 2 (RQ evidence, multi-protocol evaluation):
  5. Purged temporal evaluation (cold-start + fair random-past; 0.02 rad purge) experiments/run_purged_temporal.py
  5b. Temporal leakage/validity audit (read-only)   experiments/run_temporal_leakage_audit.py
  5c. BUAC update-admission gate (IID/cold-start/fair; MLP/GRU/CNN1D) experiments/run_buac.py
  5d. BUAC cross-architecture summary             experiments/summarize_buac.py
  6. Few-shot uplink sweep (ENHANCED)             experiments/run_fewshot.py
  7. Feature-order stability (NEW)                experiments/run_feature_order.py
  8. Disentangled task-order study (frozen)       experiments/run_task_order.py
  9. Exploratory CNN arch tune (validation-scored, not applied) experiments/run_arch_tune.py
  10. Ablation studies (frozen val-tuned hparams) experiments/run_ablation.py
  11. Per-class F1 / confusion matrix analysis    experiments/run_per_class_analysis.py
  12. Paired stats (ACC + Macro-F1, Holm)         experiments/run_stats.py
  13. Robustness/quant/memory ledger              experiments/run_robustness.py
  14. Feature-provenance leakage ablation         experiments/run_leakage_ablation.py
  15. Main-vs-fewshot consistency check           experiments/check_consistency.py

  REPORTING:
  16. Figures (95% CI, Pareto, temporal protocols) analysis/plots.py
  17. Experiment manifest                        experiments/create_manifest.py
  18. Results summary report                     experiments/generate_summary_report.py

Usage:
    python3 run_all.py
    # or skip the slow Phase-2 block:
    python3 run_all.py --no-phase2

Uses 4 worker processes by default (CL_JOBS or --jobs overrides); each worker
is pinned to one BLAS thread.  Expect roughly 4-6 hours on a 4-core laptop.
All outputs are written to results/ and figures/.
"""
import subprocess
import sys
import time
import os


STEPS_CORE = [
    ("BiC composition regression test", [sys.executable, "tests/test_bic.py"]),
    ("Exact p-value floor test", [sys.executable, "tests/test_stats_pvalue.py"]),
    ("Task-order split/scaler invariance", [sys.executable, "tests/test_task_order.py"]),
    ("Disjoint calibration/selection split", [sys.executable, "tests/test_calib_split.py"]),
    ("Gradient checks", [sys.executable, "tests/test_gradients.py"]),
    ("NCM unit tests", [sys.executable, "tests/test_ncm.py"]),
    ("Feature-group partition tests", [sys.executable, "tests/test_feature_groups.py"]),
    ("BUAC gate unit tests", [sys.executable, "tests/test_buac.py"]),
    ("Temporal sampling/purge tests", [sys.executable, "tests/test_temporal_sampling.py"]),
    ("XAI unit tests", [sys.executable, "tests/test_xai.py"]),
    ("Hyperparameter sweep (STIN, val-based)", [sys.executable, "experiments/run_hparam_sweep.py"]),
    ("Per-method/arch validation tuning (NEW)", [sys.executable, "experiments/run_validation_tune.py"]),
    ("Main experiment matrix (15 seeds)", [sys.executable, "experiments/run_main.py"]),
    ("iCaRL-lite + NCM evaluation (15 seeds)", [sys.executable, "experiments/run_icarl_ncm.py"]),
    ("NCM sanity/control (108 runs)", [sys.executable, "experiments/check_ncm_sanity.py"]),
    ("XAI study: SHAP vs IG + groups", [sys.executable, "experiments/run_xai.py"]),
]

STEPS_PHASE2 = [
    ("Purged temporal evaluation (cold-start + fair random-past; 0.02 rad purge)", [sys.executable, "experiments/run_purged_temporal.py"]),
    ("Temporal leakage/validity audit (read-only)", [sys.executable, "experiments/run_temporal_leakage_audit.py"]),
    ("BUAC admission control (MLP)", [sys.executable, "experiments/run_buac.py", "--arch", "MLP", "--prefix", "buac_mlp"]),
    ("BUAC admission control (GRU)", [sys.executable, "experiments/run_buac.py", "--arch", "GRU", "--prefix", "buac_gru"]),
    ("BUAC admission control (CNN1D)", [sys.executable, "experiments/run_buac.py", "--arch", "CNN1D", "--prefix", "buac_cnn1d"]),
    ("BUAC cross-architecture summary", [sys.executable, "experiments/summarize_buac.py"]),
    ("Few-shot uplink sweep (ENHANCED)", [sys.executable, "experiments/run_fewshot.py"]),
    ("Feature-order stability (NEW)", [sys.executable, "experiments/run_feature_order.py"]),
    ("Disentangled task-order study (frozen)", [sys.executable, "experiments/run_task_order.py"]),
    ("Exploratory CNN arch tune (validation-scored, not applied)", [sys.executable, "experiments/run_arch_tune.py"]),
    ("Ablation studies", [sys.executable, "experiments/run_ablation.py"]),
    ("Matched-memory / EWC-lambda controls", [sys.executable, "experiments/run_baseline_controls.py"]),
    ("Per-class F1 / confusion matrix analysis", [sys.executable, "experiments/run_per_class_analysis.py"]),
    ("Paired stats (ACC + Macro-F1, exact permutation + Holm)", [sys.executable, "experiments/run_stats.py"]),
    ("Robustness/quant/memory ledger", [sys.executable, "experiments/run_robustness.py"]),
    ("Feature-provenance leakage ablation", [sys.executable, "experiments/run_leakage_ablation.py"]),
    ("Main-vs-fewshot consistency check", [sys.executable, "experiments/check_consistency.py"]),
]

STEPS_REPORT = [
    ("Figures (CI, Pareto, temporal protocols)", [sys.executable, "analysis/plots.py"]),
    ("Results summary report", [sys.executable, "experiments/generate_summary_report.py"]),
    ("Experiment manifest", [sys.executable, "experiments/create_manifest.py"]),
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    os.environ.setdefault("CL_JOBS", "4")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS"):
        os.environ[var] = "1"
    print(f"Parallel pipeline: CL_JOBS={os.environ['CL_JOBS']}, "
          "one BLAS thread per worker")
    steps = list(STEPS_CORE)
    if "--no-phase2" not in sys.argv:
        steps += STEPS_PHASE2
    else:
        print("(skipping Phase-2 block: --no-phase2 given)")
    steps += STEPS_REPORT
    t_start = time.perf_counter()
    for name, cmd in steps:
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        t0 = time.perf_counter()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[FAILED] step '{name}' exited with code {result.returncode}")
            sys.exit(result.returncode)
        print(f"[done in {time.perf_counter()-t0:.1f}s]")
    print(f"\nAll steps completed in {time.perf_counter()-t_start:.1f}s.")
    print("Results: ./results/   Figures: ./figures/")
    print("Read results/RESULTS_SUMMARY.md for the full report.")


if __name__ == "__main__":
    main()
