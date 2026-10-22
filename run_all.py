#!/usr/bin/env python3
"""
Runs the complete experimental pipeline end-to-end for Q2 publication:

  CORE (Phase 1 - validation-based tuning + main matrix):
  1. Gradient checks (sanity)                    tests/test_gradients.py
  2. Hyperparameter sweep on STIN (val-based)    experiments/run_hparam_sweep.py
  3. Per-method/arch validation tuning (NEW)     experiments/run_validation_tune.py
  4. Main experiment matrix (10 seeds)           experiments/run_main.py

  PHASE 2 (RQ evidence, dual-protocol evaluation):
  5. Purged temporal evaluation (0.02 rad TimeRadians ~22s equiv.) experiments/run_purged_temporal.py
  6. Few-shot uplink sweep (ENHANCED)            experiments/run_fewshot.py
  8. Feature-order stability (NEW)               experiments/run_feature_order.py
  9b. Disentangled task-order study (frozen)      experiments/run_task_order.py
  9c. Exploratory CNN arch tune (validation-scored, not applied) experiments/run_arch_tune.py
  7. Ablation studies (frozen val-tuned hparams) experiments/run_ablation.py
  9. Per-class F1 / confusion matrix analysis    experiments/run_per_class_analysis.py
  10. Paired stats (ACC + Macro-F1, Holm)        experiments/run_stats.py
  11. Robustness/quant/memory ledger             experiments/run_robustness.py
  11b. Feature-provenance leakage ablation       experiments/run_leakage_ablation.py
  11c. Main-vs-fewshot consistency check         experiments/check_consistency.py

  REPORTING:
  12. Figures (95% CI, Pareto, dual protocol)   analysis/plots.py
  13. Experiment manifest                        experiments/create_manifest.py
  14. Results summary report                     experiments/generate_summary_report.py

Usage:
    python3 run_all.py
    # or skip the slow Phase-2 block:
    python3 run_all.py --no-phase2

Takes roughly 10-14 hours on a single CPU core (several thousand runs); the
expanded per-method tuning grid dominates runtime.
All outputs are written to results/ and figures/.
"""
import subprocess
import sys
import time
import os


STEPS_CORE = [
    ("Gradient checks", [sys.executable, "tests/test_gradients.py"]),
    ("Hyperparameter sweep (STIN, val-based)", [sys.executable, "experiments/run_hparam_sweep.py"]),
    ("Per-method/arch validation tuning (NEW)", [sys.executable, "experiments/run_validation_tune.py"]),
    ("Main experiment matrix (10 seeds)", [sys.executable, "experiments/run_main.py"]),
]

STEPS_PHASE2 = [
    ("Purged temporal evaluation (0.02 rad TimeRadians purge)", [sys.executable, "experiments/run_purged_temporal.py"]),
    ("Few-shot uplink sweep (ENHANCED)", [sys.executable, "experiments/run_fewshot.py"]),
    ("Feature-order stability (NEW)", [sys.executable, "experiments/run_feature_order.py"]),
    ("Disentangled task-order study (frozen)", [sys.executable, "experiments/run_task_order.py"]),
    ("Exploratory CNN arch tune (validation-scored, not applied)", [sys.executable, "experiments/run_arch_tune.py"]),
    ("Ablation studies", [sys.executable, "experiments/run_ablation.py"]),
    ("Per-class F1 / confusion matrix analysis", [sys.executable, "experiments/run_per_class_analysis.py"]),
    ("Paired stats (ACC + Macro-F1, exact permutation + Holm)", [sys.executable, "experiments/run_stats.py"]),
    ("Robustness/quant/memory ledger", [sys.executable, "experiments/run_robustness.py"]),
    ("Feature-provenance leakage ablation", [sys.executable, "experiments/run_leakage_ablation.py"]),
    ("Main-vs-fewshot consistency check", [sys.executable, "experiments/check_consistency.py"]),
]

STEPS_REPORT = [
    ("Figures (CI, Pareto, dual protocol)", [sys.executable, "analysis/plots.py"]),
    ("Results summary report", [sys.executable, "experiments/generate_summary_report.py"]),
    ("Experiment manifest", [sys.executable, "experiments/create_manifest.py"]),
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
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
