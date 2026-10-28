#!/usr/bin/env python3
"""Create final_experiment_manifest.json for reproducibility."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from tuning_grids import (TUNING_GRIDS, TUNING_SEEDS, VAL_FRAC,  # noqa: E402
                          CALIB_FRAC, VAL_CAP_PER_CLASS,
                          CALIB_CAP_PER_CLASS, SELECT_CAP_PER_CLASS)
from config import SEEDS  # noqa: E402
import data_utils as du  # noqa: E402


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_git_status():
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_python_version():
    return sys.version.split()[0]


def get_numpy_version():
    try:
        import numpy
        return numpy.__version__
    except:
        return "unknown"


def get_pandas_version():
    try:
        import pandas
        return pandas.__version__
    except:
        return "unknown"


def _load_consistency():
    path = os.path.join(RESULTS_DIR, "consistency_check.json")
    if not os.path.exists(path):
        return {"status": "missing", "error": "run experiments/check_consistency.py"}
    with open(path) as f:
        return json.load(f)


def _read(path):
    full = os.path.join(ROOT, path)
    return pd.read_csv(full) if os.path.exists(full) else None


def _dataset_stats():
    cu_path = os.path.join(ROOT, "data/CuCD-ID/noised_dataset.csv")
    schema_path = os.path.join(ROOT, "data/CuCD-ID/label_schema.json")
    sat_path = os.path.join(ROOT, "data/STIN/SAT20.csv")
    ter_path = os.path.join(ROOT, "data/STIN/TER20.csv")
    cu = du.load_cucd_id(cu_path, schema_path) if os.path.exists(cu_path) else None
    st = du.load_stin(sat_path, ter_path) if os.path.exists(sat_path) and os.path.exists(ter_path) else None
    out = {}
    if cu is not None:
        feat_cols = [c for c in cu.columns if c not in ("Label", "label_name")]
        out["CuCD-ID"] = {
            "rows": int(len(cu)),
            "features": int(len(feat_cols)),
            "classes": int(cu["label_name"].nunique()),
            "class_names": sorted(cu["label_name"].dropna().unique().tolist()),
        }
    if st is not None:
        feat_cols = [c for c in st.columns if c not in ("Label", "label_name")]
        out["STIN"] = {
            "rows": int(len(st)),
            "features": int(len(feat_cols)),
            "classes": int(st["label_name"].nunique()),
            "class_names": sorted(st["label_name"].dropna().unique().tolist()),
        }
    return out


def _counts():
    main = _read("results/main_results_runs.csv")
    tuning = _read("results/validation_tuning_results.csv")
    few = _read("results/fewshot.csv")
    rob = _read("results/robustness.csv")
    feat = _read("results/feature_order_stability.csv")
    main_counts = {"runs": 0, "datasets": 0, "architectures": 0, "methods": 0, "seeds": 0}
    if main is not None and not main.empty:
        main_counts = {
            "runs": int(len(main)),
            "datasets": int(main["dataset"].nunique()),
            "architectures": int(main["arch"].nunique()),
            "methods": int(main["method"].nunique()),
            "seeds": int(main["seed"].nunique()),
        }
    extra = {
        "task_order_disentangled.csv": "task_order",
        "temporal_purged_compare.csv": "temporal_purged",
        "leakage_ablation.csv": "leakage_ablation",
        "per_class_f1_runs.csv": "per_class",
        "baseline_controls.csv": "baseline_controls",
        "stats_permutation.csv": "stats_acc",
        "stats_permutation_macrof1.csv": "stats_macrof1",
        "stats_baseline_controls.csv": "stats_baseline_controls",
    }
    counts = {
        "main_matrix": int(main_counts["runs"]),
        "validation_tuning_rows": int(len(tuning)) if tuning is not None else 0,
        "validation_tuning": (int(len(tuning)) * len(TUNING_SEEDS)) if tuning is not None else 0,
        "few_shot": int(len(few)) if few is not None else 0,
        "robustness": int(len(rob)) if rob is not None else 0,
        "feature_order": int(len(feat)) if feat is not None else 0,
    }
    for fname, key in extra.items():
        d = _read(os.path.join("results", fname))
        counts[key] = int(len(d)) if d is not None else 0
    xai = _read("results/xai_metrics.csv")
    if xai is not None and not xai.empty:
        counts["xai_models"] = int(len(xai[
            ["dataset", "arch", "method", "seed"]].drop_duplicates()))
        counts["xai_baseline_evaluations"] = int(len(xai))
    buac_runs = 0
    buac_stage_rows = 0
    for arch in ("mlp", "gru", "cnn1d"):
        d = _read(os.path.join("results", f"buac_{arch}_stage_signals.csv"))
        if d is not None and not d.empty:
            buac_stage_rows += int(len(d))
            buac_runs += int(len(d[["arch", "method", "protocol", "seed"]].drop_duplicates()))
    counts["buac_runs"] = buac_runs
    counts["buac_stage_rows"] = buac_stage_rows
    sensitivity = _read("results/xai_baseline_sensitivity.csv")
    if sensitivity is not None:
        counts["xai_sensitivity_cells"] = int(len(sensitivity))
    return main_counts, counts


def main():
    ds_stats = _dataset_stats()
    main_counts, run_counts = _counts()
    manifest = {
        "project": "cubesat-cl-ids",
        "version": "v6.5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git": {
            "commit": get_git_commit(),
            "status": get_git_status()
        },
        "environment": {
            "python_version": get_python_version(),
            "numpy_version": get_numpy_version(),
            "pandas_version": get_pandas_version()
        },
        "datasets": {
            "CuCD-ID": {
                "path": "data/CuCD-ID/noised_dataset.csv",
                "schema": "data/CuCD-ID/label_schema.json",
                "rows": ds_stats.get("CuCD-ID", {}).get("rows"),
                "features": ds_stats.get("CuCD-ID", {}).get("features"),
                "classes": ds_stats.get("CuCD-ID", {}).get("classes"),
                "class_names": ds_stats.get("CuCD-ID", {}).get("class_names", []),
                "provenance": "Fayyaz et al., Data in Brief (2026), doi:10.1016/j.dib.2026.112598; Mendeley Data doi:10.17632/7n2d42pm3n.3",
                "collection": "NASA NOS3/cFS software-in-the-loop digital twin; all telemetry simulated"
            },
            "STIN": {
                "path": "data/STIN/SAT20.csv + TER20.csv",
                "rows": ds_stats.get("STIN", {}).get("rows"),
                "features": ds_stats.get("STIN", {}).get("features"),
                "classes": ds_stats.get("STIN", {}).get("classes"),
                "class_names": ds_stats.get("STIN", {}).get("class_names", []),
                "provenance": "Li et al., IEEE Access (2020), doi:10.1109/ACCESS.2020.3041641; data: github.com/kun9717/STIN-data-set",
                "collection": "Flow-based satellite/terrestrial network testbed (SAT20/TER20)"
            }
        },
        "architectures": {
            "MLP": {"type": "tabular", "h1": 32, "h2": 16, "role": "primary_tabular_backbone"},
            "GRU": {"type": "sequence", "seq_len": "auto", "hidden": 10, "role": "architectural_stress_test"},
            "CNN1D": {"type": "sequence", "c1": 6, "c2": 10, "kernel": 3, "role": "architectural_stress_test"}
        },
        "methods": [
            "finetune", "replay", "ewc", "full", "lwf", "bic", "icarl", "joint", "joint_scratch"
        ],
        "methods_description": {
            "finetune": "Naive sequential fine-tuning (lower bound)",
            "replay": "Experience replay only",
            "ewc": "Online EWC only",
            "full": "Replay + EWC (CMMU)",
            "lwf": "Learning without Forgetting (distillation only)",
            "bic": "Bias Correction (replay + distillation + bias correction)",
            "icarl": "iCaRL-lite (herding + replay + distillation, softmax eval)",
            "joint": "Joint warm-start (legacy, cumulative weights)",
            "joint_scratch": "Joint from-scratch (offline joint-training reference; not a strict upper bound — warm-start joint can exceed it)"
        },
        "seeds": {
            "main_matrix": list(SEEDS),
            "few_shot": list(range(5)),
            "few_shot_protocols": ["fixed", "scaled"],
            "validation_tuning": TUNING_SEEDS,
            "ablations": list(SEEDS),
            "robustness": [0, 1, 2],
            "feature_order": [0, 1]
        },
        "main_matrix": main_counts,
        "run_counts": run_counts,
        "train_val_test_protocol": {
            "val_frac": VAL_FRAC,
            "val_cap_per_class": VAL_CAP_PER_CLASS,
            "calib_frac_of_val": CALIB_FRAC,
            "calib_cap_per_class": CALIB_CAP_PER_CLASS,
            "select_cap_per_class": SELECT_CAP_PER_CLASS,
            "calibration_use": "BiC fitting and val-balanced EWC Fisher",
            "selection_use": "hyperparameter scoring only; excluded from onboard payload",
            "test_frac_cucd_id": 0.30,
            "test_frac_stin": 0.20,
            "test_frac_temporal_fair_cucd_id": 0.20,
            "val_frac_temporal_fair_cucd_id": 0.10,
            "split_mode_random": "random stratified (IID)",
            "split_mode_temporal": "temporal_purged (cold-start earliest-head)",
            "split_mode_temporal_fair": "temporal_purged_fair (random-past train/val, future test)",
            "initial_n_per_class_t0": 700,
            "uplink_n_per_class": 150,
            "note": "Test locked first; val carved from train pool and split into disjoint calibration/selection halves; no test leakage"
        },
        "temporal_purge": {
            "variable": "TimeRadians",
            "purge_value": 0.02,
            "unit": "radians",
            "equivalent_seconds": "~22s",
            "rationale": "Exceeds 20s sliding-window feature horizon + documented 20-22s window-clearing pauses"
        },
        "hyperparameter_tuning": {
            "protocol": "validation-based (no test leakage)",
            "tuning_dataset": "CuCD-ID (primary)",
            "tuning_seeds": TUNING_SEEDS,
            "selection_metric": "ACC_select - max(FM_select, 0)",
            "selection_split": "disjoint selection half (not the calibration half)",
            "frozen_config_file": "results/best_hparams_per_method_arch.json",
            "methods_tuned": list(TUNING_GRIDS.keys()),
            "grids": TUNING_GRIDS
        },
        "quantization": {
            "type": "simulated_post_training_INT8",
            "description": "Per-tensor symmetric INT8 round-trip (FP32->INT8->FP32); NOT actual INT8 hardware inference",
            "corruption_name": "int8_sim"
        },
        "replay_buffer": {
            "type": "reservoir_sampling + class_balanced_option",
            "deployment_quantization": "INT8 features + INT8 labels + INT8 task_ids",
            "training_precision": "FP32"
        },
        "ewc": {
            "type": "online_diagonal_fisher",
            "deployment_precision": "FP16",
            "training_precision": "FP32",
            "gamma": "configurable"
        },
        "evaluation_metrics": [
            "ACC", "BWT", "FM", "MacroF1", "BalancedAccuracy",
            "DET_AUROC", "DET_AUPR", "DET_FPR95",
            "per_class_F1", "per_class_forgetting"
        ],
        "statistical_tests": {
            "test": "exact_paired_permutation",
            "multiple_comparison_correction": "holm",
            "effect_size": "cohens_d",
            "confidence_interval": "95_percent_t (figures); percentile bootstrap (stats CSV)"
        },
        "resource_accounting": {
            "persistent_deployment": {
                "model_weights": "INT8",
                "replay_buffer": "INT8",
                "ewc_state": "FP16",
                "total_KB": "per_config"
            },
            "peak_training_ram": {
                "model_weights": "FP32",
                "gradients": "FP32",
                "adam_states": "FP32 (2x params)",
                "activations": "FP32",
                "batch": "FP32",
                "replay_batch": "FP32",
                "teacher_model": "FP32 (for distillation)",
                "temporary": "FP32",
                "total_KB": "per_config"
            },
            "compute": {
                "macs_per_forward": "per_sample (final-head-width proxy)",
                "macs_per_update": "estimated/conservative: per_sample_proxy * (3 * performed_student_samples + performed_teacher_samples)"
            }
        },
        "outputs": {
            "main_runs": "results/main_results_runs.csv",
            "main_summary": "results/main_results_summary.csv",
            "accuracy_matrices": "results/accuracy_matrices.json",
            "ncm_runs": "results/icarl_ncm_runs.csv",
            "ncm_summary": "results/icarl_ncm_summary.csv",
            "ncm_vs_baselines": "results/icarl_ncm_vs_baselines.csv",
            "ncm_accuracy_matrices": "results/accuracy_matrices_ncm.json",
            "ncm_sanity_runs": "results/ncm_sanity_runs.csv",
            "ncm_sanity_summary": "results/ncm_sanity_summary.csv",
            "ncm_sanity_vs_main": "results/ncm_sanity_vs_main.csv",
            "ncm_sanity_vs_main_runs": "results/ncm_sanity_vs_main_runs.csv",
            "ncm_sanity_per_class": "results/ncm_sanity_per_class.csv",
            "ncm_sanity_accuracy_matrices": "results/accuracy_matrices_ncm_sanity.json",
            "xai_metrics": "results/xai_metrics.csv",
            "xai_summary": "results/xai_summary.csv",
            "xai_examples": "results/xai_examples.csv",
            "xai_feature_importance": "results/xai_feature_importance.csv",
            "xai_feature_summary": "results/xai_feature_summary.csv",
            "xai_metrics_by_correctness": "results/xai_metrics_by_correctness.csv",
            "xai_summary_by_correctness": "results/xai_summary_by_correctness.csv",
            "xai_group_attribution_runs": "results/xai_group_attribution_runs.csv",
            "xai_group_attribution_summary": "results/xai_group_attribution_summary.csv",
            "xai_group_deletion_runs": "results/xai_group_deletion_runs.csv",
            "xai_group_deletion_summary": "results/xai_group_deletion_summary.csv",
            "xai_group_summary": "results/xai_group_summary.csv",
            "xai_baseline_sensitivity": "results/xai_baseline_sensitivity.csv",
            "xai_supplement": "results/XAI_SUPPLEMENT.md",
            "ncm_figures": [
                "figures/ncm_comparison_CuCD-ID.png",
                "figures/ncm_comparison_STIN.png",
                "figures/ncm_paired_delta.png",
                "figures/ncm_sanity_control.png"
            ],
            "xai_figures": [
                "figures/xai_main_metrics.png",
                "figures/xai_feature_groups.png",
                "figures/xai_agreement_CuCD-ID.png",
                "figures/xai_agreement_STIN.png",
                "figures/xai_faithfulness_CuCD-ID.png",
                "figures/xai_faithfulness_STIN.png",
                "figures/xai_feature_importance.png"
            ],
            "temporal_compare": "results/temporal_purged_compare.csv",
            "temporal_figures": [
                "figures/temporal_bracket.png",
                "figures/temporal_mlp_protocols.png",
                "figures/temporal_leakage_audit.png"
            ],
            "buac": [
                "results/buac_summary.csv",
                "results/buac_mlp_gate_eval.csv",
                "results/buac_gru_gate_eval.csv",
                "results/buac_cnn1d_gate_eval.csv",
                "results/buac_mlp_decisions.csv",
                "results/buac_gru_decisions.csv",
                "results/buac_cnn1d_decisions.csv",
                "results/buac_mlp_budget_sensitivity.csv",
                "results/buac_gru_budget_sensitivity.csv",
                "results/buac_cnn1d_budget_sensitivity.csv",
                "figures/buac_gate.png",
            ],
            "fewshot": "results/fewshot.csv",
            "feature_order": "results/feature_order_stability.csv",
            "ablations": [
                "results/ablation_buffer_size.csv",
                "results/ablation_ewc_lambda.csv"
            ],
            "legacy_quarantine": {
                "dir": "results/legacy/",
                "files": [
                    "results/legacy/ablation_task_order.csv",
                    "results/legacy/temporal_compare.csv",
                    "results/legacy/stats_wilcoxon.csv"
                ],
                "note": "superseded outputs retained for provenance only; not used by any figure, table, or claim"
            },
            "per_class": [
                "results/per_class_f1_summary.csv",
                "results/confusion_matrices.json"
            ],
            "robustness": [
                "results/robustness.csv",
                "results/memory_ledger.csv",
                "results/activation_memory_ledger.csv"
            ],
            "baseline_controls": [
                "results/baseline_controls.csv",
                "results/stats_baseline_controls.csv"
            ],
            "stats_acc": "results/stats_permutation.csv",
            "stats_macrof1": "results/stats_permutation_macrof1.csv",
            "leakage_ablation": ["results/leakage_ablation.csv", "results/leakage_ablation_summary.csv", "results/leakage_group_effects.csv"],
            "temporal_validity_audit": [
                "results/temporal_leakage_audit.json",
                "results/temporal_leakage_audit.csv",
                "results/temporal_leakage_audit_summary.csv",
                "results/temporal_leakage_histogram.csv",
                "results/temporal_block_coverage.csv",
                "results/temporal_class_occupancy.csv",
                "results/temporal_class_block_ranges.csv",
                "results/temporal_drift_by_class.csv",
                "results/temporal_drift_features.csv",
                "figures/temporal_leakage_audit.png"
            ],
            "consistency": "results/consistency_check.json",
            "figures_dir": "figures/",
            "report": "results/RESULTS_SUMMARY.md",
            "final_report": "v6_5_project_report_final.md"
        },
        "consistency_checks": _load_consistency(),
        "known_limitations": [
            "Simulation-only (no hardware-in-the-loop)",
            "CuCD-ID is generated in a NOS3/cFS software-in-the-loop digital twin, not captured from an orbiting spacecraft",
            "Random-split (IID) headline numbers are leakage-optimistic: sliding-window aggregates and scenario/time identifiers can cross train/test boundaries; leakage_ablation.csv quantifies feature-group reliance",
            "Random-split headline results are not a deployment guarantee; temporal-purged results are a separate forward-extrapolation estimand",
            "STIN has no benign class (no detection metrics)",
            "GRU/CNN use reshaped tabular features (not genuine temporal)",
            "INT8 quantization is simulated round-trip only",
            "No radiation/orbit simulation",
            "Temporal purge uses TimeRadians proxy (not wall-clock seconds)",
            "Empirical Fisher approximation in EWC",
            "Small per-task sample sizes (uplink budget constraint)",
            "BiC is a lite bias-correction hook (α,β per stage), not the full two-stage BiC training procedure",
            "iCaRL is a lite variant (herding + replay + distillation, softmax rather than nearest-mean evaluation)",
            "The original 90-run NCM comparison changes both herding (input-space to embedding-space) and the deployed classifier (softmax to NCM); it does not isolate the classifier. The 108-run ncm_sanity control separates these factors and shows NCM-all approximately NCM-buffer, so buffer capacity is not the limitation",
            "NCM is a negative/mixed result, not a superiority claim; the clean joint all-data head-only contrast still prefers softmax on CuCD-ID MLP/CNN1D and STIN CNN1D/MLP",
            "XAI attributions use a training-medoid primary background because the zero vector is degenerate for cosine/NCM; a class-balanced mean-of-class-medoids background is a sensitivity check. IG uses 512 steps; KernelSHAP uses 256 sampled coalitions and its efficiency constraint makes completeness exact by construction. Main-text metrics are correct-only Top5Ratio/deletion/insertion; agreement/completeness are diagnostics, not causal proof",
            "The XAI correctness split can leave fewer than three errors per cell; those error-only cells are unstable and excluded from claims",
            "Few-shot calibration is reported under both fixed (150/class) and scaled (K/class) protocols; calibration-using methods include the calibration payload, replay/iCaRL do not",
            "Task-order study swaps class order only; task budgets/boundaries and the canonical split are fixed. Six permutations start with a single-class task, where the first softmax update has exactly zero gradient; these are reported separately as degenerate, with the two-class-T0-first permutations as the primary estimand",
            "Peak RAM is host-estimated and includes batch-32 activations plus retained calibration state for EWC/full/BiC; it is not measured on hardware",
            "Main method comparisons are not equal-memory; see baseline_controls.csv for the matched-replay and EWC λ controls",
            "Statistical comparisons are the full pre-specified family of all method pairs within dataset x architecture (216 per metric), Holm-corrected; they are exploratory, not a winner-selected headline list",
            "CuCD-ID has no scenario/run identifiers; a true scenario-level split is not possible from the released CSV, so the grouped time-block proxy is a sensitivity estimand only",
            "A read-only nearest-time audit shows dense random-split temporal adjacency (most test rows are within the 0.02 rad purge horizon of a drawn training row), but natural time blocks are class-incomplete and within-class feature drift is large; the audit is a validity boundary, not a leakage-free accuracy estimate",
            "The temporal-purged collapse is a joint leakage + mission-time drift + T0-frozen scaler/prior-shift estimand, not a causal leakage correction",
            "The combined `full` method is not a uniform improvement over replay: on CuCD-ID it is below replay for all three architectures, and the `full` tuning grid is smaller than the single-method grids",
            "The lambda=0 control shows that the selected EWC strength can be harmful in some cells (notably CuCD-ID/GRU); this is a tuning/selection limitation, not evidence that the full combination is superior",
            "The feature-order study uses one split and two training seeds; it is exploratory and does not establish permutation invariance",
            "On primary task orders, `fixed_position` and `moves_with_task` are identical by construction (all primary orders start with T0); they are not independent evidence",
            "Validation tuning uses CuCD-ID only and tuning seeds 0-2 overlap the confirmatory seeds; selection is validation-only (no test leakage), but the seeds are not fully independent",
            "STIN is a cross-dataset transfer test for hyperparameters tuned on CuCD-ID; weak STIN results are not evidence that the same tuning is optimal in-domain"
        ]
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    manifest_path = os.path.join(RESULTS_DIR, "final_experiment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()