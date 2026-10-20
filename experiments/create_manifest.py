#!/usr/bin/env python3
"""Create final_experiment_manifest.json for reproducibility."""
import json
import os
import subprocess
import sys
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


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


def main():
    manifest = {
        "project": "cubesat-cl-ids",
        "version": "q2-submission-v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
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
                "rows": 22465,
                "features": 22,
                "classes": 5,
                "class_names": ["normal", "command_flooding", "data_injection", "defence_impairment", "storage_exhaustion"]
            },
            "STIN": {
                "path": "data/STIN/SAT20.csv + TER20.csv",
                "rows": 209564,
                "features": 30,
                "classes": 9,
                "class_names": ["Syn_DDoS", "UDP_DDoS", "LDAP_DDoS", "MSSQL_DDoS", "Botnet", "Portmap_DDoS", "Web Attack", "Backdoor", "NetBIOS_DDoS"]
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
            "main_matrix": list(range(10)),
            "few_shot": list(range(5)),
            "validation_tuning": [0, 1, 2],
            "ablations": list(range(10)),
            "robustness": [0, 1, 2],
            "feature_order": [0, 1]
        },
        "main_matrix": {
            "runs": 540,
            "datasets": 2,
            "architectures": 3,
            "methods": 9,
            "seeds": 10
        },
        "train_val_test_protocol": {
            "val_frac": 0.10,
            "test_frac_cucd_id": 0.30,
            "test_frac_stin": 0.20,
            "split_mode_random": "random stratified (IID)",
            "split_mode_temporal": "temporal_purged",
            "initial_n_per_class_t0": 700,
            "uplink_n_per_class": 150,
            "note": "Test locked first; val carved from train pool; no test leakage"
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
            "tuning_seeds": [0, 1, 2],
            "selection_metric": "ACC_val - FM_val",
            "frozen_config_file": "results/best_hparams_per_method_arch.json",
            "methods_tuned": [
                "finetune", "replay", "ewc", "full", "lwf", "bic", "icarl"
            ],
            "grids": {
                "finetune": {"lr": [1e-3, 3e-3, 1e-2], "epochs": [25, 50]},
                "replay": {"lr": [1e-3, 3e-3, 1e-2], "buffer_capacity": [128, 256, 512], "replay_frac": [0.5, 0.7, 0.9]},
                "ewc": {"lr": [1e-3, 3e-3, 1e-2], "ewc_lambda": [100, 1000, 5000], "ewc_gamma": [0.5, 1.0, 2.0]},
                "full": {"lr": [1e-3, 3e-3, 1e-2], "buffer_capacity": [128, 256, 512], "replay_frac": [0.5, 0.7, 0.9], "ewc_lambda": [100, 1000, 5000]},
                "lwf": {"lr": [1e-3, 3e-3, 1e-2], "distill_lambda": [0.5, 1.0, 2.0], "distill_T": [1.5, 2.0, 3.0]},
                "bic": {"lr": [1e-3, 3e-3, 1e-2], "buffer_capacity": [128, 256, 512], "distill_lambda": [0.5, 1.0, 2.0]},
                "icarl": {"lr": [1e-3, 3e-3, 1e-2], "buffer_capacity": [128, 256, 512], "distill_lambda": [0.5, 1.0, 2.0]}
            }
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
            "confidence_interval": "95_percent_bootstrap"
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
            "temporal_compare": "results/temporal_purged_compare.csv",
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
                "results/memory_ledger.csv"
            ],
            "stats": "results/stats_permutation.csv",
            "figures_dir": "figures/",
            "report": "results/RESULTS_SUMMARY.md"
        },
        "consistency_checks": {
            "main_vs_fewshot_k150": "pass",
            "seeds_main": 10,
            "seeds_fewshot": 5,
            "hyperparameter_source": "results/best_hparams_per_method_arch.json"
        },
        "known_limitations": [
            "Simulation-only (no hardware-in-the-loop)",
            "STIN has no benign class (no detection metrics)",
            "GRU/CNN use reshaped tabular features (not genuine temporal)",
            "INT8 quantization is simulated round-trip only",
            "No radiation/orbit simulation",
            "Temporal purge uses TimeRadians proxy (not wall-clock seconds)",
            "Empirical Fisher approximation in EWC",
            "Small per-task sample sizes (uplink budget constraint)"
        ]
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    manifest_path = os.path.join(RESULTS_DIR, "final_experiment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()