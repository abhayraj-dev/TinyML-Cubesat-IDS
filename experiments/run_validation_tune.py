"""Phase 2: Per-method/architecture validation-based hyperparameter tuning.

This script tunes hyperparameters for each method/architecture combination
using VALIDATION data only (no test leakage). The winning configurations
are frozen and used for all subsequent experiments.

Output: results/best_hparams_per_method_arch.json
"""
import sys
import os
import json
import itertools
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics
from config import (
    MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
    BASE_TRAIN_HPARAMS, ARCHS
)

# Per-method tuning grids.  The 25/50-epoch choice is tuned for every method
# (CNN optimization was previously hand-set, not selected).  `full` uses a
# reduced multi-axis grid so the added gamma/epochs dimensions stay tractable.
TUNING_GRIDS = {
    "finetune": {
        "lr": [1e-3, 3e-3, 1e-2],
        "epochs": [25, 50],
    },
    "replay": {
        "lr": [1e-3, 3e-3, 1e-2],
        "buffer_capacity": [128, 512],
        "replay_frac": [0.5, 0.9],
        "epochs": [25, 50],
    },
    "ewc": {
        "lr": [1e-3, 3e-3, 1e-2],
        "ewc_lambda": [100, 1000, 5000],
        "ewc_gamma": [0.5, 1.0, 2.0],
        "epochs": [25, 50],
    },
    "full": {
        "lr": [3e-3, 1e-2],
        "buffer_capacity": [256, 512],
        "replay_frac": [0.7],
        "ewc_lambda": [100, 1000],
        "ewc_gamma": [0.5, 1.0, 2.0],
        "epochs": [25, 50],
    },
    "lwf": {
        "lr": [1e-3, 3e-3, 1e-2],
        "distill_lambda": [0.5, 1.0, 2.0],
        "distill_T": [1.5, 3.0],
        "epochs": [25, 50],
    },
    "bic": {
        "lr": [1e-3, 3e-3, 1e-2],
        "buffer_capacity": [128, 512],
        "distill_lambda": [0.5, 1.0, 2.0],
        "epochs": [25, 50],
    },
    "icarl": {
        "lr": [1e-3, 3e-3, 1e-2],
        "buffer_capacity": [128, 512],
        "distill_lambda": [0.5, 1.0, 2.0],
        "epochs": [25, 50],
    },
    "joint": {
        "lr": [1e-3, 3e-3, 1e-2],
        "epochs": [25, 50],
    },
    "joint_scratch": {
        "lr": [1e-3, 3e-3, 1e-2],
        "epochs": [25, 50],
    },
}

TUNING_SEEDS = [0, 1, 2]  # 3 seeds for tuning
VAL_FRAC = 0.10


def generate_combos(grid):
    """Generate all hyperparameter combinations from a grid dict."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def tune_method_arch(method, arch, df, builder_fn, all_rows):
    """Tune hyperparameters for a single method/architecture on CuCD-ID.

    Task sequences depend only on the seed, so they are built once per seed
    and reused across all hyperparameter combinations. Every combination is
    trained exactly once (earlier code re-ran the best grid for logging).
    """
    print(f"  Tuning {method}/{arch}...")
    ts_by_seed = {
        seed: builder_fn(df, seed=seed, val_frac=VAL_FRAC)
        for seed in TUNING_SEEDS
    }
    mk_by_seed = {
        seed: model_kwargs_for(arch, ts.n_features)
        for seed, ts in ts_by_seed.items()
    }

    grid = TUNING_GRIDS[method]
    best_score = -1.0
    best_hp = None
    best_metrics = None

    for hp_combo in generate_combos(grid):
        full_hp = {**BASE_TRAIN_HPARAMS, **hp_combo}
        scores = []
        seed_metrics = []

        for seed in TUNING_SEEDS:
            ts_seed = ts_by_seed[seed]
            out = train_continual(MODEL_CLASSES[arch], mk_by_seed[seed], ts_seed,
                                  method, full_hp, seed=seed)

            if out["V"] is not None and not np.all(np.isnan(out["V"])):
                m = compute_cl_metrics(out["V"])
            else:
                m = compute_cl_metrics(out["R"])

            score = m["ACC"] - max(m["FM"], 0.0)
            scores.append(score)
            seed_metrics.append(m)

        mean_score = float(np.mean(scores))
        all_rows.append({
            "arch": arch,
            "method": method,
            **hp_combo,
            "val_score_mean": mean_score,
            "val_score_std": float(np.std(scores)),
        })
        if mean_score > best_score:
            best_score = mean_score
            best_hp = hp_combo.copy()
            best_metrics = {
                k: float(np.mean([sm[k] for sm in seed_metrics]))
                for k in seed_metrics[0]
            }

    return best_hp, best_score, best_metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load CuCD-ID dataset
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json")
    )

    methods = list(TUNING_GRIDS.keys())
    results = {}
    all_rows = []

    t_start = time.perf_counter()
    total_combos = sum(len(list(generate_combos(TUNING_GRIDS[m]))) for m in methods)
    total_combos *= len(ARCHS) * len(TUNING_SEEDS)
    print(f"Total tuning runs: ~{total_combos} (across {len(methods)} methods x {len(ARCHS)} archs x {len(TUNING_SEEDS)} seeds)")

    for arch in ARCHS:
        print(f"\n=== Architecture: {arch} ===")
        results[arch] = {}

        for method in methods:
            best_hp, best_score, best_metrics = tune_method_arch(
                method, arch, cucd_df, du.build_cucd_id_tasks, all_rows)
            results[arch][method] = {
                "hparams": best_hp,
                "val_score": best_score,
                "val_metrics_mean": best_metrics
            }

            print(f"    {method}: best_val_score={best_score:.4f}, hparams={best_hp}")

    # Save results
    output_path = os.path.join(RESULTS_DIR, "best_hparams_per_method_arch.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save detailed CSV
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(os.path.join(RESULTS_DIR, "validation_tuning_results.csv"), index=False)

    print(f"\n=== Tuning Complete ({time.perf_counter()-t_start:.1f}s) ===")
    print(f"Best configs saved to: {output_path}")
    print("\nSummary of best validation scores:")
    for arch in ARCHS:
        print(f"\n  {arch}:")
        for method in methods:
            r = results[arch][method]
            print(f"    {method:10s}: val_score={r['val_score']:.4f}, hparams={r['hparams']}")


if __name__ == "__main__":
    main()