"""Phase 8: Feature-order permutation stability test (NEW).

Tests whether MLP (tabular) is stable to feature permutation while
GRU/CNN (sequence-aware) are sensitive -- confirming MLP as the
primary valid backbone for CuCD-ID tabular telemetry.

3-4 random feature permutations x 3 archs x 4 methods x 2 seeds on CuCD-ID.

Output: results/feature_order_stability.csv
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics, compute_macro_f1_balanced_acc
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, ARCHS)
from run_main import final_detection_metrics


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


def permute_features(X, perm_idx):
    """Apply feature permutation to data."""
    return X[:, perm_idx]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frozen_hparams = load_frozen_hparams()

    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json")
    )

    # Base task sequence (unpermuted) to get feature count
    ts_base = du.build_cucd_id_tasks(cucd_df, seed=0, val_frac=0.10)
    n_features = ts_base.n_features

    # Generate 3-4 random feature permutations
    rng = np.random.default_rng(42)
    n_perms = 4
    perms = [np.arange(n_features)]  # Original order
    for _ in range(n_perms - 1):
        perm = rng.permutation(n_features)
        perms.append(perm)

    METHODS = ["replay", "full", "icarl", "bic"]
    SEEDS = [0, 1]  # 2 seeds per config

    rows = []
    t_start = time.perf_counter()
    total = len(ARCHS) * len(METHODS) * len(perms) * len(SEEDS)
    done = 0

    for arch in ARCHS:
        for perm_idx, perm in enumerate(perms):
            # Build task sequence with permuted features
            # We need to apply permutation at data loading time
            # For simplicity, permute after building (since builder uses same seed)
            ts = du.build_cucd_id_tasks(cucd_df, seed=0, val_frac=0.10)

            # Apply permutation to all task data
            for task in ts.tasks:
                task.X_train = task.X_train[:, perm]
                task.X_test = task.X_test[:, perm]
                if task.has_val:
                    task.X_val = task.X_val[:, perm]

            for method in METHODS:
                arch_hparams = frozen_hparams.get(arch, {}).get(method, {}).get("hparams", {})

                for seed in SEEDS:
                    mk = model_kwargs_for(arch, ts.n_features)
                    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
                    out = train_continual(MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed)
                    m = compute_cl_metrics(out["R"])
                    mf_ba = compute_macro_f1_balanced_acc(out["model"], ts)
                    det = final_detection_metrics(ts, out)

                    rows.append(dict(
                        arch=arch,
                        perm_idx=perm_idx,
                        method=method,
                        seed=seed,
                        is_original=(perm_idx == 0),
                        **m, **mf_ba,
                        DET_AUROC=det["DET_AUROC"]
                    ))

                    done += 1
                    if done % 24 == 0:
                        print(f"  [{done}/{total}] {arch} perm={perm_idx} {method} "
                              f"seed={seed} ACC={m['ACC']:.3f} ({time.perf_counter()-t_start:.1f}s)")

    df = pd.DataFrame(rows)
    output_path = os.path.join(RESULTS_DIR, "feature_order_stability.csv")
    df.to_csv(output_path, index=False)

    # Summary: compare original vs permuted stability
    print("\n=== Feature Order Stability Summary ===")
    for arch in ARCHS:
        arch_df = df[df.arch == arch]
        orig = arch_df[arch_df.is_original].groupby("method")["ACC"].mean()
        permd = arch_df[~arch_df.is_original].groupby("method")["ACC"].mean()
        print(f"\n{arch}:")
        for m in METHODS:
            if m in orig.index and m in permd.index:
                drop = orig[m] - permd[m]
                print(f"  {m:10s}: orig={orig[m]:.3f} perm={permd[m]:.3f} drop={drop:+.3f}")

    print(f"\nWrote {output_path} ({len(df)} runs, {time.perf_counter()-t_start:.1f}s)")


if __name__ == "__main__":
    main()