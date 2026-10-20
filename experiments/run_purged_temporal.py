"""Phase 3: Purged temporal evaluation (NEW).

Compares IID (random) vs Forward Temporal + 22s purge gap on CuCD-ID.
The purge gap uses TimeRadians (0.02 rad) to exceed the 20-second sliding-window
feature horizon and the documented 20-22s window-clearing pauses.

Dual-protocol evaluation:
- Protocol A (IID): Random stratified split, benchmark/comparability
- Protocol B (Forward Temporal + Purge): Generalization under temporal shift

Output: results/temporal_purged_compare.csv
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics, compute_detection_metrics
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS)

# Best hyperparameters will be loaded from validation tuning
METHODS = ["finetune", "replay", "ewc", "full", "lwf", "bic", "icarl", "joint", "joint_scratch"]
SEEDS = list(range(10))  # 10 seeds for main comparison
VAL_FRAC = 0.10
PURGE_TIME_RAD = 0.02  # radians on TimeRadians (~22s equiv., exceeds 20s window + clearing pauses)


def load_best_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


def final_detection_metrics(ts, out):
    """CuCD-ID only: 1-p(normal) detector on concatenated all-task test pool."""
    if ts.dataset_name != "CuCD-ID":
        return {"DET_AUROC": float("nan"), "DET_AUPR": float("nan"),
                "DET_FPR95": float("nan"), "DET_PREV": float("nan")}
    try:
        normal_idx = list(ts.class_names_global).index("normal")
    except ValueError:
        return {"DET_AUROC": float("nan"), "DET_AUPR": float("nan"),
                "DET_FPR95": float("nan"), "DET_PREV": float("nan")}
    Xs, ys, off = [], [], 0
    for task in ts.tasks:
        Xs.append(task.X_test)
        ys.append(task.y_test + off)
        off += task.n_new_classes
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    logits, _ = out["model"].forward(X_all)
    corr = out.get("corrector")
    if corr is not None and getattr(corr, "fitted", False):
        logits = corr.apply(logits)
    return compute_detection_metrics(y_all, logits, normal_idx)


def main():
    import json
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_hparams_all = load_best_hparams()

    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json")
    )

    rows = []
    t_start = time.perf_counter()
    total = len(METHODS) * 2 * len(SEEDS)  # 2 protocols x 10 seeds
    done = 0

    for method in METHODS:
        for protocol in ["iid", "temporal_purged"]:
            for seed in SEEDS:
                if protocol == "iid":
                    ts = du.build_cucd_id_tasks(cucd_df, seed=seed, val_frac=VAL_FRAC)
                else:
                    ts = du.build_cucd_id_tasks_purged_temporal(
                        cucd_df, seed=seed, val_frac=VAL_FRAC, purge_time_rad=PURGE_TIME_RAD)

                mk = model_kwargs_for("MLP", ts.n_features)

                # Get method/arch specific hparams
                arch_hparams = best_hparams_all.get("MLP", {}).get(method, {}).get("hparams", {})
                hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}

                out = train_continual(MODEL_CLASSES["MLP"], mk, ts, method, hp, seed=seed)
                m = compute_cl_metrics(out["R"])
                det = final_detection_metrics(ts, out)

                rows.append(dict(method=method, protocol=protocol, seed=seed,
                                 purge_time_rad=PURGE_TIME_RAD if protocol == "temporal_purged" else 0.0,
                                 **m, **det))

                done += 1
                if done % 20 == 0:
                    print(f"  [{done}/{total}] {method} {protocol} seed={seed} "
                          f"ACC={m['ACC']:.3f} ({time.perf_counter()-t_start:.1f}s)")

    df = pd.DataFrame(rows)
    output_path = os.path.join(RESULTS_DIR, "temporal_purged_compare.csv")
    df.to_csv(output_path, index=False)

    print("\nMean ACC by (method, protocol):")
    print(df.groupby(["method", "protocol"])["ACC"].agg(["mean", "std"]).round(3).to_string())
    print(f"\nWrote {output_path} ({len(df)} runs, {time.perf_counter()-t_start:.1f}s)")


if __name__ == "__main__":
    main()