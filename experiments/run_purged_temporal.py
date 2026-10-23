"""Phase 3: Purged temporal evaluation (confirmatory protocol).

Compares IID (random) vs Forward Temporal + 22s purge gap on CuCD-ID.
The purge gap uses TimeRadians (0.02 rad) to exceed the 20-second sliding-window
feature horizon and the documented 20-22s window-clearing pauses.

Three-protocol evaluation:
- Protocol A (IID): Random stratified split, benchmark/comparability
- Protocol B (Forward Temporal + Purge): Generalization under temporal shift
- Protocol C (Forward Temporal + Purge + no time/ID/window inputs)

Output: results/temporal_purged_compare.csv
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from parallel import configure_single_thread, describe, parallel_map, get_jobs
configure_single_thread()

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics, compute_detection_metrics
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, VAL_FRAC, load_frozen_hparams_strict)

METHODS = ["finetune", "replay", "ewc", "full", "lwf", "bic", "icarl", "joint", "joint_scratch"]
SEEDS = list(range(10))
PURGE_TIME_RAD = 0.02
LEAKY_FEATURES = [
    "TimeRadians", "MsgId", "CmdCode", "ApId",
    "FlowLengthInWindow", "SlidingWindowMeanIntervalSec",
    "SlidingWindowMaxIntervalSec", "SlidingWindowMinIntervalSec",
    "MessageCountInWindow", "UniqueMessageIDsInWindow",
    "AverageMessageLengthInWindow", "MessageRateInWindow",
    "StdDevMessageLengthInWindow",
]
PROTOCOLS = [
    ("iid", None),
    ("temporal_purged", None),
    ("temporal_purged_no_leaky_features", LEAKY_FEATURES),
]
_FROZEN = None
_TS = {}


def final_detection_metrics(ts, out):
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


def _run_one(item):
    method, protocol, seed = item
    ts = _TS[(protocol, seed)]
    mk = model_kwargs_for("MLP", ts.n_features)
    arch_hparams = _FROZEN.get("MLP", {}).get(method, {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
    out = train_continual(MODEL_CLASSES["MLP"], mk, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    det = final_detection_metrics(ts, out)
    return dict(
        method=method, protocol=protocol, seed=seed,
        n_features=ts.n_features,
        purge_time_rad=PURGE_TIME_RAD if protocol != "iid" else 0.0,
        **m, **det)


def main():
    global _FROZEN, _TS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams_strict()
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))

    for protocol, drop_cols in PROTOCOLS:
        for seed in SEEDS:
            if protocol == "iid":
                ts = du.build_cucd_id_tasks(cucd_df, seed=seed, val_frac=VAL_FRAC)
            else:
                ts = du.build_cucd_id_tasks_purged_temporal(
                    cucd_df, seed=seed, val_frac=VAL_FRAC,
                    purge_time_rad=PURGE_TIME_RAD, feature_drop=drop_cols)
            _TS[(protocol, seed)] = ts

    items = [(method, protocol, seed)
             for method in METHODS
             for protocol, _ in PROTOCOLS
             for seed in SEEDS]
    jobs = get_jobs()
    print(f"Purged temporal: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 20 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['method']} {row['protocol']} "
                  f"seed={row['seed']} ACC={row['ACC']:.3f} "
                  f"({time.perf_counter()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    output_path = os.path.join(RESULTS_DIR, "temporal_purged_compare.csv")
    df.to_csv(output_path, index=False)
    print("\nMean ACC by (method, protocol):")
    print(df.groupby(["method", "protocol"])["ACC"]
          .agg(["mean", "std"]).round(3).to_string())
    print(f"\nWrote {output_path} ({len(df)} runs, "
          f"{time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
