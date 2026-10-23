"""Main experiment matrix (Phase 1), confirmatory protocol.

2 datasets x 3 architectures x 9 methods x 15 seeds = 810 runs.
Hyperparameters are frozen from nested validation tuning; the validation pool
is split into disjoint calibration (BiC/EWC fitting) and selection
(hyperparameter scoring) halves.  Test data is never used for selection.

Writes:
  results/main_results_runs.csv
  results/main_results_summary.csv
  results/accuracy_matrices.json
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from parallel import configure_single_thread, describe, parallel_map, get_jobs
configure_single_thread()

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import (compute_cl_metrics, compute_detection_metrics,
                     compute_macro_f1_balanced_acc,
                     model_persistent_footprint, model_peak_training_ram,
                     estimate_macs_per_sample, estimate_macs_per_update)
from config import (MODEL_CLASSES, ARCHS, METHODS, SEEDS, model_kwargs_for,
                    DATA_DIR, RESULTS_DIR, BASE_TRAIN_HPARAMS,
                    load_frozen_hparams_strict, VAL_FRAC, VAL_CAP_PER_CLASS)

DATASETS = ["CuCD-ID", "STIN"]
CALIB_METHODS = {"ewc", "full", "bic"}
_FROZEN = None
_DATASETS = {}


def final_detection_metrics(ts, out):
    """CuCD-ID only: 1-p(normal) detector on the final all-task test pool."""
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


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    return load_frozen_hparams_strict()


def build_datasets():
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    stin_df = du.load_stin(
        os.path.join(DATA_DIR, "STIN", "SAT20.csv"),
        os.path.join(DATA_DIR, "STIN", "TER20.csv"))
    datasets = {}
    for seed in SEEDS:
        datasets[("CuCD-ID", seed)] = du.build_cucd_id_tasks(
            cucd_df, seed=seed, val_frac=VAL_FRAC,
            val_cap_per_class=VAL_CAP_PER_CLASS)
        datasets[("STIN", seed)] = du.build_stin_tasks(
            stin_df, seed=seed, val_frac=VAL_FRAC,
            val_cap_per_class=VAL_CAP_PER_CLASS)
    return datasets


def _run_one(item):
    dset_name, arch, method, seed = item
    ts = _DATASETS[(dset_name, seed)]
    model_kwargs = model_kwargs_for(arch, ts.n_features)
    arch_hparams = _FROZEN.get(arch, {}).get(method, {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}

    out = train_continual(
        MODEL_CLASSES[arch], model_kwargs, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    det = final_detection_metrics(ts, out)
    mf_ba = compute_macro_f1_balanced_acc(
        out["model"], ts, out.get("corrector"))
    persist = model_persistent_footprint(
        out["model"], out["buffer"], out["ewc"])

    if out["buffer"] is not None:
        n_replay_step = max(
            1, int(hp["batch_size"] * hp.get("replay_frac", 0.0)))
    else:
        n_replay_step = 0

    # Onboard update state: calibration halves are retained across task
    # boundaries for methods that fit BiC / EWC Fisher.  The selection half is
    # an experimental scoring artifact and is deliberately excluded.
    retained_val_bytes = 0
    if method in CALIB_METHODS:
        for _task in ts.tasks:
            if _task.X_calib is not None:
                retained_val_bytes += int(_task.X_calib.nbytes)
            if _task.y_calib is not None:
                retained_val_bytes += int(_task.y_calib.nbytes)

    peak = model_peak_training_ram(
        out["model"], out["buffer"], out["ewc"],
        teacher=out.get("teacher"), batch_size=hp["batch_size"],
        n_features=ts.n_features, n_replay=n_replay_step,
        retained_state_bytes=retained_val_bytes)

    macs_fwd = estimate_macs_per_sample(out["model"], ts.n_features)
    macs_update = int(
        macs_fwd * 3 * int(out.get("train_samples_total", 0))
        + macs_fwd * int(out.get("teacher_samples_total", 0)))

    row = dict(
        dataset=dset_name, arch=arch, method=method, seed=seed,
        n_tasks=len(ts.tasks), **m, **det, **mf_ba,
        train_seconds_total=float(sum(out["per_task_train_seconds"])),
        train_seconds_per_task=float(np.mean(out["per_task_train_seconds"])),
        **persist,
        peak_total_KB=peak["peak_total_KB"],
        peak_adam_KB=peak["peak_adam_bytes_fp32"] / 1024.0,
        retained_val_KB=retained_val_bytes / 1024.0,
        macs_per_forward=macs_fwd,
        macs_per_update=macs_update,
    )
    return row, out["R"].tolist()


def main():
    global _FROZEN, _DATASETS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams()
    _DATASETS = build_datasets()
    print("Using nested-validation frozen hyperparameters")
    print(f"Validation protocol: {VAL_FRAC:.0%} total, split 50/50 into "
          "disjoint calibration and selection halves")

    items = [(dset, arch, method, seed)
             for dset in DATASETS
             for arch in ARCHS
             for method in METHODS
             for seed in SEEDS]
    jobs = get_jobs()
    print(f"Main matrix: {len(items)} runs; {describe(jobs)}")

    run_rows = []
    matrices = {}
    t_start = time.perf_counter()
    for i, (row, matrix) in enumerate(
            parallel_map(_run_one, items, jobs=jobs), 1):
        run_rows.append(row)
        matrices[f"{row['dataset']}|{row['arch']}|{row['method']}|{row['seed']}"] = matrix
        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['dataset']} {row['arch']} "
                  f"{row['method']} seed={row['seed']} ACC={row['ACC']:.3f} "
                  f"BWT={row['BWT']:.3f} ({time.perf_counter()-t_start:.1f}s)")

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(os.path.join(RESULTS_DIR, "main_results_runs.csv"), index=False)

    summary = runs_df.groupby(["dataset", "arch", "method"]).agg(
        ACC_mean=("ACC", "mean"), ACC_std=("ACC", "std"),
        BWT_mean=("BWT", "mean"), BWT_std=("BWT", "std"),
        FM_mean=("FM", "mean"), FM_std=("FM", "std"),
        MacroF1_mean=("MacroF1", "mean"), MacroF1_std=("MacroF1", "std"),
        BalancedAccuracy_mean=("BalancedAccuracy", "mean"),
        BalancedAccuracy_std=("BalancedAccuracy", "std"),
        DET_AUROC_mean=("DET_AUROC", "mean"), DET_AUROC_std=("DET_AUROC", "std"),
        DET_AUPR_mean=("DET_AUPR", "mean"),
        DET_FPR95_mean=("DET_FPR95", "mean"),
        train_seconds_per_task_mean=("train_seconds_per_task", "mean"),
        persistent_total_KB_mean=("persistent_total_KB", "mean"),
        persistent_model_bytes_mean=("persistent_model_bytes_int8", "mean"),
        persistent_cmmu_bytes_mean=("persistent_cmmu_bytes", "mean"),
        peak_total_KB_mean=("peak_total_KB", "mean"),
        peak_adam_KB_mean=("peak_adam_KB", "mean"),
        retained_val_KB_mean=("retained_val_KB", "mean"),
        macs_per_forward_mean=("macs_per_forward", "mean"),
        macs_per_update_mean=("macs_per_update", "mean"),
    ).reset_index()
    summary["persistent_model_KB_mean"] = summary["persistent_model_bytes_mean"] / 1024.0
    summary["persistent_cmmu_KB_mean"] = summary["persistent_cmmu_bytes_mean"] / 1024.0
    summary = summary.drop(columns=["persistent_model_bytes_mean",
                                    "persistent_cmmu_bytes_mean"])
    summary.to_csv(os.path.join(RESULTS_DIR, "main_results_summary.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "accuracy_matrices.json"), "w") as f:
        json.dump(matrices, f)

    print(f"\nDone. Total time: {time.perf_counter()-t_start:.1f}s")
    print(f"{len(runs_df)} runs written to results/main_results_runs.csv")
    print("\nSummary (ACC_mean):")
    print(summary.pivot_table(index=["dataset", "arch"], columns="method",
                              values="ACC_mean").round(3))


if __name__ == "__main__":
    main()
