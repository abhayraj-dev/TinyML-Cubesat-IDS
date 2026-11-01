"""SCAD external validation: scenario-disjoint 3-task class-incremental run."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from parallel import configure_single_thread, get_jobs, describe, parallel_map
configure_single_thread()

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import (compute_cl_metrics, compute_macro_f1_balanced_acc,
                     model_persistent_footprint, model_peak_training_ram,
                     estimate_macs_per_sample, estimate_macs_per_update)
from config import (MODEL_CLASSES, ARCHS, METHODS, SEEDS, model_kwargs_for,
                    RESULTS_DIR, BASE_TRAIN_HPARAMS,
                    load_frozen_hparams_strict)

CALIB_METHODS = {"ewc", "full", "bic"}
_FROZEN = None
_DATASETS = {}


def _run_one(item):
    arch, method, seed = item
    ts = _DATASETS[seed]
    model_kwargs = model_kwargs_for(arch, ts.n_features)
    arch_hparams = _FROZEN.get(arch, {}).get(method, {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}

    out = train_continual(
        MODEL_CLASSES[arch], model_kwargs, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    mf_ba = compute_macro_f1_balanced_acc(
        out["model"], ts, out.get("corrector"), ncm=out.get("ncm"))
    persist = model_persistent_footprint(
        out["model"], out["buffer"], out["ewc"], ncm=out.get("ncm"))

    if out["buffer"] is not None:
        n_replay_step = max(
            1, int(hp["batch_size"] * hp.get("replay_frac", 0.0)))
    else:
        n_replay_step = 0

    retained_val_bytes = 0
    if method in CALIB_METHODS:
        for task in ts.tasks:
            if task.X_calib is not None:
                retained_val_bytes += int(task.X_calib.nbytes)
            if task.y_calib is not None:
                retained_val_bytes += int(task.y_calib.nbytes)

    peak = model_peak_training_ram(
        out["model"], out["buffer"], out["ewc"],
        teacher=out.get("teacher"), batch_size=hp["batch_size"],
        n_features=ts.n_features, n_replay=n_replay_step,
        retained_state_bytes=retained_val_bytes, ncm=out.get("ncm"))
    macs_fwd = estimate_macs_per_sample(out["model"], ts.n_features)
    macs_update = int(
        macs_fwd * 3 * int(out.get("train_samples_total", 0))
        + macs_fwd * int(out.get("teacher_samples_total", 0)))

    row = dict(
        dataset="SCAD", arch=arch, method=method, seed=seed,
        n_tasks=len(ts.tasks), **m, **mf_ba,
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


def _parse_list(value, default):
    if not value:
        return list(default)
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    global _FROZEN, _DATASETS
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--residual-mode", choices=["mean", "zscore"],
                        default="mean")
    parser.add_argument("--initial-n", type=int, default=700)
    parser.add_argument("--uplink-n", type=int, default=150)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    archs = _parse_list(args.archs, ARCHS)
    methods = _parse_list(args.methods, METHODS)
    seeds = [int(s) for s in _parse_list(args.seeds, SEEDS)]
    for value in archs:
        assert value in MODEL_CLASSES, f"Unknown architecture {value}"
    for value in methods:
        assert value in METHODS, f"Unknown method {value}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_runs = os.path.join(RESULTS_DIR, f"scad_runs{suffix}.csv")
    out_summary = os.path.join(RESULTS_DIR, f"scad_summary{suffix}.csv")
    out_matrices = os.path.join(RESULTS_DIR, f"scad_accuracy_matrices{suffix}.json")
    out_splits = os.path.join(RESULTS_DIR, f"scad_splits{suffix}.json")

    _FROZEN = load_frozen_hparams_strict()
    print("Using nested-validation frozen hyperparameters")
    data = du.load_scad(
        residual=not args.raw,
        max_scenarios_per_class=args.max_scenarios,
        residual_mode=args.residual_mode,
    )
    _DATASETS = {
        seed: du.build_scad_tasks(
            data, seed=seed,
            initial_n_per_class=args.initial_n,
            uplink_n_per_class=args.uplink_n,
        )
        for seed in seeds
    }
    splits = {str(seed): _DATASETS[seed].scad_split_report for seed in seeds}
    with open(out_splits, "w") as f:
        json.dump(splits, f, indent=2)

    items = [(arch, method, seed)
             for arch in archs for method in methods for seed in seeds]
    jobs = args.jobs or get_jobs()
    print(f"SCAD matrix: {len(items)} runs; {describe(jobs)}")
    print(f"Budgets: initial={args.initial_n}/class, uplink={args.uplink_n}/class")
    if args.raw:
        print("Preprocessing: raw")
    else:
        print(f"Preprocessing: within-scenario {args.residual_mode} residual")
    for task in _DATASETS[seeds[0]].tasks:
        print(f"  {task.name}: train={task.X_train.shape[0]} "
              f"val={task.X_val.shape[0]} test={task.X_test.shape[0]} "
              f"classes={task.class_names}")

    run_rows = []
    matrices = {}
    t_start = time.perf_counter()
    for i, (row, matrix) in enumerate(
            parallel_map(_run_one, items, jobs=jobs), 1):
        run_rows.append(row)
        matrices[f"{row['arch']}|{row['method']}|{row['seed']}"] = matrix
        if i % 5 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['arch']} {row['method']} "
                  f"seed={row['seed']} ACC={row['ACC']:.3f} "
                  f"BWT={row['BWT']:.3f} ({time.perf_counter()-t_start:.1f}s)")

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(out_runs, index=False)
    summary = runs_df.groupby(["arch", "method"]).agg(
        ACC_mean=("ACC", "mean"), ACC_std=("ACC", "std"),
        BWT_mean=("BWT", "mean"), BWT_std=("BWT", "std"),
        FM_mean=("FM", "mean"), FM_std=("FM", "std"),
        MacroF1_mean=("MacroF1", "mean"), MacroF1_std=("MacroF1", "std"),
        BalancedAccuracy_mean=("BalancedAccuracy", "mean"),
        BalancedAccuracy_std=("BalancedAccuracy", "std"),
        train_seconds_per_task_mean=("train_seconds_per_task", "mean"),
        persistent_total_KB_mean=("persistent_total_KB", "mean"),
        peak_total_KB_mean=("peak_total_KB", "mean"),
        macs_per_forward_mean=("macs_per_forward", "mean"),
    ).reset_index().sort_values(["arch", "ACC_mean"], ascending=[True, False])
    summary.to_csv(out_summary, index=False)
    with open(out_matrices, "w") as f:
        json.dump(matrices, f)

    print(f"\nDone. Total time: {time.perf_counter()-t_start:.1f}s")
    print(f"{len(runs_df)} runs written to {out_runs}")
    print("\nSummary (ACC_mean):")
    print(summary.pivot_table(index="arch", columns="method",
                              values="ACC_mean").round(3))


if __name__ == "__main__":
    main()
