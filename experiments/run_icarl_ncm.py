"""iCaRL-lite + NCM evaluation (faithful iCaRL classifier).

This runs the same frozen ``icarl`` training configurations as the main matrix,
but changes the deployed classifier to NCM:

  * exemplars are selected by herding in the model's L2-normalised penultimate
    feature space (not raw input space);
  * class means are built only from the stored exemplars;
  * inference is cosine nearest-class-mean.

The training objective, replay buffer size, distillation and seeds are
unchanged relative to the main iCaRL run, but this evaluation changes BOTH
the herding path (input-space to embedding-space) and the deployed classifier
(softmax to NCM).  It therefore does NOT isolate the classifier change.  Use
``experiments/check_ncm_sanity.py`` for the matched control that separates
input-space herding, embedding-space herding, buffer NCM, all-data NCM and a
joint head-only contrast.  Writes:

  results/icarl_ncm_runs.csv
  results/icarl_ncm_summary.csv
  results/icarl_ncm_vs_baselines.csv
  results/accuracy_matrices_ncm.json
"""
import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from parallel import configure_single_thread, describe, parallel_map, get_jobs
configure_single_thread()

import numpy as np
import pandas as pd

from train import train_continual
from metrics import (compute_cl_metrics, compute_macro_f1_balanced_acc,
                     model_persistent_footprint, model_peak_training_ram,
                     estimate_macs_per_sample)
from config import (MODEL_CLASSES, ARCHS, SEEDS, model_kwargs_for,
                    RESULTS_DIR, BASE_TRAIN_HPARAMS,
                    load_frozen_hparams_strict)
from run_main import build_datasets, final_detection_metrics

DEFAULT_DATASETS = ["CuCD-ID", "STIN"]
DEFAULT_ARCHS = list(ARCHS)
METHOD = "icarl_ncm"
TRAIN_METHOD = "icarl"
BASELINES = ["icarl", "bic", "replay", "full"]

_FROZEN = None
_DATASETS = {}


def load_frozen_hparams():
    return load_frozen_hparams_strict()


def _run_one(item):
    dset_name, arch, seed = item
    ts = _DATASETS[(dset_name, seed)]
    model_kwargs = model_kwargs_for(arch, ts.n_features)
    arch_hp = _FROZEN.get(arch, {}).get(TRAIN_METHOD, {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hp}

    out = train_continual(
        MODEL_CLASSES[arch], model_kwargs, ts, TRAIN_METHOD, hp, seed=seed,
        ncm_eval=True, embedding_herding=True)
    m = compute_cl_metrics(out["R"])
    det = final_detection_metrics(ts, out)
    mf_ba = compute_macro_f1_balanced_acc(
        out["model"], ts, ncm=out.get("ncm"))
    persist = model_persistent_footprint(
        out["model"], out["buffer"], out["ewc"], ncm=out.get("ncm"))

    n_replay_step = max(1, int(hp["batch_size"] * hp.get("replay_frac", 0.0)))
    peak = model_peak_training_ram(
        out["model"], out["buffer"], out["ewc"],
        teacher=out.get("teacher"), batch_size=hp["batch_size"],
        n_features=ts.n_features, n_replay=n_replay_step, ncm=out.get("ncm"))
    macs_fwd = estimate_macs_per_sample(out["model"], ts.n_features)
    macs_update = int(macs_fwd * 3 * int(out.get("train_samples_total", 0))
                      + macs_fwd * int(out.get("teacher_samples_total", 0)))

    row = dict(
        dataset=dset_name, arch=arch, method=METHOD, seed=seed,
        n_tasks=len(ts.tasks), **m, **det, **mf_ba,
        train_seconds_total=float(sum(out["per_task_train_seconds"])),
        train_seconds_per_task=float(np.mean(out["per_task_train_seconds"])),
        **persist,
        peak_total_KB=peak["peak_total_KB"],
        peak_adam_KB=peak["peak_adam_bytes_fp32"] / 1024.0,
        macs_per_forward=macs_fwd,
        macs_per_update=macs_update,
        ncm_class_means_bytes=int(out["ncm"].memory_bytes(bytes_per_val=2)),
    )
    return row, out["R"].tolist()


def _summary(df):
    agg = df.groupby(["dataset", "arch", "method"]).agg(
        ACC_mean=("ACC", "mean"), ACC_std=("ACC", "std"),
        BWT_mean=("BWT", "mean"), BWT_std=("BWT", "std"),
        FM_mean=("FM", "mean"), FM_std=("FM", "std"),
        MacroF1_mean=("MacroF1", "mean"), MacroF1_std=("MacroF1", "std"),
        BalancedAccuracy_mean=("BalancedAccuracy", "mean"),
        BalancedAccuracy_std=("BalancedAccuracy", "std"),
        DET_AUROC_mean=("DET_AUROC", "mean"),
        DET_AUPR_mean=("DET_AUPR", "mean"),
        DET_FPR95_mean=("DET_FPR95", "mean"),
        train_seconds_per_task_mean=("train_seconds_per_task", "mean"),
        persistent_total_KB_mean=("persistent_total_KB", "mean"),
        peak_total_KB_mean=("peak_total_KB", "mean"),
        ncm_class_means_bytes_mean=("ncm_class_means_bytes", "mean"),
    ).reset_index()
    return agg


def _paired_compare(ncm_df, base_df):
    base = base_df[base_df.method.isin(BASELINES)].copy()
    rows = []
    for baseline in BASELINES:
        b = base[base.method == baseline][["dataset", "arch", "seed", "ACC",
                                           "MacroF1", "BWT", "FM"]]
        if b.empty:
            continue
        n = ncm_df[["dataset", "arch", "seed", "ACC", "MacroF1", "BWT", "FM"]]
        m = n.merge(b, on=["dataset", "arch", "seed"], suffixes=("_ncm", "_base"))
        for (dset, arch), g in m.groupby(["dataset", "arch"]):
            d = g["ACC_ncm"] - g["ACC_base"]
            rows.append(dict(
                dataset=dset, arch=arch, baseline=baseline, n=len(g),
                ACC_ncm=float(g["ACC_ncm"].mean()),
                ACC_base=float(g["ACC_base"].mean()),
                ACC_delta_mean=float(d.mean()),
                ACC_delta_std=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                wins=int((d > 0).sum()), ties=int((d == 0).sum()),
                MacroF1_delta_mean=float((g["MacroF1_ncm"] - g["MacroF1_base"]).mean()),
                BWT_delta_mean=float((g["BWT_ncm"] - g["BWT_base"]).mean()),
                FM_delta_mean=float((g["FM_ncm"] - g["FM_base"]).mean()),
            ))
    return pd.DataFrame(rows)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.datasets = ["CuCD-ID"]
        args.archs = ["MLP"]
        args.seeds = [0]
    return args


def main():
    global _FROZEN, _DATASETS
    args = _parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams()
    _DATASETS = build_datasets()

    items = [(dset, arch, seed)
             for dset in args.datasets for arch in args.archs
             for seed in args.seeds]
    jobs = get_jobs()
    print(f"iCaRL+NCM: {len(items)} runs; {describe(jobs)}")

    rows, matrices = [], {}
    t0 = time.perf_counter()
    for i, (row, matrix) in enumerate(
            parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        matrices[f"{row['dataset']}|{row['arch']}|{row['seed']}"] = matrix
        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['dataset']} {row['arch']} "
                  f"seed={row['seed']} ACC={row['ACC']:.3f} "
                  f"BWT={row['BWT']:.3f} ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    runs_path = os.path.join(RESULTS_DIR, "icarl_ncm_runs.csv")
    df.to_csv(runs_path, index=False)
    summary = _summary(df)
    summary.to_csv(os.path.join(RESULTS_DIR, "icarl_ncm_summary.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "accuracy_matrices_ncm.json"), "w") as f:
        json.dump(matrices, f)

    base_path = os.path.join(RESULTS_DIR, "main_results_runs.csv")
    if os.path.exists(base_path):
        base_df = pd.read_csv(base_path)
        compare = _paired_compare(df, base_df)
        compare.to_csv(os.path.join(RESULTS_DIR,
                                    "icarl_ncm_vs_baselines.csv"), index=False)
        print("\nPaired ACC differences (NCM - baseline), mean by architecture:")
        print(compare.pivot_table(index=["dataset", "arch"],
                                  columns="baseline",
                                  values="ACC_delta_mean").round(4).to_string())

    print("\nNCM summary (ACC mean):")
    print(summary.pivot_table(index=["dataset", "arch"], columns="method",
                              values="ACC_mean").round(3).to_string())
    print(f"\nWrote {runs_path} ({len(df)} runs, "
          f"{time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
