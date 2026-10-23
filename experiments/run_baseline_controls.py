"""Priority-2 matched-memory and EWC-lambda controls.

Runs, on the same locked splits/seeds as the main matrix:
  - replay_matched: replay with the `full` buffer capacity (512) and replay
    fraction (0.7), so replay-vs-full is an equal-replay-budget comparison.
  - ewc_l0 / ewc_l10: EWC with lambda 0 and 10, the values the ablation grid
    already selected for several dataset/arch combinations.
  - full_l0 / full_l10: the combined method with EWC switched off or weak.

Writes:
  results/baseline_controls.csv
  results/stats_baseline_controls.csv
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from parallel import configure_single_thread, describe, parallel_map, get_jobs
configure_single_thread()

import numpy as np
import pandas as pd

import data_utils as du
from config import (MODEL_CLASSES, ARCHS, SEEDS, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, VAL_FRAC, model_kwargs_for,
                    load_frozen_hparams_strict)
from train import train_continual
from metrics import compute_cl_metrics
from run_stats import paired_permutation_test, holm_correction


def _build_controls(frozen):
    controls = []
    for arch in ARCHS:
        replay_hp = frozen[arch]["replay"]["hparams"]
        full_hp = frozen[arch]["full"]["hparams"]
        ewc_hp = frozen[arch]["ewc"]["hparams"]
        controls.append(("replay_matched", "replay", arch, {
            **replay_hp,
            "buffer_capacity": int(full_hp["buffer_capacity"]),
            "replay_frac": float(full_hp["replay_frac"]),
        }))
        for lam in (0.0, 10.0):
            controls.append((f"ewc_l{int(lam)}", "ewc", arch, {
                **ewc_hp, "ewc_lambda": lam,
            }))
            controls.append((f"full_l{int(lam)}", "full", arch, {
                **full_hp, "ewc_lambda": lam,
            }))
    return controls


def compute_stats(df):
    main_path = os.path.join(RESULTS_DIR, "main_results_runs.csv")
    if not os.path.exists(main_path):
        print("WARNING: main_results_runs.csv missing; skipping paired stats.")
        return
    main_df = pd.read_csv(main_path)
    refs = {
        "replay_matched": "replay",
        "ewc_l0": "ewc", "ewc_l10": "ewc",
        "full_l0": "full", "full_l10": "full",
    }
    stat_rows = []
    for metric in ("ACC",):
        metric_rows = []
        for condition, ref_method in refs.items():
            for (dset_name, arch), group in df.groupby(["dataset", "arch"]):
                cond = group[group.condition == condition]
                ref = main_df[(main_df.dataset == dset_name) &
                              (main_df.arch == arch) &
                              (main_df.method == ref_method)]
                merged = cond.merge(ref, on=["dataset", "arch", "seed"],
                                    suffixes=("_cond", "_ref"))
                if len(merged) < 2:
                    continue
                res = paired_permutation_test(
                    merged[f"{metric}_cond"].to_numpy(),
                    merged[f"{metric}_ref"].to_numpy())
                metric_rows.append({
                    "metric": metric,
                    "comparison": f"{condition} vs {ref_method}",
                    "condition": condition, "reference": ref_method,
                    "dataset": dset_name, "arch": arch,
                    "n_pairs": int(len(merged)),
                    "mean_cond": float(merged[f"{metric}_cond"].mean()),
                    "mean_ref": float(merged[f"{metric}_ref"].mean()),
                    **res,
                })
        if metric_rows:
            p_adj = holm_correction([r["p"] for r in metric_rows])
            for r, pa in zip(metric_rows, p_adj):
                r["p_holm"] = float(pa)
                stat_rows.extend([r])
    stats = pd.DataFrame(stat_rows)
    stat_path = os.path.join(RESULTS_DIR, "stats_baseline_controls.csv")
    stats.to_csv(stat_path, index=False)
    print(f"Wrote {stat_path} ({len(stats)} comparisons)")
    if not stats.empty:
        print(stats[["metric", "comparison", "dataset", "arch",
                     "mean_diff", "p", "p_holm"]].to_string(index=False))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-only", action="store_true",
                    help="Do not train; recompute paired stats from existing CSV.")
    return ap.parse_args()


_FROZEN = None
_DATASETS = {}
_CONTROLS = []


def _run_one(item):
    dset_name, condition, method, arch, hp_extra, seed = item
    ts = _DATASETS[(dset_name, seed)]
    hp = {**BASE_TRAIN_HPARAMS, **hp_extra}
    out = train_continual(
        MODEL_CLASSES[arch], model_kwargs_for(arch, ts.n_features),
        ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    return dict(
        dataset=dset_name, arch=arch, method=method, condition=condition,
        seed=seed, **m,
        buffer_capacity=hp.get("buffer_capacity"),
        replay_frac=hp.get("replay_frac"),
        ewc_lambda=hp.get("ewc_lambda"),
        lr=hp.get("lr"), epochs=hp.get("epochs"),
        train_seconds=float(sum(out["per_task_train_seconds"])),
    )


def main():
    global _FROZEN, _DATASETS, _CONTROLS
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "baseline_controls.csv")
    if args.stats_only:
        if not os.path.exists(out_csv):
            raise SystemExit(f"{out_csv} missing; run training first")
        compute_stats(pd.read_csv(out_csv))
        return

    _FROZEN = load_frozen_hparams_strict()
    cucd = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                           os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    stin = du.load_stin(os.path.join(DATA_DIR, "STIN", "SAT20.csv"),
                        os.path.join(DATA_DIR, "STIN", "TER20.csv"))
    for seed in SEEDS:
        _DATASETS[("CuCD-ID", seed)] = du.build_cucd_id_tasks(
            cucd, seed=seed, val_frac=VAL_FRAC)
        _DATASETS[("STIN", seed)] = du.build_stin_tasks(
            stin, seed=seed, val_frac=VAL_FRAC)

    _CONTROLS = _build_controls(_FROZEN)
    items = [(dset_name, condition, method, arch, hp_extra, seed)
             for dset_name in ("CuCD-ID", "STIN")
             for condition, method, arch, hp_extra in _CONTROLS
             for seed in SEEDS]
    jobs = get_jobs()
    print(f"Baseline controls: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['dataset']} {row['arch']} "
                  f"{row['condition']} seed={row['seed']} "
                  f"({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} runs)")
    compute_stats(df)


if __name__ == "__main__":
    main()
