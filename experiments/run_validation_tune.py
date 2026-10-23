"""Phase 2: nested validation-based hyperparameter tuning.

Each task's validation pool is split into two disjoint halves by the task
builder: calibration (BiC/EWC fitting) and selection (this script's score).
Hyperparameters are selected ONLY on the selection half; test data is never
touched.  Runs are parallelised across independent (method, arch, grid, seed)
jobs.

Output: results/best_hparams_per_method_arch.json
"""
import sys
import os
import json
import itertools
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from parallel import configure_single_thread, describe, parallel_map, get_jobs
configure_single_thread()

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics
from config import (
    MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
    BASE_TRAIN_HPARAMS, ARCHS, VAL_FRAC, VAL_CAP_PER_CLASS
)

from tuning_grids import TUNING_GRIDS, TUNING_SEEDS

_FROZEN_TS = {}


def generate_combos(grid):
    """Generate all hyperparameter combinations from a grid dict."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _run_one(item):
    arch, method, combo_idx, hp_combo, seed = item
    ts = _FROZEN_TS[seed]
    mk = model_kwargs_for(arch, ts.n_features)
    hp = {**BASE_TRAIN_HPARAMS, **hp_combo}
    out = train_continual(
        MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed)
    V = out.get("V")
    if V is None or np.all(np.isnan(V)):
        raise RuntimeError(
            f"No selection rows for {arch}/{method} seed={seed}; "
            "refusing to fall back to test data")
    metrics = compute_cl_metrics(V)
    return dict(arch=arch, method=method, combo_idx=combo_idx, seed=seed,
                hp_combo=hp_combo, **metrics)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Only tune these methods. Existing rows for others "
                         "are preserved.")
    ap.add_argument("--archs", nargs="*", default=None,
                    help="Only tune these architectures.")
    ap.add_argument("--jobs", type=int, default=None,
                    help="Worker processes (default: 4 / CL_JOBS).")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    global _FROZEN_TS

    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))

    methods = args.methods or list(TUNING_GRIDS.keys())
    archs = args.archs or list(ARCHS)
    unknown = [m for m in methods if m not in TUNING_GRIDS]
    unknown += [a for a in archs if a not in ARCHS]
    if unknown:
        raise SystemExit(f"Unknown method/arch: {unknown}")
    selective = (set(methods) != set(TUNING_GRIDS)) or (set(archs) != set(ARCHS))

    output_path = os.path.join(RESULTS_DIR, "best_hparams_per_method_arch.json")
    results = {}
    if selective and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    all_rows = []
    csv_path = os.path.join(RESULTS_DIR, "validation_tuning_results.csv")
    if selective and os.path.exists(csv_path):
        old_rows = pd.read_csv(csv_path)
        keep = ~(old_rows["method"].isin(methods) & old_rows["arch"].isin(archs))
        all_rows = old_rows[keep].to_dict("records")

    _FROZEN_TS = {
        seed: du.build_cucd_id_tasks(
            cucd_df, seed=seed, val_frac=VAL_FRAC,
            val_cap_per_class=VAL_CAP_PER_CLASS)
        for seed in TUNING_SEEDS
    }

    combo_by_method = {
        method: list(generate_combos(TUNING_GRIDS[method]))
        for method in methods
    }
    items = []
    for arch in archs:
        for method in methods:
            for combo_idx, hp_combo in enumerate(combo_by_method[method]):
                for seed in TUNING_SEEDS:
                    items.append((arch, method, combo_idx, hp_combo, seed))

    total_combos = sum(len(v) for v in combo_by_method.values())
    print(f"Total tuning runs: {len(items)} "
          f"({total_combos} combos x {len(archs)} archs x "
          f"{len(TUNING_SEEDS)} seeds)")
    jobs = get_jobs(args.jobs)
    print(f"Parallelism: {describe(jobs)}")
    print("Selection uses the disjoint selection half only; "
          "BiC/EWC fit on the calibration half.")

    t_start = time.perf_counter()
    raw_rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        raw_rows.append(row)
        if i % 100 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['arch']} {row['method']} "
                  f"seed={row['seed']} ACC={row['ACC']:.3f} "
                  f"({time.perf_counter()-t_start:.1f}s)")

    # Aggregate exactly as the previous sequential script: one row per
    # (arch, method, combo), mean/std over the three tuning seeds.
    grouped = {}
    for row in raw_rows:
        key = (row["arch"], row["method"], row["combo_idx"])
        grouped.setdefault(key, []).append(row)

    results = results or {}
    for arch in archs:
        results.setdefault(arch, {})
        for method in methods:
            best_score = -float("inf")
            best_hp = None
            best_metrics = None
            for combo_idx, hp_combo in enumerate(combo_by_method[method]):
                seed_rows = grouped.get((arch, method, combo_idx), [])
                if not seed_rows:
                    continue
                scores = [r["ACC"] - max(r["FM"], 0.0) for r in seed_rows]
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
                        k: float(np.mean([r[k] for r in seed_rows]))
                        for k in ("ACC", "BWT", "FM")
                    }
            results[arch][method] = {
                "hparams": best_hp,
                "val_score": best_score,
                "val_metrics_mean": best_metrics,
            }
            print(f"    {arch}/{method}: best_val_score={best_score:.4f}, "
                  f"hparams={best_hp}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(
        os.path.join(RESULTS_DIR, "validation_tuning_results.csv"), index=False)

    print(f"\n=== Tuning Complete ({time.perf_counter()-t_start:.1f}s) ===")
    print(f"Best configs saved to: {output_path}")
    print("Summary of best validation scores:")
    for arch in archs:
        print(f"\n  {arch}:")
        for method in methods:
            r = results[arch][method]
            print(f"    {method:10s}: val_score={r['val_score']:.4f}, "
                  f"hparams={r['hparams']}")


if __name__ == "__main__":
    main()
