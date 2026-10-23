"""Phase 2c: disentangled task-order study (RQ5), confirmatory split.

Two budget modes:
  - fixed_position: position 0 always gets the 700-sample pre-launch corpus;
  - moves_with_task: budgets travel with the task.

Important estimand split: a first task with one class is DEGENERATE under a
single-logit softmax -- cross-entropy is constant and its gradient is exactly
zero, so the first update is a no-op.  We therefore report:
  * primary   -- 6 permutations with the two-class T0 first (non-degenerate);
  * degenerate -- 6 permutations with a single-class first task (no-op update).
The primary comparison estimates order sensitivity without conflating it with
a skipped first update.

Writes: results/task_order_disentangled.csv
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
from metrics import compute_cl_metrics
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, VAL_FRAC)

T0_FIRST = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3],
    [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
]
SINGLE_FIRST = [
    [3, 2, 1, 0], [1, 3, 0, 2], [2, 1, 0, 3],
    [3, 0, 2, 1], [1, 0, 3, 2], [2, 3, 0, 1],
]
PERMS = T0_FIRST + SINGLE_FIRST
MODES = ["fixed_position", "moves_with_task"]
METHODS = ["replay", "icarl"]
SEEDS = [0, 1, 2]
_FROZEN = None
_TS = {}


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


def _run_one(item):
    order, mode, method, seed = item
    ts = _TS[(tuple(order), mode, seed)]
    hp = {**BASE_TRAIN_HPARAMS,
          **_FROZEN.get("MLP", {}).get(method, {}).get("hparams", {})}
    out = train_continual(
        MODEL_CLASSES["MLP"], model_kwargs_for("MLP", ts.n_features),
        ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    n_first = len(du.CUCD_TASK_DEFS[order[0]][1])
    return dict(
        order="[" + ", ".join(str(int(x)) for x in order) + "]",
        budget_mode=mode, method=method, seed=seed,
        normal_first=(order[0] == 0),
        first_task=order[0],
        first_task_n_classes=n_first,
        single_class_first=(n_first == 1),
        degenerate_single_class_first=(n_first == 1),
        **m)


def main():
    global _FROZEN, _TS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams()
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))

    for order in PERMS:
        for mode in MODES:
            for seed in SEEDS:
                _TS[(tuple(order), mode, seed)] = du.build_cucd_id_tasks_order(
                    cucd_df, order, seed=seed, val_frac=VAL_FRAC,
                    budget_mode=mode)

    items = [(order, mode, method, seed)
             for order in PERMS
             for mode in MODES
             for method in METHODS
             for seed in SEEDS]
    jobs = get_jobs()
    print(f"Task-order study: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 24 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] order={row['order']} "
                  f"{row['budget_mode']} {row['method']} seed={row['seed']} "
                  f"ACC={row['ACC']:.3f} ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "task_order_disentangled.csv"),
              index=False)
    print("\nMean ACC by (budget_mode, method):")
    print(df.groupby(["budget_mode", "method"])["ACC"]
          .agg(["mean", "std"]).round(3).to_string())
    print("\nMean ACC by (budget_mode, degenerate_single_class_first):")
    print(df.groupby(["budget_mode", "degenerate_single_class_first"])["ACC"]
          .agg(["mean", "std"]).round(3).to_string())

    primary = df[~df.degenerate_single_class_first]
    degenerate = df[df.degenerate_single_class_first]
    if len(primary):
        spread = primary.groupby("order")["ACC"].mean()
        print(f"\nPRIMARY (two-class T0 first): per-order mean spread="
              f"{spread.max()-spread.min():.4f} across {len(spread)} orders")
    if len(degenerate):
        spread = degenerate.groupby("order")["ACC"].mean()
        print(f"DEGENERATE (single-class first, no-op update): per-order mean "
              f"spread={spread.max()-spread.min():.4f} across {len(spread)} orders")
    print(f"\nWrote results/task_order_disentangled.csv ({len(df)} runs, "
          f"{time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
