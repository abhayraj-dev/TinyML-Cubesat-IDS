"""Phase 2c: disentangled task-order study (RQ5).

Legacy ablation confounded class order with training-volume placement (the
700-sample T0 budget moved with the task). Here every permutation is run in
two budget modes (see data_utils.build_cucd_id_tasks_order):
  - fixed_position: position 0 always gets the 700-sample pre-launch corpus
    (isolates pure class-order effects);
  - moves_with_task: budgets travel with the task (reproduces legacy numbers).

12 permutations x 2 modes x [replay, icarl] x 3 seeds on CuCD-ID/MLP.
Post-hoc: normal-first vs normal-last contrast (normal lives in T0).

Writes: results/task_order_disentangled.csv
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS)


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()

FIXED_PERMS = [
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    [1, 3, 0, 2],
]
rng = np.random.default_rng(7)
EXTRA = []
while len(EXTRA) < 9:
    p = list(rng.permutation(4))
    if p not in FIXED_PERMS and p not in EXTRA:
        EXTRA.append(p)
PERMS = FIXED_PERMS + EXTRA
MODES = ["fixed_position", "moves_with_task"]
METHODS = ["replay", "icarl"]
SEEDS = [0, 1, 2]
VAL_FRAC = 0.10


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frozen_hparams = load_frozen_hparams()
    cucd_df = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                              os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    rows = []
    t0 = time.perf_counter()
    total = len(PERMS) * len(MODES) * len(METHODS) * len(SEEDS)
    done = 0
    for order in PERMS:
        normal_first = (order[0] == 0)
        for mode in MODES:
            for method in METHODS:
                arch_hparams = frozen_hparams.get("MLP", {}).get(method, {}).get("hparams", {})
                for seed in SEEDS:
                    ts = du.build_cucd_id_tasks_order(cucd_df, order, seed=seed,
                                                      val_frac=VAL_FRAC, budget_mode=mode)
                    mk = model_kwargs_for("MLP", ts.n_features)
                    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
                    out = train_continual(MODEL_CLASSES["MLP"], mk, ts, method, hp, seed=seed)
                    m = compute_cl_metrics(out["R"])
                    rows.append(dict(order=str(order), budget_mode=mode, method=method,
                                     seed=seed, normal_first=normal_first, **m))
                    done += 1
                    if done % 24 == 0:
                        print(f"  [{done}/{total}] order={order} {mode} {method} "
                              f"seed={seed} ACC={m['ACC']:.3f} ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "task_order_disentangled.csv"), index=False)
    print("\nMean ACC by (budget_mode, method):")
    print(df.groupby(["budget_mode", "method"])["ACC"].agg(["mean", "std"]).round(3).to_string())
    print("\nMean ACC by (budget_mode, normal_first):")
    print(df.groupby(["budget_mode", "normal_first"])["ACC"].agg(["mean", "std"]).round(3).to_string())
    print(f"\nWrote results/task_order_disentangled.csv ({len(df)} runs, {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
