"""Phase 2d (SUPERSEDED by run_purged_temporal.py; not in run_all.py):
temporal-vs-random split comparison (leakage quantification).

CuCD-ID features contain SlidingWindow*/InWindow aggregates over neighbouring
rows, so random row-shuffles may leak correlated windows across train/val/test
and inflate absolute ACC. Compares the identical protocol (budgets, val_frac,
hparams, seeds) under split_mode random vs temporal (time-ordered contiguous
blocks by TimeRadians, test = future block).

CuCD-ID/MLP x [replay, icarl, bic, joint_scratch] x {random, temporal} x 5 seeds.

Writes: results/temporal_compare.csv
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS)
from run_main import load_best_hparams, final_detection_metrics

METHODS = ["replay", "icarl", "bic", "joint_scratch"]
SEEDS = [0, 1, 2, 3, 4]
VAL_FRAC = 0.10


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_hp = load_best_hparams()
    cucd_df = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                              os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    rows = []
    t0 = time.perf_counter()
    total = len(METHODS) * 2 * len(SEEDS)
    done = 0
    for method in METHODS:
        for mode in ["random", "temporal"]:
            for seed in SEEDS:
                if mode == "random":
                    ts = du.build_cucd_id_tasks(cucd_df, seed=seed, val_frac=VAL_FRAC)
                else:
                    ts = du.build_cucd_id_tasks_temporal(cucd_df, seed=seed, val_frac=VAL_FRAC)
                mk = model_kwargs_for("MLP", ts.n_features)
                hp = {**BASE_TRAIN_HPARAMS, **best_hp}
                out = train_continual(MODEL_CLASSES["MLP"], mk, ts, method, hp, seed=seed)
                m = compute_cl_metrics(out["R"])
                det = final_detection_metrics(ts, out)
                rows.append(dict(method=method, split=mode, seed=seed, **m,
                                 DET_AUROC=det["DET_AUROC"]))
                done += 1
                print(f"  [{done}/{total}] {method} {mode} seed={seed} "
                      f"ACC={m['ACC']:.3f} ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "temporal_compare.csv"), index=False)
    print("\nMean ACC by (method, split):")
    print(df.groupby(["method", "split"])["ACC"].agg(["mean", "std"]).round(3).to_string())
    print(f"\nWrote results/temporal_compare.csv ({len(df)} runs, {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
