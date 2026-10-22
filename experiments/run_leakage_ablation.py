"""Feature-provenance ablation for the random-split leakage claim.

CuCD-ID contains CCSDS identifiers, a circular time field, and 20-second
sliding-window aggregates. A random row split can share overlapping windows
between train and test, so a model may exploit aggregate/identifier shortcuts.
This ablation RETRAINS the model after removing feature groups (unlike the
test-time `nowindow` corruption in run_robustness.py) and reports the change
in pooled accuracy and Macro-F1.

Protocol: CuCD-ID / MLP, frozen per-method/arch hyperparameters, 10 seeds,
same random split and budgets as the main matrix. Variants:
  all                 - all 22 features (main-matrix reference)
  no_time             - drop TimeRadians
  no_ids              - drop MsgId, CmdCode, ApId
  no_window           - drop FlowLengthInWindow, SlidingWindow*, *InWindow
  no_time_ids_window  - drop all three groups

Writes: results/leakage_ablation.csv and results/leakage_ablation_summary.csv
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
from metrics import compute_cl_metrics, compute_macro_f1_balanced_acc
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, SEEDS, load_frozen_hparams_strict)

METHODS = ["replay", "bic", "joint_scratch"]
VAL_FRAC = 0.10

ID_COLS = ["MsgId", "CmdCode", "ApId"]
WINDOW_COLS = [
    "FlowLengthInWindow",
    "SlidingWindowMeanIntervalSec",
    "SlidingWindowMaxIntervalSec",
    "SlidingWindowMinIntervalSec",
    "MessageCountInWindow",
    "UniqueMessageIDsInWindow",
    "AverageMessageLengthInWindow",
    "MessageRateInWindow",
    "StdDevMessageLengthInWindow",
]

VARIANTS = {
    "all": [],
    "no_time": ["TimeRadians"],
    "no_ids": ID_COLS,
    "no_window": WINDOW_COLS,
    "no_time_ids_window": ["TimeRadians"] + ID_COLS + WINDOW_COLS,
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frozen = load_frozen_hparams_strict()
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"),
    )

    rows = []
    t0 = time.perf_counter()
    total = len(VARIANTS) * len(METHODS) * len(SEEDS)
    done = 0

    for variant, drop_cols in VARIANTS.items():
        missing = [c for c in drop_cols if c not in cucd_df.columns]
        if missing:
            raise KeyError(f"Leakage ablation: columns not found: {missing}")
        df_variant = cucd_df.drop(columns=drop_cols)
        for method in METHODS:
            hp_method = frozen.get("MLP", {}).get(method, {}).get("hparams", {})
            for seed in SEEDS:
                ts = du.build_cucd_id_tasks(df_variant, seed=seed, val_frac=VAL_FRAC)
                mk = model_kwargs_for("MLP", ts.n_features)
                hp = {**BASE_TRAIN_HPARAMS, **hp_method}
                out = train_continual(MODEL_CLASSES["MLP"], mk, ts, method, hp, seed=seed)
                m = compute_cl_metrics(out["R"])
                mf_ba = compute_macro_f1_balanced_acc(
                    out["model"], ts, out.get("corrector"))
                rows.append(dict(
                    variant=variant,
                    dropped_features="|".join(drop_cols) if drop_cols else "",
                    n_features=ts.n_features,
                    method=method,
                    seed=seed,
                    **m,
                    **mf_ba,
                ))
                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] {variant} {method} seed={seed} "
                          f"ACC={m['ACC']:.3f} F1={mf_ba['MacroF1']:.3f} "
                          f"({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "leakage_ablation.csv"), index=False)

    summary = df.groupby(["variant", "method"]).agg(
        n_features=("n_features", "first"),
        ACC_mean=("ACC", "mean"), ACC_std=("ACC", "std"),
        MacroF1_mean=("MacroF1", "mean"), MacroF1_std=("MacroF1", "std"),
        BWT_mean=("BWT", "mean"), FM_mean=("FM", "mean"),
    ).reset_index()
    baseline = summary[summary.variant == "all"].set_index("method")
    for metric in ["ACC_mean", "MacroF1_mean"]:
        summary[f"{metric}_delta_vs_all"] = summary.apply(
            lambda r: r[metric] - baseline.loc[r["method"], metric], axis=1)
    summary.to_csv(os.path.join(RESULTS_DIR, "leakage_ablation_summary.csv"), index=False)

    print("\nMean pooled ACC / Macro-F1 by variant and method:")
    print(summary[["variant", "method", "n_features", "ACC_mean",
                   "MacroF1_mean", "ACC_mean_delta_vs_all",
                   "MacroF1_mean_delta_vs_all"]].round(4).to_string(index=False))
    print(f"\nWrote results/leakage_ablation.csv ({len(df)} runs, "
          f"{time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
