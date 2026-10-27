"""Feature-provenance ablation for the random-split leakage claim.

CuCD-ID contains CCSDS identifiers, a circular time field, and 20-second
sliding-window aggregates. A random row split can share overlapping windows
between train and test, so a model may exploit aggregate/identifier shortcuts.
This ablation RETRAINS the model after removing feature groups (unlike the
test-time `nowindow` corruption in run_robustness.py) and reports the change
in pooled accuracy and Macro-F1.

Protocol: CuCD-ID / MLP, frozen per-method/arch hyperparameters, 15 seeds,
same random split and budgets as the main matrix. Variants:
  all                 - all 22 features (main-matrix reference)
  no_time             - drop TimeRadians
  no_ids              - drop MsgId, CmdCode, ApId
  no_window           - drop FlowLengthInWindow, SlidingWindow*, *InWindow
  no_time_ids_window  - drop all three provenance groups
  only_time_ids_window- keep only time+IDs+window, drop raw telemetry

The former ``only_raw_telemetry`` variant was an exact alias of
``no_time_ids_window`` (both drop the same columns), so it was removed; the
ablation is six unique variants x 3 methods x 15 seeds = 270 runs.

Writes:
  results/leakage_ablation.csv
  results/leakage_ablation_summary.csv
  results/leakage_group_effects.csv
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
from metrics import compute_cl_metrics, compute_macro_f1_balanced_acc
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, SEEDS, VAL_FRAC,
                    load_frozen_hparams_strict)
from feature_groups import CUCD_ID_GROUPS

METHODS = ["replay", "bic", "joint_scratch"]

TIME_COLS = CUCD_ID_GROUPS["time"]
ID_COLS = CUCD_ID_GROUPS["ids"]
WINDOW_COLS = CUCD_ID_GROUPS["window"]
RAW_TELEMETRY_COLS = CUCD_ID_GROUPS["raw_telemetry"]
LEAKAGE_COLS = TIME_COLS + ID_COLS + WINDOW_COLS

VARIANTS = {
    "all": [],
    "no_time": TIME_COLS,
    "no_ids": ID_COLS,
    "no_window": WINDOW_COLS,
    "no_time_ids_window": LEAKAGE_COLS,
    "only_time_ids_window": RAW_TELEMETRY_COLS,
}

GROUP_EFFECTS = [
    ("time", "no_time", "drop time"),
    ("ids", "no_ids", "drop IDs"),
    ("window", "no_window", "drop window"),
    ("all_leakage_provenance", "no_time_ids_window",
     "drop time+IDs+window"),
    ("raw_telemetry", "only_time_ids_window", "drop raw telemetry"),
]


_FROZEN = None
_TS = {}


def _run_one(item):
    variant, method, seed = item
    ts = _TS[(variant, seed)]
    mk = model_kwargs_for("MLP", ts.n_features)
    hp_method = _FROZEN.get("MLP", {}).get(method, {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **hp_method}
    out = train_continual(MODEL_CLASSES["MLP"], mk, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    mf_ba = compute_macro_f1_balanced_acc(
        out["model"], ts, out.get("corrector"))
    drop_cols = VARIANTS[variant]
    return dict(
        variant=variant,
        dropped_features="|".join(drop_cols) if drop_cols else "",
        n_features=ts.n_features,
        method=method, seed=seed,
        **m, **mf_ba)


def main():
    global _FROZEN, _TS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams_strict()
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"),
    )

    for variant, drop_cols in VARIANTS.items():
        missing = [c for c in drop_cols if c not in cucd_df.columns]
        if missing:
            raise KeyError(f"Leakage ablation: columns not found: {missing}")
        df_variant = cucd_df.drop(columns=drop_cols)
        for seed in SEEDS:
            _TS[(variant, seed)] = du.build_cucd_id_tasks(
                df_variant, seed=seed, val_frac=VAL_FRAC)

    items = [(variant, method, seed)
             for variant in VARIANTS
             for method in METHODS
             for seed in SEEDS]
    jobs = get_jobs()
    print(f"Leakage ablation: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['variant']} {row['method']} "
                  f"seed={row['seed']} ACC={row['ACC']:.3f} "
                  f"F1={row['MacroF1']:.3f} ({time.perf_counter()-t0:.0f}s)")

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
    summary.to_csv(os.path.join(RESULTS_DIR, "leakage_ablation_summary.csv"),
                   index=False)

    effect_rows = []
    for group, variant, description in GROUP_EFFECTS:
        sub = summary[summary.variant == variant]
        for _, r in sub.iterrows():
            effect_rows.append(dict(
                group=group, variant=variant, description=description,
                method=r["method"], n_features=int(r["n_features"]),
                dropped_features="|".join(VARIANTS[variant]),
                ACC_mean=float(r["ACC_mean"]),
                ACC_std=float(r["ACC_std"]),
                ACC_delta_vs_all=float(r["ACC_mean_delta_vs_all"]),
                MacroF1_mean=float(r["MacroF1_mean"]),
                MacroF1_delta_vs_all=float(r["MacroF1_mean_delta_vs_all"]),
            ))
    pd.DataFrame(effect_rows).to_csv(
        os.path.join(RESULTS_DIR, "leakage_group_effects.csv"), index=False)

    print("\nMean pooled ACC / Macro-F1 by variant and method:")
    print(summary[["variant", "method", "n_features", "ACC_mean",
                   "MacroF1_mean", "ACC_mean_delta_vs_all",
                   "MacroF1_mean_delta_vs_all"]].round(4).to_string(index=False))
    print(f"\nWrote results/leakage_ablation.csv ({len(df)} runs, "
          f"{time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
