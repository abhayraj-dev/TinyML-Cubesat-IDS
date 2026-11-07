"""First-differencing robustness check for CuCD-ID.

Motivation
----------
Singh, Pandey and Kumar (IEEE Access, 2026) show that scenario label and
recording session are confounded in CuCD-ID: constant session-level resource
offsets can separate the classes.  Their remedy is to first-difference the
resource-state features before evaluating temporal blocks.

This experiment asks a narrower question: do the class-incremental update
conclusions in this project survive when the same first-differencing control
is applied?  The CIL methods, budgets, frozen hyperparameters and seeds are
unchanged.  Only the feature representation changes.

Four representations are evaluated:
  * raw            -- MsgLength plus the eight Memory* resource-state columns;
                      time, packet IDs and window aggregates are kept.
  * raw_window     -- the raw set plus the nine 20-second sliding-window
                      columns; time and packet IDs are kept.  This is a stress
                      test, not a better control.
  * resource_state -- the 17 retained resource-state features in Singh et al.
                      (eight Memory* gauges plus nine window aggregates), with
                      MsgLength, time and packet IDs dropped.  This is the
                      closest analogue of their control.
  * raw_no_prov    -- the raw set, then time, packet IDs and window aggregates
                      are dropped.  This is stricter than Singh et al. because
                      it removes the differenced window signal entirely.

Protocols:
  * iid                    -- the original random split, for comparability
  * temporal_purged_fair    -- fair forward-temporal split with purge gap,
                              random-past training/validation, future test

Outputs (under experiments/first_difference/results/):
  first_difference_runs.csv
  first_difference_summary.csv
  first_difference_comparison.csv
  first_difference_report.md
  first_difference_setup.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from parallel import configure_single_thread, describe, get_jobs, parallel_map
configure_single_thread()

import numpy as np
import pandas as pd

import data_utils as du
from feature_groups import CUCD_ID_GROUPS
from train import train_continual
from metrics import compute_cl_metrics, compute_macro_f1_balanced_acc
from run_purged_temporal import final_detection_metrics
from config import (
    BASE_TRAIN_HPARAMS,
    DATA_DIR,
    METHODS,
    MODEL_CLASSES,
    RESULTS_DIR,
    VAL_FRAC,
    load_frozen_hparams_strict,
    model_kwargs_for,
)

ARCH = "MLP"
SEEDS = list(range(15))
PROTOCOLS = ["iid", "temporal_purged_fair"]
DIFF_SETS = {
    "raw": list(CUCD_ID_GROUPS["raw_telemetry"]),
    "raw_window": list(CUCD_ID_GROUPS["raw_telemetry"])
    + list(CUCD_ID_GROUPS["window"]),
    "resource_state": [
        c for c in CUCD_ID_GROUPS["raw_telemetry"] if c != "MsgLength"
    ] + list(CUCD_ID_GROUPS["window"]),
    "raw_no_prov": list(CUCD_ID_GROUPS["raw_telemetry"]),
}
PROVENANCE_COLS = (
    ["TimeRadians"] + list(CUCD_ID_GROUPS["ids"]) + list(CUCD_ID_GROUPS["window"])
)
RESOURCE_STATE_DROP = (
    ["TimeRadians"] + list(CUCD_ID_GROUPS["ids"]) + ["MsgLength"]
)
FEATURE_DROP = {
    "raw": [],
    "raw_window": [],
    "resource_state": RESOURCE_STATE_DROP,
    "raw_no_prov": PROVENANCE_COLS,
}

_FROZEN = None
_DATASETS = {}
_TS = {}
_BASELINE = None


def _parse_csv(value, default):
    if not value:
        return list(default)
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_seeds(value):
    if not value:
        return list(SEEDS)
    if "-" in value:
        lo, hi = [int(x) for x in value.split("-", 1)]
        return list(range(lo, hi + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def first_difference_frame(df: pd.DataFrame, diff_cols) -> pd.DataFrame:
    """First-difference selected columns within each class in time order.

    Constant session-level offsets disappear.  The first row of each class is
    dropped because it has no predecessor.  TimeRadians, packet identifiers and
    all non-differenced columns are left untouched.
    """
    diff_cols = list(diff_cols)
    missing = [c for c in diff_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing first-difference columns: {missing}")
    if "TimeRadians" not in df.columns:
        raise ValueError("CuCD-ID first-differencing requires TimeRadians")

    parts = []
    for label, sub in df.groupby("label_name", sort=False):
        if len(sub) < 2:
            raise ValueError(f"Class '{label}' has fewer than two rows")
        ordered, _ = du._chronological_order(sub, time_col="TimeRadians")
        values = ordered[diff_cols].to_numpy(dtype=np.float64, copy=True)
        deltas = np.diff(values, axis=0)
        kept = ordered.iloc[1:].copy()
        kept[diff_cols] = deltas.astype(np.float32)
        parts.append(kept)
    return pd.concat(parts, axis=0).sort_index()


def _drop_features(ts, drop_names):
    """Drop feature columns after the split has been built.

    Splitting still needs TimeRadians, so provenance fields cannot be removed
    from the dataframe before the temporal builder runs.  This helper removes
    them from every task array and from the scaler metadata afterwards.
    """
    drop = set(drop_names)
    if not drop:
        return ts
    keep = [i for i, name in enumerate(ts.feature_names) if name not in drop]
    missing = drop.difference(ts.feature_names)
    if missing:
        raise ValueError(f"Feature drop requested but not present: {sorted(missing)}")
    for task in ts.tasks:
        for attr in ("X_train", "X_test", "X_val", "X_calib", "X_select"):
            arr = getattr(task, attr, None)
            if arr is not None:
                setattr(task, attr, np.ascontiguousarray(arr[:, keep]))
    ts.feature_names = [ts.feature_names[i] for i in keep]
    if ts.scaler_mean is not None:
        ts.scaler_mean = ts.scaler_mean[keep]
    if ts.scaler_std is not None:
        ts.scaler_std = ts.scaler_std[keep]
    return ts


def _build_tasks(df: pd.DataFrame, protocol: str, seed: int):
    if protocol == "iid":
        return du.build_cucd_id_tasks(
            df, seed=seed, test_frac=0.30, val_frac=VAL_FRAC)
    if protocol == "temporal_purged_fair":
        return du.build_cucd_id_tasks_purged_temporal(
            df, seed=seed, test_frac=0.20, val_frac=0.10,
            purge_time_rad=0.02, train_sample_mode="random_past",
            val_mode="random_past")
    raise ValueError(f"Unknown protocol: {protocol}")


def _run_one(item):
    diff_set, protocol, method, seed = item
    ts = _TS[(diff_set, protocol, seed)]
    model_kwargs = model_kwargs_for(ARCH, ts.n_features)
    arch_hparams = _FROZEN.get(ARCH, {}).get(method, {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
    out = train_continual(
        MODEL_CLASSES[ARCH], model_kwargs, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    mf = compute_macro_f1_balanced_acc(
        out["model"], ts, out.get("corrector"), ncm=out.get("ncm"))
    det = final_detection_metrics(ts, out)
    return dict(
        diff_set=diff_set,
        protocol=protocol,
        arch=ARCH,
        method=method,
        seed=seed,
        n_features=ts.n_features,
        **m,
        **mf,
        **det,
    )


def _summary(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.groupby(["diff_set", "protocol", "method"]).agg(
        ACC_mean=("ACC", "mean"),
        ACC_std=("ACC", "std"),
        BWT_mean=("BWT", "mean"),
        BWT_std=("BWT", "std"),
        FM_mean=("FM", "mean"),
        FM_std=("FM", "std"),
        MacroF1_mean=("MacroF1", "mean"),
        MacroF1_std=("MacroF1", "std"),
        BalancedAccuracy_mean=("BalancedAccuracy", "mean"),
        BalancedAccuracy_std=("BalancedAccuracy", "std"),
        DET_AUROC_mean=("DET_AUROC", "mean"),
        DET_AUROC_std=("DET_AUROC", "std"),
        n_runs=("ACC", "size"),
    ).reset_index()


def _comparison(runs: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base = baseline[
        baseline["protocol"].isin(PROTOCOLS)
        & baseline["method"].isin(METHODS)
    ][["protocol", "method", "seed", "ACC", "BWT", "FM"]].copy()
    merged = runs.merge(
        base, on=["protocol", "method", "seed"], how="inner",
        suffixes=("_diff", "_base"), validate="many_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged["delta_ACC"] = merged["ACC_diff"] - merged["ACC_base"]
    merged["delta_BWT"] = merged["BWT_diff"] - merged["BWT_base"]
    merged["delta_FM"] = merged["FM_diff"] - merged["FM_base"]
    comp = merged.groupby(["diff_set", "protocol", "method"]).agg(
        baseline_ACC=("ACC_base", "mean"),
        diff_ACC=("ACC_diff", "mean"),
        delta_ACC=("delta_ACC", "mean"),
        delta_ACC_std=("delta_ACC", "std"),
        ACC_wins=("delta_ACC", lambda s: int((s > 0).sum())),
        baseline_BWT=("BWT_base", "mean"),
        diff_BWT=("BWT_diff", "mean"),
        delta_BWT=("delta_BWT", "mean"),
        baseline_FM=("FM_base", "mean"),
        diff_FM=("FM_diff", "mean"),
        delta_FM=("delta_FM", "mean"),
        n_seeds=("delta_ACC", "size"),
    ).reset_index()
    return comp


def _rank_table(summary: pd.DataFrame, diff_set: str, protocol: str):
    sub = summary[
        (summary["diff_set"] == diff_set) & (summary["protocol"] == protocol)
    ].sort_values("ACC_mean", ascending=False)
    return sub[["method", "ACC_mean", "ACC_std", "FM_mean", "BWT_mean"]] \
        .reset_index(drop=True)


def _write_report(path, setup, summary, comparison):
    lines = [
        "# First-differencing robustness check",
        "",
        "Question: do the class-incremental conclusions survive when the "
        "resource-state session offsets identified by Singh et al. are removed "
        "by first-differencing?",
        "",
        "Setup:",
        "",
        f"- Architecture: `{setup['arch']}`",
        f"- Seeds: {setup['seeds'][0]}--{setup['seeds'][-1]} "
        f"({len(setup['seeds'])} seeds)",
        f"- Methods: {', '.join(setup['methods'])}",
        f"- Protocols: {', '.join(setup['protocols'])}",
        f"- Rows before differencing: {setup['rows_before']}",
        "- Rows after differencing: "
        + ", ".join(f"{k}={v}" for k, v in setup["rows_after"].items()),
        "",
        "Differenced feature sets:",
        "",
    ]
    for name, cols in setup["diff_sets"].items():
        lines.append(f"- `{name}` ({len(cols)} columns): `" + "`, `".join(cols) + "`")
    for name, drops in setup.get("feature_drop", {}).items():
        if drops:
            lines.append(f"- `{name}` also drops: `" + "`, `".join(drops) + "`")
    lines += ["", "Baseline source: `results/temporal_purged_compare_MLP.csv`", ""]

    for diff_set in setup["diff_sets"]:
        lines += [f"## {diff_set}", ""]
        for protocol in setup["protocols"]:
            table = _rank_table(summary, diff_set, protocol)
            lines += [f"### {protocol}", "", table.to_markdown(index=False), ""]
            comp = comparison[
                (comparison["diff_set"] == diff_set)
                & (comparison["protocol"] == protocol)
            ].sort_values("delta_ACC", ascending=False)
            if not comp.empty:
                show = comp[[
                    "method", "baseline_ACC", "diff_ACC", "delta_ACC",
                    "ACC_wins", "n_seeds",
                ]].copy()
                lines += ["Delta versus baseline:", "",
                          show.to_markdown(index=False), ""]

    lines += [
        "## Reading the result",
        "",
        "- A positive `delta_ACC` means first-differencing improved accuracy.",
        "- The key check is not whether every number moves, but whether the "
        "qualitative ranking changes: memory-free methods still collapse, "
        "exemplar methods still retain most accuracy, and the best exemplar "
        "method remains consistent with the original conclusion.",
        "- If the fair-temporal advantage disappears after differencing, the "
        "original result is partly a session-offset artifact and should not be "
        "used as the headline claim.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    global _FROZEN, _DATASETS, _TS, _BASELINE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diff-sets", default="raw,raw_window,resource_state,raw_no_prov")
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="0-14")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument(
        "--append", action="store_true",
        help="Keep existing rows for other diff sets and merge this run.")
    args = parser.parse_args()

    diff_sets = _parse_csv(args.diff_sets, DIFF_SETS)
    protocols = _parse_csv(args.protocols, PROTOCOLS)
    methods = _parse_csv(args.methods, METHODS)
    seeds = _parse_seeds(args.seeds)

    for name in diff_sets:
        if name not in DIFF_SETS:
            raise ValueError(f"Unknown diff set '{name}'")
    for protocol in protocols:
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol '{protocol}'")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'")

    _FROZEN = load_frozen_hparams_strict()
    baseline_path = os.path.join(
        RESULTS_DIR, "temporal_purged_compare_MLP.csv")
    _BASELINE = pd.read_csv(baseline_path)

    raw_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))

    _DATASETS = {
        name: first_difference_frame(raw_df, DIFF_SETS[name])
        for name in diff_sets
    }
    for name, frame in _DATASETS.items():
        print(f"{name}: {len(frame)} rows, {frame.shape[1]} columns")

    for diff_set in diff_sets:
        for protocol in protocols:
            for seed in seeds:
                ts = _build_tasks(_DATASETS[diff_set], protocol, seed)
                ts = _drop_features(ts, FEATURE_DROP[diff_set])
                _TS[(diff_set, protocol, seed)] = ts

    items = [
        (diff_set, protocol, method, seed)
        for diff_set in diff_sets
        for protocol in protocols
        for method in methods
        for seed in seeds
    ]
    jobs = get_jobs(args.jobs or None)
    print(f"First-difference matrix: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 20 == 0 or i == len(items):
            print(
                f"  [{i}/{len(items)}] {row['diff_set']} {row['protocol']} "
                f"{row['method']} seed={row['seed']} ACC={row['ACC']:.3f} "
                f"({time.perf_counter() - t0:.1f}s)")

    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    runs_path = os.path.join(out_dir, "first_difference_runs.csv")
    summary_path = os.path.join(out_dir, "first_difference_summary.csv")
    comp_path = os.path.join(out_dir, "first_difference_comparison.csv")
    report_path = os.path.join(out_dir, "first_difference_report.md")
    setup_path = os.path.join(out_dir, "first_difference_setup.json")

    runs = pd.DataFrame(rows)
    if args.append and os.path.exists(runs_path):
        previous = pd.read_csv(runs_path)
        previous = previous[~previous["diff_set"].isin(diff_sets)]
        runs = pd.concat([previous, runs], ignore_index=True)
        print(f"Merged with existing runs: {len(runs)} total rows")

    summary = _summary(runs)
    comparison = _comparison(runs, _BASELINE)
    all_diff_sets = [d for d in DIFF_SETS if d in set(runs["diff_set"])]

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comp_path, index=False)

    setup = {
        "arch": ARCH,
        "methods": methods,
        "protocols": protocols,
        "seeds": seeds,
        "rows_before": int(len(raw_df)),
        "rows_after": {k: int(len(v)) for k, v in _DATASETS.items()},
        "diff_sets": {k: DIFF_SETS[k] for k in all_diff_sets},
        "feature_drop": {k: FEATURE_DROP[k] for k in all_diff_sets},
        "baseline": baseline_path,
        "n_runs": len(runs),
        "seconds": float(time.perf_counter() - t0),
    }
    with open(setup_path, "w") as f:
        json.dump(setup, f, indent=2)
    _write_report(report_path, setup, summary, comparison)

    print("\nMean ACC by (diff_set, protocol, method):")
    print(summary.pivot_table(
        index=["diff_set", "protocol"], columns="method",
        values="ACC_mean").round(3).to_string())
    print(f"\nWrote {runs_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {comp_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
