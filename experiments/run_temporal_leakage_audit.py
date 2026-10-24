#!/usr/bin/env python3
"""Read-only audit of the random-split temporal-validity problem.

This script trains no models and does not overwrite any experiment result. It
replays the exact random per-class splits used by build_cucd_id_tasks,
including the builder's per-task RNG permutation, and verifies row-level parity
before writing diagnostic tables:

  results/temporal_leakage_audit.csv          per-seed nearest-train distances
  results/temporal_leakage_audit_summary.csv pooled per-class summary
  results/temporal_leakage_histogram.csv      log-binned distance histograms
  results/temporal_class_occupancy.csv        class time spans / block coverage
  results/temporal_block_coverage.csv         natural time blocks and labels
  results/temporal_class_block_ranges.csv     per-class time within each block
  results/temporal_drift_by_class.csv         early-vs-late feature drift
  results/temporal_drift_features.csv         per-feature drift, long form
  results/temporal_leakage_audit.json         machine-readable summary

Nearest-distance uses circular `TimeRadians` distance to the training rows
actually drawn for the model, not the larger pre-draw train pool. The audit
therefore measures adjacency to examples the model can actually memorize.
Time blocks and early/late drift use the repository's unwrapped chronological
time so an orbital wrap does not create a false break.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from config import (DATA_DIR, RESULTS_DIR, SEEDS, VAL_FRAC,
                    VAL_CAP_PER_CLASS)

PURGE_TIME_RAD = 0.02
BLOCK_GAP_RAD = 0.02
TEST_FRAC = 0.30
INITIAL_N_PER_CLASS = 700
UPLINK_N_PER_CLASS = 150
DRIFT_EXCLUDE = {"TimeRadians", "unwrapped_time_rad"}
CLASS_ORDER = [label for _, labels in du.CUCD_TASK_DEFS for label in labels]


def _nearest_circular_distance(train_times, test_times):
    train_sorted = np.sort(np.asarray(train_times, dtype=float))
    train_extended = np.concatenate([
        train_sorted - 2.0 * np.pi, train_sorted, train_sorted + 2.0 * np.pi])
    test_times = np.asarray(test_times, dtype=float)
    pos = np.searchsorted(train_extended, test_times)
    lo = train_extended[np.clip(pos - 1, 0, len(train_extended) - 1)]
    hi = train_extended[np.clip(pos, 0, len(train_extended) - 1)]
    return np.minimum(np.abs(test_times - lo), np.abs(test_times - hi))


def _replay_random_split(df, seed):
    rng = np.random.default_rng(seed)
    splits = {}
    for task_idx, (_, class_list) in enumerate(du.CUCD_TASK_DEFS):
        n_draw = INITIAL_N_PER_CLASS if task_idx == 0 else UPLINK_N_PER_CLASS
        for label in class_list:
            idx = df.index[df["label_name"] == label].to_numpy().copy()
            train_pool, val_pool, test_idx = du._stratified_train_val_test_split(
                idx, TEST_FRAC, VAL_FRAC, rng)
            train_idx = du._draw(train_pool, n_draw, rng)
            val_idx = du._draw(
                val_pool, min(len(val_pool), VAL_CAP_PER_CLASS), rng)
            splits[label] = {
                "train_pool": train_pool,
                "val": val_idx,
                "train": train_idx,
                "test": test_idx,
            }
        rng.permutation(sum(len(splits[label]["train"])
                            for label in class_list))
    return splits


def _sorted_rows(values):
    values = np.asarray(values)
    return values[np.lexsort(values.T[::-1])]


def _assert_split_parity(df):
    for seed in SEEDS:
        ts = du.build_cucd_id_tasks(df, seed=seed, val_frac=VAL_FRAC)
        splits = _replay_random_split(df, seed=seed)
        for task, (_, class_list) in zip(ts.tasks, du.CUCD_TASK_DEFS):
            expected = {"train": [], "test": [], "val": []}
            for label in class_list:
                for split_name in expected:
                    idx = splits[label][split_name]
                    raw = df.loc[idx, ts.feature_names].to_numpy(dtype=np.float32)
                    expected[split_name].append(
                        du._apply_scaler(raw, ts.scaler_mean, ts.scaler_std))
            for split_name, actual in [
                ("train", task.X_train),
                ("test", task.X_test),
                ("val", task.X_val),
            ]:
                if actual is None:
                    continue
                expected_rows = _sorted_rows(np.concatenate(expected[split_name]))
                actual_rows = _sorted_rows(actual)
                assert np.array_equal(expected_rows, actual_rows), (
                    f"Replayed {split_name} rows do not match "
                    f"build_cucd_id_tasks for seed={seed}, {task.name}")


def _audit_distances(df):
    rows = []
    pooled = {label: [] for label in CLASS_ORDER}
    for seed in SEEDS:
        splits = _replay_random_split(df, seed)
        for label in CLASS_ORDER:
            parts = splits[label]
            train_times = df.loc[parts["train"], "TimeRadians"].to_numpy(float)
            test_times = df.loc[parts["test"], "TimeRadians"].to_numpy(float)
            distances = _nearest_circular_distance(train_times, test_times)
            pooled[label].append(distances)
            rows.append({
                "seed": seed,
                "label": label,
                "n_rows": int((df["label_name"] == label).sum()),
                "n_train_pool": len(parts["train_pool"]),
                "n_train_draw": len(parts["train"]),
                "n_test": len(parts["test"]),
                "nn_median_rad": float(np.median(distances)),
                "nn_q25_rad": float(np.quantile(distances, 0.25)),
                "nn_q75_rad": float(np.quantile(distances, 0.75)),
                "nn_p90_rad": float(np.quantile(distances, 0.90)),
                "nn_max_rad": float(distances.max()),
                "frac_within_purge": float(
                    np.mean(distances <= PURGE_TIME_RAD)),
                "frac_within_0p001_rad": float(
                    np.mean(distances <= 0.001)),
            })
    return pd.DataFrame(rows), pooled


def _summarize_distances(audit_df, pooled):
    rows = []
    for label in CLASS_ORDER:
        distances = np.concatenate(pooled[label])
        seed_fracs = audit_df.loc[
            audit_df["label"] == label, "frac_within_purge"]
        rows.append({
            "label": label,
            "n_rows": int(audit_df.loc[audit_df["label"] == label,
                                       "n_rows"].iloc[0]),
            "n_train_draw": int(audit_df.loc[audit_df["label"] == label,
                                            "n_train_draw"].iloc[0]),
            "n_test": int(audit_df.loc[audit_df["label"] == label,
                                      "n_test"].iloc[0]),
            "nn_median_rad": float(np.median(distances)),
            "nn_q25_rad": float(np.quantile(distances, 0.25)),
            "nn_q75_rad": float(np.quantile(distances, 0.75)),
            "nn_p90_rad": float(np.quantile(distances, 0.90)),
            "nn_max_rad": float(distances.max()),
            "frac_within_purge": float(np.mean(distances <= PURGE_TIME_RAD)),
            "frac_within_purge_seed_min": float(seed_fracs.min()),
            "frac_within_purge_seed_max": float(seed_fracs.max()),
            "frac_within_0p001_rad": float(np.mean(distances <= 0.001)),
        })
    return pd.DataFrame(rows)


def _histogram(audit_df, pooled):
    all_distances = np.concatenate([np.concatenate(pooled[label])
                                    for label in CLASS_ORDER])
    positive = all_distances[all_distances > 0]
    lo = max(float(positive.min()), 1e-8)
    hi = max(float(all_distances.max()), PURGE_TIME_RAD * 2)
    bins = np.logspace(np.floor(np.log10(lo)), np.ceil(np.log10(hi)), 45)
    rows = []
    for label in CLASS_ORDER:
        distances = np.concatenate(pooled[label])
        counts, _ = np.histogram(distances, bins=bins)
        for left, right, count in zip(bins[:-1], bins[1:], counts):
            rows.append({
                "label": label,
                "bin_left_rad": float(left),
                "bin_right_rad": float(right),
                "count": int(count),
                "fraction": float(count / len(distances)),
            })
    return pd.DataFrame(rows)


def _natural_blocks(df):
    times = np.sort(df["unwrapped_time_rad"].to_numpy(float))
    cuts = np.flatnonzero(np.diff(times) > BLOCK_GAP_RAD)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(times) - 1]
    rows = []
    for block_idx, (start, end) in enumerate(zip(starts, ends)):
        lo, hi = float(times[start]), float(times[end])
        mask = ((df["unwrapped_time_rad"] >= lo)
                & (df["unwrapped_time_rad"] <= hi))
        counts = df.loc[mask, "label_name"].value_counts()
        row = {
            "block_id": f"B{block_idx}",
            "time_min_rad": lo,
            "time_max_rad": hi,
            "gap_before_rad": 0.0 if block_idx == 0
            else lo - float(times[starts[block_idx] - 1]),
            "n_rows": int(mask.sum()),
            "n_classes": int(len(counts)),
            "labels_present": "|".join(
                label for label in CLASS_ORDER if counts.get(label, 0) > 0),
            "class_counts": "; ".join(
                f"{label}={int(counts.get(label, 0))}"
                for label in CLASS_ORDER if counts.get(label, 0) > 0),
        }
        for label in CLASS_ORDER:
            row[f"n_{label}"] = int(counts.get(label, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _class_occupancy(df, blocks):
    rows = []
    for label in CLASS_ORDER:
        sub = df[df["label_name"] == label].sort_values("unwrapped_time_rad")
        times = sub["unwrapped_time_rad"].to_numpy(float)
        occupied = [row["block_id"] for _, row in blocks.iterrows()
                    if row[f"n_{label}"] > 0]
        diffs = np.diff(np.sort(times))
        rows.append({
            "label": label,
            "n_rows": len(sub),
            "time_min_rad": float(times.min()),
            "time_max_rad": float(times.max()),
            "time_span_rad": float(times.max() - times.min()),
            "median_time_rad": float(np.median(times)),
            "n_blocks_occupied": len(occupied),
            "block_ids": ",".join(occupied),
            "largest_within_class_gap_rad": float(diffs.max())
            if len(diffs) else 0.0,
        })
    return pd.DataFrame(rows)


def _class_block_ranges(df, blocks):
    rows = []
    for label in CLASS_ORDER:
        for _, block in blocks.iterrows():
            if block[f"n_{label}"] == 0:
                continue
            mask = ((df["label_name"] == label)
                    & (df["unwrapped_time_rad"] >= block["time_min_rad"])
                    & (df["unwrapped_time_rad"] <= block["time_max_rad"]))
            times = df.loc[mask, "unwrapped_time_rad"].to_numpy(float)
            rows.append({
                "label": label,
                "block_id": block["block_id"],
                "time_min_rad": float(times.min()),
                "time_max_rad": float(times.max()),
                "n_rows": int(len(times)),
            })
    return pd.DataFrame(rows)


def _feature_drift(df):
    features = [c for c in df.columns
                if c not in ("Label", "label_name") and c not in DRIFT_EXCLUDE]
    long_rows = []
    class_rows = []
    for label in CLASS_ORDER:
        sub = df[df["label_name"] == label].sort_values("unwrapped_time_rad")
        split_time = float(sub["unwrapped_time_rad"].median())
        early = sub[sub["unwrapped_time_rad"] <= split_time]
        late = sub[sub["unwrapped_time_rad"] > split_time]
        class_features = []
        for feature in features:
            a = early[feature].to_numpy(float)
            b = late[feature].to_numpy(float)
            pooled_var = (((len(a) - 1) * a.var(ddof=1)
                           + (len(b) - 1) * b.var(ddof=1))
                          / max(1, len(a) + len(b) - 2))
            pooled_std = float(np.sqrt(pooled_var))
            z = 0.0 if pooled_std < 1e-12 else float(
                (b.mean() - a.mean()) / pooled_std)
            class_features.append({
                "label": label,
                "feature": feature,
                "early_n": len(a),
                "late_n": len(b),
                "early_mean": float(a.mean()),
                "late_mean": float(b.mean()),
                "pooled_std": pooled_std,
                "z": z,
                "abs_z": abs(z),
            })
        long_rows.extend(class_features)
        class_features.sort(key=lambda row: row["abs_z"], reverse=True)
        top = class_features[0]
        class_rows.append({
            "label": label,
            "split_time_rad": split_time,
            "early_n": top["early_n"],
            "late_n": top["late_n"],
            "mean_abs_z": float(np.mean([row["abs_z"]
                                         for row in class_features])),
            "max_abs_z": top["abs_z"],
            "top_feature": top["feature"],
            "top_feature_abs_z": top["abs_z"],
            "top5_features": "; ".join(
                f"{row['feature']} ({row['abs_z']:.2f})"
                for row in class_features[:5]),
        })
    return pd.DataFrame(long_rows), pd.DataFrame(class_rows)


def _json_records(df):
    return json.loads(df.to_json(orient="records"))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    _assert_split_parity(df)
    chrono, unwrapped = du._chronological_order(df, time_col="TimeRadians")
    df = df.copy()
    df["unwrapped_time_rad"] = np.nan
    df.loc[chrono.index, "unwrapped_time_rad"] = unwrapped

    audit_df, pooled = _audit_distances(df)
    summary_df = _summarize_distances(audit_df, pooled)
    histogram_df = _histogram(audit_df, pooled)
    blocks_df = _natural_blocks(df)
    occupancy_df = _class_occupancy(df, blocks_df)
    ranges_df = _class_block_ranges(df, blocks_df)
    drift_long_df, drift_class_df = _feature_drift(df)

    outputs = {
        "temporal_leakage_audit.csv": audit_df,
        "temporal_leakage_audit_summary.csv": summary_df,
        "temporal_leakage_histogram.csv": histogram_df,
        "temporal_class_occupancy.csv": occupancy_df,
        "temporal_block_coverage.csv": blocks_df,
        "temporal_class_block_ranges.csv": ranges_df,
        "temporal_drift_by_class.csv": drift_class_df,
        "temporal_drift_features.csv": drift_long_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(os.path.join(RESULTS_DIR, filename), index=False)

    all_test_distances = np.concatenate(
        [np.concatenate(pooled[label]) for label in CLASS_ORDER])
    payload = {
        "audit_type": "read-only temporal leakage and validity audit",
        "trained_models": False,
        "random_split": {
            "test_frac": TEST_FRAC,
            "val_frac": VAL_FRAC,
            "seeds": list(SEEDS),
            "initial_n_per_class_t0": INITIAL_N_PER_CLASS,
            "uplink_n_per_class": UPLINK_N_PER_CLASS,
            "distance_target": "training rows actually drawn, not train pool",
        },
        "purge_time_rad": PURGE_TIME_RAD,
        "nearest_distance": "circular TimeRadians distance",
        "time_basis": "unwrapped chronological time from _chronological_order",
        "n_test_rows_pooled": int(all_test_distances.size),
        "overall_nn_median_rad": float(np.median(all_test_distances)),
        "overall_frac_within_purge": float(
            np.mean(all_test_distances <= PURGE_TIME_RAD)),
        "n_natural_blocks": int(len(blocks_df)),
        "block_gap_rad": BLOCK_GAP_RAD,
        "class_summary": _json_records(summary_df),
        "blocks": _json_records(blocks_df),
        "class_occupancy": _json_records(occupancy_df),
        "class_block_ranges": _json_records(ranges_df),
        "drift_summary": _json_records(drift_class_df),
        "drift_excluded_features": sorted(DRIFT_EXCLUDE),
        "interpretation": [
            "Nearest-time adjacency measures interpolation, not deployment.",
            "Natural time blocks are class-incomplete; leave-one-block-out is "
            "not a valid four-task scenario split.",
            "The audit cannot estimate a leakage-free accuracy and does not "
            "replace a scenario-held-out test.",
        ],
    }
    with open(os.path.join(RESULTS_DIR, "temporal_leakage_audit.json"),
              "w") as f:
        json.dump(payload, f, indent=2)

    print("Temporal leakage audit (read-only; no models trained)")
    print(summary_df.to_string(index=False))
    print("\nNatural time blocks:")
    print(blocks_df[["block_id", "time_min_rad", "time_max_rad", "n_rows",
                     "n_classes", "class_counts"]].to_string(index=False))
    print("\nWrote:")
    for filename in outputs:
        print(" ", os.path.join(RESULTS_DIR, filename))
    print("  " + os.path.join(RESULTS_DIR, "temporal_leakage_audit.json"))


if __name__ == "__main__":
    main()
