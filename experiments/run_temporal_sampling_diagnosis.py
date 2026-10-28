"""Diagnose the temporal split's train-sampling sensitivity.

Compares the original earliest-head temporal sample with a random-past sample
while keeping the future test block and purge gap fixed. The random-past mode
draws the same per-class budget from the whole past pool, so it isolates
cold-start sample support from temporal shift itself.

Usage:
    python3 experiments/run_temporal_sampling_diagnosis.py \
        --arch MLP --seeds 0 1 2 --methods joint_scratch bic finetune --jobs 3
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from parallel import configure_single_thread, describe, get_jobs, parallel_map
configure_single_thread()

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics, compute_macro_f1_balanced_acc
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, VAL_FRAC, load_frozen_hparams_strict)

PURGE_TIME_RAD = 0.02
PROTOCOLS = [
    ("iid", "iid", None, VAL_FRAC, 0.30, "random"),
    ("temporal_purged_earliest", "temporal_purged", "earliest",
     VAL_FRAC, 0.30, "block"),
    ("temporal_purged_random_past", "temporal_purged", "random_past",
     VAL_FRAC, 0.30, "block"),
    ("temporal_purged_random_past_noval", "temporal_purged", "random_past",
     0.0, 0.30, "random_past"),
    ("temporal_purged_earliest_70", "temporal_purged", "earliest",
     0.10, 0.20, "block"),
    ("temporal_purged_earliest_70_randval", "temporal_purged", "earliest",
     0.10, 0.20, "random_past"),
    ("temporal_purged_random_past_70", "temporal_purged", "random_past",
     0.10, 0.20, "random_past"),
]
LEAKY_FEATURES = {
    "TimeRadians", "MsgId", "CmdCode", "ApId",
    "FlowLengthInWindow", "SlidingWindowMeanIntervalSec",
    "SlidingWindowMaxIntervalSec", "SlidingWindowMinIntervalSec",
    "MessageCountInWindow", "UniqueMessageIDsInWindow",
    "AverageMessageLengthInWindow", "MessageRateInWindow",
    "StdDevMessageLengthInWindow",
}

_FROZEN = None
_TS = {}
_ARCH = "MLP"


def _build_tasks(df, seed):
    ts = {}
    for (protocol, split_mode, sample_mode, val_frac, test_frac,
         val_mode) in PROTOCOLS:
        if split_mode == "iid":
            ts[protocol] = du.build_cucd_id_tasks(
                df, seed=seed, val_frac=val_frac, test_frac=test_frac)
        else:
            ts[protocol] = du.build_cucd_id_tasks_purged_temporal(
                df, seed=seed, val_frac=val_frac, test_frac=test_frac,
                purge_time_rad=PURGE_TIME_RAD,
                train_sample_mode=sample_mode, val_mode=val_mode)
    return ts


def _global_arrays(ts):
    names = list(ts.class_names_global)
    global_id = {name: i for i, name in enumerate(names)}
    X_train, y_train, X_test, y_test = [], [], [], []
    for task in ts.tasks:
        X_train.append(task.X_train)
        y_train.append(np.array(
            [global_id[task.class_names[c]] for c in task.y_train]))
        X_test.append(task.X_test)
        y_test.append(np.array(
            [global_id[task.class_names[c]] for c in task.y_test]))
    return (np.concatenate(X_train), np.concatenate(y_train),
            np.concatenate(X_test), np.concatenate(y_test))


def _probe_features(X_train, X_test, ts, use_no_leaky):
    keep = np.arange(len(ts.feature_names))
    if use_no_leaky:
        keep = np.array([i for i, name in enumerate(ts.feature_names)
                         if name not in LEAKY_FEATURES])
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]
    mean = ts.scaler_mean[keep]
    std = ts.scaler_std[keep]
    return (X_train - mean) / std, (X_test - mean) / std


def _ridge_probe(X_train, y_train, X_test, y_test, lam=1.0):
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    A = np.c_[np.ones(len(X_train)), X_train]
    B = np.c_[np.ones(len(X_test)), X_test]
    Y = np.eye(n_classes)[y_train]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    pred = (B @ W).argmax(axis=1)
    acc = float((pred == y_test).mean())
    recalls = [float((pred[y_test == c] == c).mean())
               for c in range(n_classes) if np.any(y_test == c)]
    return acc, float(np.mean(recalls)) if recalls else float("nan")


def _knn_probe(X_train, y_train, X_test, y_test, chunk=256):
    pred = []
    for i in range(0, len(X_test), chunk):
        dist = ((X_test[i:i + chunk, None, :] - X_train[None, :, :]) ** 2).sum(-1)
        pred.append(y_train[dist.argmin(axis=1)])
    pred = np.concatenate(pred)
    return float((pred == y_test).mean())


def _probe_row(protocol, seed, ts):
    X_train, y_train, X_test, y_test = _global_arrays(ts)
    rows = []
    for feature_set, no_leaky in (("all", False), ("no_leaky", True)):
        A, B = _probe_features(
            X_train.astype(np.float64), X_test.astype(np.float64),
            ts, no_leaky)
        ridge_acc, ridge_ba = _ridge_probe(A, y_train, B, y_test)
        knn_acc = _knn_probe(A, y_train, B, y_test)
        rows.append(dict(
            protocol=protocol, seed=seed, feature_set=feature_set,
            train_sample_mode=ts.train_sample_mode,
            val_sample_mode=ts.val_sample_mode,
            n_train=len(y_train), n_test=len(y_test),
            ridge_acc=ridge_acc, ridge_ba=ridge_ba, knn_acc=knn_acc))
    return rows


def _run_one(item):
    arch, method, protocol, seed = item
    ts = _TS[(protocol, seed)]
    mk = model_kwargs_for(arch, ts.n_features)
    arch_hparams = _FROZEN.get(arch, {}).get(method, {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
    out = train_continual(
        MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed)
    metrics = compute_cl_metrics(out["R"])
    final = compute_macro_f1_balanced_acc(
        out["model"], ts, corrector=out.get("corrector"),
        ncm=out.get("ncm"))
    return dict(
        arch=arch, method=method, protocol=protocol, seed=seed,
        train_sample_mode=ts.train_sample_mode,
        val_sample_mode=ts.val_sample_mode,
        purge_time_rad=PURGE_TIME_RAD if protocol != "iid" else 0.0,
        **metrics, **final)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="MLP", choices=sorted(MODEL_CLASSES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+",
                        default=["joint_scratch", "bic", "finetune"])
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--prefix", default="temporal_sampling")
    return parser.parse_args()


def main():
    global _FROZEN, _TS, _ARCH
    args = _parse_args()
    _ARCH = args.arch
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams_strict()
    df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))

    for seed in args.seeds:
        for protocol, ts in _build_tasks(df, seed).items():
            _TS[(protocol, seed)] = ts

    probe_rows = []
    for seed in args.seeds:
        for protocol, *_ in PROTOCOLS:
            probe_rows.extend(_probe_row(protocol, seed, _TS[(protocol, seed)]))
    probe_path = os.path.join(RESULTS_DIR, f"{args.prefix}_probes.csv")
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(probe_path, index=False)
    print("\nProbe summary (mean over seeds):")
    print(probe_df.groupby(["protocol", "feature_set"])[
        ["ridge_acc", "ridge_ba", "knn_acc"]].mean().round(3).to_string())
    print(f"\nWrote {probe_path} ({len(probe_df)} rows)")

    if args.probe_only:
        return

    items = [(args.arch, method, protocol, seed)
             for method in args.methods
             for protocol, *_ in PROTOCOLS
             if not (method == "bic" and protocol.endswith("_noval"))
             for seed in args.seeds]
    jobs = get_jobs(args.jobs)
    print(f"\nTemporal sampling diagnosis: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 5 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['method']} {row['protocol']} "
                  f"seed={row['seed']} ACC={row['ACC']:.3f} "
                  f"({time.perf_counter() - t0:.1f}s)")

    df_out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f"{args.prefix}_diagnosis.csv")
    df_out.to_csv(path, index=False)
    print("\nMean metrics by (method, protocol):")
    print(df_out.groupby(["method", "protocol"])[
        ["ACC", "BalancedAccuracy", "FM"]].mean().round(3).to_string())
    print(f"\nWrote {path} ({len(df_out)} runs, "
          f"{time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
