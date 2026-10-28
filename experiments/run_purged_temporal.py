"""Phase 3: Purged temporal evaluation.

Compares IID (random), the original cold-start forward-temporal protocol, and
a fair forward-temporal protocol that draws train/validation from the purged
past while locking test in the future.

Protocols:
- iid: random stratified split (benchmark comparability)
- temporal_purged: original cold-start earliest-head train sample, future test
- temporal_purged_fair: random-past train and validation, 70/10/20 split,
  future test with purge
- temporal_purged_no_leaky_features: cold-start with time/ID/window dropped
- temporal_purged_fair_no_leaky_features: fair split with leaky groups dropped

Output:
- results/temporal_purged_compare.csv (all architectures)
- results/temporal_purged_compare_{arch}.csv (per architecture)
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
from train import train_continual
from metrics import compute_cl_metrics, compute_detection_metrics
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, VAL_FRAC, load_frozen_hparams_strict)

METHODS = ["finetune", "replay", "ewc", "full", "lwf", "bic", "icarl",
           "joint", "joint_scratch"]
ARCHS = ["MLP", "GRU", "CNN1D"]
SEEDS = list(range(15))
PURGE_TIME_RAD = 0.02
LEAKY_FEATURES = [
    "TimeRadians", "MsgId", "CmdCode", "ApId",
    "FlowLengthInWindow", "SlidingWindowMeanIntervalSec",
    "SlidingWindowMaxIntervalSec", "SlidingWindowMinIntervalSec",
    "MessageCountInWindow", "UniqueMessageIDsInWindow",
    "AverageMessageLengthInWindow", "MessageRateInWindow",
    "StdDevMessageLengthInWindow",
]
PROTOCOLS = [
    dict(name="iid", builder="iid", test_frac=0.30, val_frac=VAL_FRAC),
    dict(name="temporal_purged", builder="temporal",
         train_sample_mode="earliest", val_mode="block",
         test_frac=0.30, val_frac=VAL_FRAC),
    dict(name="temporal_purged_fair", builder="temporal",
         train_sample_mode="random_past", val_mode="random_past",
         test_frac=0.20, val_frac=0.10),
    dict(name="temporal_purged_no_leaky_features", builder="temporal",
         train_sample_mode="earliest", val_mode="block",
         test_frac=0.30, val_frac=VAL_FRAC,
         feature_drop=LEAKY_FEATURES),
    dict(name="temporal_purged_fair_no_leaky_features", builder="temporal",
         train_sample_mode="random_past", val_mode="random_past",
         test_frac=0.20, val_frac=0.10, feature_drop=LEAKY_FEATURES),
]
_FROZEN = None
_TS = {}


def final_detection_metrics(ts, out):
    if ts.dataset_name != "CuCD-ID":
        return {"DET_AUROC": float("nan"), "DET_AUPR": float("nan"),
                "DET_FPR95": float("nan"), "DET_PREV": float("nan")}
    try:
        normal_idx = list(ts.class_names_global).index("normal")
    except ValueError:
        return {"DET_AUROC": float("nan"), "DET_AUPR": float("nan"),
                "DET_FPR95": float("nan"), "DET_PREV": float("nan")}
    Xs, ys, off = [], [], 0
    for task in ts.tasks:
        Xs.append(task.X_test)
        ys.append(task.y_test + off)
        off += task.n_new_classes
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    logits, _ = out["model"].forward(X_all)
    corr = out.get("corrector")
    if corr is not None and getattr(corr, "fitted", False):
        logits = corr.apply(logits)
    return compute_detection_metrics(y_all, logits, normal_idx)


def _build_tasks(df, protocol, seed):
    cfg = next(p for p in PROTOCOLS if p["name"] == protocol)
    if cfg["builder"] == "iid":
        return du.build_cucd_id_tasks(
            df, seed=seed, test_frac=cfg["test_frac"],
            val_frac=cfg["val_frac"])
    return du.build_cucd_id_tasks_purged_temporal(
        df, seed=seed, test_frac=cfg["test_frac"],
        val_frac=cfg["val_frac"], purge_time_rad=PURGE_TIME_RAD,
        feature_drop=cfg.get("feature_drop"),
        train_sample_mode=cfg["train_sample_mode"],
        val_mode=cfg["val_mode"])


def _run_one(item):
    arch, method, protocol, seed = item
    ts = _TS[(protocol, seed)]
    mk = model_kwargs_for(arch, ts.n_features)
    arch_hparams = _FROZEN.get(arch, {}).get(method, {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
    out = train_continual(
        MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    det = final_detection_metrics(ts, out)
    cfg = next(p for p in PROTOCOLS if p["name"] == protocol)
    return dict(
        arch=arch, method=method, protocol=protocol, seed=seed,
        n_features=ts.n_features,
        train_sample_mode=ts.train_sample_mode,
        val_sample_mode=ts.val_sample_mode,
        test_frac=cfg["test_frac"],
        purge_time_rad=PURGE_TIME_RAD if protocol != "iid" else 0.0,
        **m, **det)


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="all", choices=["all"] + ARCHS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--prefix", default="temporal_purged_compare")
    return ap.parse_args()


def main():
    global _FROZEN, _TS
    args = _parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams_strict()
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))

    if "-" in args.seeds:
        lo, hi = [int(x) for x in args.seeds.split("-")]
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(args.seeds)]
    archs = ARCHS if args.arch == "all" else [args.arch]
    methods = [m for m in args.methods if m in METHODS]

    for protocol in [p["name"] for p in PROTOCOLS]:
        for seed in seeds:
            _TS[(protocol, seed)] = _build_tasks(cucd_df, protocol, seed)

    items = [(arch, method, protocol, seed)
             for arch in archs
             for method in methods
             for protocol in [p["name"] for p in PROTOCOLS]
             for seed in seeds]
    jobs = get_jobs(args.jobs)
    print(f"Purged temporal: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 20 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['arch']} {row['method']} "
                  f"{row['protocol']} seed={row['seed']} "
                  f"ACC={row['ACC']:.3f} ({time.perf_counter() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    output_path = os.path.join(RESULTS_DIR, f"{args.prefix}.csv")
    df.to_csv(output_path, index=False)
    for arch in archs:
        sub = df[df.arch == arch]
        sub.to_csv(
            os.path.join(RESULTS_DIR, f"{args.prefix}_{arch}.csv"), index=False)
    print("\nMean ACC by (arch, method, protocol):")
    print(df.groupby(["arch", "method", "protocol"])["ACC"]
          .agg(["mean", "std"]).round(3).to_string())
    print(f"\nWrote {output_path} ({len(df)} runs, "
          f"{time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
