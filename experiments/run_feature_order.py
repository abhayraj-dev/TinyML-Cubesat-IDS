"""Phase 8: Feature-order permutation stability test (corrected).

4 feature permutations x 3 archs x 4 methods x 2 seeds on CuCD-ID.
BiC-corrected Macro-F1/Balanced-Acc are now evaluated with the fitted
corrector, matching the main matrix.  Parallelised over independent runs.

Output: results/feature_order_stability.csv
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
                    BASE_TRAIN_HPARAMS, ARCHS, VAL_FRAC)
from run_main import final_detection_metrics

METHODS = ["replay", "full", "icarl", "bic"]
SEEDS = [0, 1]
_FROZEN = None
_TS = {}


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


def _run_one(item):
    arch, perm_idx, method, seed = item
    ts = _TS[perm_idx]
    mk = model_kwargs_for(arch, ts.n_features)
    hp = {**BASE_TRAIN_HPARAMS,
          **_FROZEN.get(arch, {}).get(method, {}).get("hparams", {})}
    out = train_continual(MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    mf_ba = compute_macro_f1_balanced_acc(
        out["model"], ts, out.get("corrector"))
    det = final_detection_metrics(ts, out)
    return dict(
        arch=arch, perm_idx=perm_idx, method=method, seed=seed,
        is_original=(perm_idx == 0),
        **m, **mf_ba, DET_AUROC=det["DET_AUROC"])


def main():
    global _FROZEN, _TS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams()

    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    ts_base = du.build_cucd_id_tasks(
        cucd_df, seed=0, val_frac=VAL_FRAC)
    n_features = ts_base.n_features

    rng = np.random.default_rng(42)
    perms = [np.arange(n_features)]
    for _ in range(3):
        perms.append(rng.permutation(n_features))

    for perm_idx, perm in enumerate(perms):
        ts = du.build_cucd_id_tasks(
            cucd_df, seed=0, val_frac=VAL_FRAC)
        for task in ts.tasks:
            du.permute_task_features(task, perm)
        _TS[perm_idx] = ts

    items = [(arch, perm_idx, method, seed)
             for arch in ARCHS
             for perm_idx in range(len(perms))
             for method in METHODS
             for seed in SEEDS]
    jobs = get_jobs()
    print(f"Feature-order stability: {len(items)} runs; {describe(jobs)}")
    t_start = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 24 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['arch']} perm={row['perm_idx']} "
                  f"{row['method']} seed={row['seed']} ACC={row['ACC']:.3f} "
                  f"({time.perf_counter()-t_start:.1f}s)")

    df = pd.DataFrame(rows)
    output_path = os.path.join(RESULTS_DIR, "feature_order_stability.csv")
    df.to_csv(output_path, index=False)

    print("\n=== Feature Order Stability Summary (BiC-corrected) ===")
    for arch in ARCHS:
        arch_df = df[df.arch == arch]
        orig = arch_df[arch_df.is_original].groupby("method")["ACC"].mean()
        permd = arch_df[~arch_df.is_original].groupby("method")["ACC"].mean()
        print(f"\n{arch}:")
        for method in METHODS:
            if method in orig.index and method in permd.index:
                drop = orig[method] - permd[method]
                print(f"  {method:10s}: orig={orig[method]:.3f} "
                      f"perm={permd[method]:.3f} drop={drop:+.3f}")

    print(f"\nWrote {output_path} ({len(df)} runs, "
          f"{time.perf_counter()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
