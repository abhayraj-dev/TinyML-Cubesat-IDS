"""
Ablation study, run on top of the FROZEN per-method/arch hyperparameters
from validation tuning (not the single sweep winners). This ensures each
method/architecture combination uses its own optimized configuration.

  (a) Replay buffer capacity sensitivity   -- CuCD-ID + STIN, all 3
                                                architectures, method=full
  (b) EWC lambda sensitivity                -- CuCD-ID + STIN, all 3
                                                architectures, method=full

NOTE: the legacy 3-permutation task-order ablation was removed here; the
disentangled 12-permutation study (`run_task_order.py`,
`task_order_disentangled.csv`) supersedes it (see results/legacy/README.md).
"""
import sys, os, json, itertools, time
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
                    BASE_TRAIN_HPARAMS, SEEDS, VAL_FRAC)


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


def get_task_seqs():
    cucd_df = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                               os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    stin_df = du.load_stin(os.path.join(DATA_DIR, "STIN", "SAT20.csv"),
                           os.path.join(DATA_DIR, "STIN", "TER20.csv"))
    return cucd_df, stin_df


def get_frozen_hp(frozen_hparams, arch, method):
    """Get frozen hyperparameters for a specific arch/method."""
    return frozen_hparams.get(arch, {}).get(method, {}).get("hparams", {})


CAPACITIES = [32, 64, 128, 256, 512, 1024]
LAMBDAS = [0, 10, 100, 1000, 5000, 20000, 50000]
ARCHS_ABL = ["MLP", "GRU", "CNN1D"]
_FROZEN = None
_TS = {}


def _run_one(item):
    dset_name, arch, kind, value, seed = item
    ts = _TS[(dset_name, seed)]
    base_hp = get_frozen_hp(_FROZEN, arch, "full")
    if kind == "buffer":
        hp = {**BASE_TRAIN_HPARAMS, **base_hp, "buffer_capacity": value}
        row_key = {"buffer_capacity": value}
    else:
        hp = {**BASE_TRAIN_HPARAMS, **base_hp, "ewc_lambda": value}
        row_key = {"ewc_lambda": value}
    mk = model_kwargs_for(arch, ts.n_features)
    out = train_continual(MODEL_CLASSES[arch], mk, ts, "full", hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    return dict(dataset=dset_name, arch=arch, seed=seed, **row_key, **m)


def main():
    global _FROZEN, _TS
    _FROZEN = load_frozen_hparams()
    cucd_df, stin_df = get_task_seqs()
    for seed in SEEDS:
        _TS[("CuCD-ID", seed)] = du.build_cucd_id_tasks(
            cucd_df, seed=seed, val_frac=VAL_FRAC)
        _TS[("STIN", seed)] = du.build_stin_tasks(
            stin_df, seed=seed, val_frac=VAL_FRAC)

    items = (
        [(dset, arch, "buffer", cap, seed)
         for dset in ("CuCD-ID", "STIN")
         for arch in ARCHS_ABL
         for cap in CAPACITIES
         for seed in SEEDS]
        + [(dset, arch, "ewc", lam, seed)
           for dset in ("CuCD-ID", "STIN")
           for arch in ARCHS_ABL
           for lam in LAMBDAS
           for seed in SEEDS]
    )
    jobs = get_jobs()
    print(f"Ablations: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 30 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['dataset']} {row['arch']} "
                  f"({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    buf = df[df.get("buffer_capacity").notna()].copy()
    ewc = df[df.get("ewc_lambda").notna()].copy()
    buf.to_csv(os.path.join(RESULTS_DIR, "ablation_buffer_size.csv"), index=False)
    ewc.to_csv(os.path.join(RESULTS_DIR, "ablation_ewc_lambda.csv"), index=False)

    print("\n[Ablation A] Buffer capacity sensitivity, all 3 architectures "
          "(mean over seeds):")
    print(buf.groupby(["dataset", "arch", "buffer_capacity"])[["ACC", "BWT", "FM"]]
          .mean().round(4))
    print("\n[Ablation B] EWC lambda sensitivity, both datasets x 3 architectures "
          "(mean over seeds):")
    print(ewc.groupby(["dataset", "arch", "ewc_lambda"])[["ACC", "BWT", "FM"]]
          .mean().round(4))
    print(f"\nDone ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
