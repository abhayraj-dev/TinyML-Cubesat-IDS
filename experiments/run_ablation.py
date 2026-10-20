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
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics
from config import MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR, BASE_TRAIN_HPARAMS, SEEDS


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


def ablation_buffer_size(cucd_df, stin_df, frozen_hparams):
    rows = []
    capacities = [32, 64, 128, 256, 512, 1024]
    for dset_name, df, builder in [("CuCD-ID", cucd_df, du.build_cucd_id_tasks),
                                    ("STIN", stin_df, du.build_stin_tasks)]:
        for arch in ["MLP", "GRU", "CNN1D"]:
            # Use frozen hparams for full/arch
            base_hp = get_frozen_hp(frozen_hparams, arch, "full")
            for cap in capacities:
                for seed in SEEDS:
                    ts = builder(df, seed=seed)
                    mk = model_kwargs_for(arch, ts.n_features)
                    hp = {**BASE_TRAIN_HPARAMS, **base_hp, "buffer_capacity": cap}
                    out = train_continual(MODEL_CLASSES[arch], mk, ts, "full", hp, seed=seed)
                    m = compute_cl_metrics(out["R"])
                    rows.append(dict(dataset=dset_name, arch=arch, buffer_capacity=cap, seed=seed, **m))
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(RESULTS_DIR, "ablation_buffer_size.csv"), index=False)
    print("\n[Ablation A] Buffer capacity sensitivity, all 3 architectures (mean over seeds):")
    print(df_out.groupby(["dataset", "arch", "buffer_capacity"])[["ACC", "BWT", "FM"]].mean().round(4))
    return df_out


def ablation_ewc_lambda(cucd_df, stin_df, frozen_hparams):
    rows = []
    lambdas = [0, 10, 100, 1000, 5000, 20000, 50000]
    for dset_name, df, builder in [("CuCD-ID", cucd_df, du.build_cucd_id_tasks),
                                    ("STIN", stin_df, du.build_stin_tasks)]:
        for arch in ["MLP", "GRU", "CNN1D"]:
            base_hp = get_frozen_hp(frozen_hparams, arch, "full")
            for lam in lambdas:
                for seed in SEEDS:
                    ts = builder(df, seed=seed)
                    mk = model_kwargs_for(arch, ts.n_features)
                    hp = {**BASE_TRAIN_HPARAMS, **base_hp, "ewc_lambda": lam}
                    out = train_continual(MODEL_CLASSES[arch], mk, ts, "full", hp, seed=seed)
                    m = compute_cl_metrics(out["R"])
                    rows.append(dict(dataset=dset_name, arch=arch, ewc_lambda=lam, seed=seed, **m))
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(RESULTS_DIR, "ablation_ewc_lambda.csv"), index=False)
    print("\n[Ablation B] EWC lambda sensitivity, both datasets x 3 architectures (mean over seeds):")
    print(df_out.groupby(["dataset", "arch", "ewc_lambda"])[["ACC", "BWT", "FM"]].mean().round(4))
    return df_out


def main():
    frozen_hparams = load_frozen_hparams()
    cucd_df, stin_df = get_task_seqs()
    ablation_buffer_size(cucd_df, stin_df, frozen_hparams)
    ablation_ewc_lambda(cucd_df, stin_df, frozen_hparams)


if __name__ == "__main__":
    main()
