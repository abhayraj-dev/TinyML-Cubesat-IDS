"""Phase 2a: per-architecture LR/epoch tune (CNN flag).

Motivation: full-matrix evidence shows CNN `joint_scratch < joint(warm)` on
both datasets (CuCD-ID 0.858 vs 0.898; STIN 0.395 vs 0.535) while MLP/GRU
have scratch >= joint. That pattern points at optimization (LR/epochs for
the conv trunk), not at continual learning. The frozen STIN/MLP hparams
(lr=0.003, epochs=25) were never tuned for CNN.

Scope (kept small on purpose): CuCD-ID only, CNN1D only, methods
replay/icarl/joint_scratch (replay = continual reference, scratch = upper
bound whose gap must close under a fair tune). 3 seeds for speed.
EXPLORATORY ONLY: selection uses validation (V) metrics; winners are NOT
applied to reported numbers (per-method/arch frozen configs govern).

Writes: results/arch_tune.csv (one row per lr x epochs x method x seed).
"""
import sys
import os
import itertools
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS)


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()

LR_GRID = [1e-3, 3e-3, 1e-2]
EPOCH_GRID = [25, 50]
METHODS = ["replay", "icarl", "joint_scratch"]
SEEDS = [0, 1, 2]
VAL_FRAC = 0.10


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frozen_hparams = load_frozen_hparams()
    cucd_df = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                              os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    rows = []
    t0 = time.perf_counter()
    total = len(LR_GRID) * len(EPOCH_GRID) * len(METHODS) * len(SEEDS)
    done = 0
    for lr, epochs, method in itertools.product(LR_GRID, EPOCH_GRID, METHODS):
        base_hp = frozen_hparams.get("CNN1D", {}).get(method, {}).get("hparams", {})
        for seed in SEEDS:
            ts = du.build_cucd_id_tasks(cucd_df, seed=seed, val_frac=VAL_FRAC)
            mk = model_kwargs_for("CNN1D", ts.n_features)
            hp = {**BASE_TRAIN_HPARAMS, **base_hp, "lr": lr, "epochs": epochs}
            out = train_continual(MODEL_CLASSES["CNN1D"], mk, ts, method, hp, seed=seed)
            m_src = out["V"] if out.get("V") is not None else out["R"]
            m = compute_cl_metrics(m_src)
            rows.append(dict(lr=lr, epochs=epochs, method=method, seed=seed, **m))
            done += 1
            if done % 9 == 0:
                print(f"  [{done}/{total}] lr={lr} epochs={epochs} {method} "
                      f"seed={seed} ACC={m['ACC']:.3f} ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "arch_tune.csv"), index=False)
    print("\nMean ACC per (method, lr, epochs):")
    print(df.groupby(["method", "lr", "epochs"])["ACC"].mean().round(3).to_string())
    print(f"\nWrote results/arch_tune.csv ({len(df)} runs, {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
