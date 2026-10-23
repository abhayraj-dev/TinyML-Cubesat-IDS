"""
Hyperparameter sweep for the CMMU (learning rate, EWC lambda, replay-buffer
capacity, replay fraction). Performed on STIN (the secondary / stress-test
dataset), using the MLP architecture as a fast proxy, method='full'
(replay+EWC jointly, i.e. the config that will actually be deployed).

Hyperparameters are selected on VALIDATION performance to avoid test-set leakage.
The winning configuration is a COARSE PRIOR only (STIN/MLP/full proxy) --
final per-method x per-architecture hyperparameters come from
experiments/run_validation_tune.py on CuCD-ID validation and are frozen to
results/best_hparams_per_method_arch.json. Nothing here is reused globally.
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
                    BASE_TRAIN_HPARAMS, VAL_FRAC)

GRID = dict(
    lr=[1e-3, 3e-3],
    ewc_lambda=[100, 1000, 5000, 20000],
    buffer_capacity=[64, 128, 256, 512, 1024],
    replay_frac=[0.3, 0.5, 0.7, 0.9],
)


_TS = None
_MK = None


def _run_combo(hp):
    full_hp = {**BASE_TRAIN_HPARAMS, **hp}
    out = train_continual(MODEL_CLASSES["MLP"], _MK, _TS, "full", full_hp, seed=0)
    if out.get("V") is None or np.all(np.isnan(out["V"])):
        raise RuntimeError("No selection rows for STIN sweep")
    m = compute_cl_metrics(out["V"])
    score = m["ACC"] - max(m["FM"], 0.0)
    return {**hp, **m, "score": score}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = du.load_stin(os.path.join(DATA_DIR, "STIN", "SAT20.csv"),
                       os.path.join(DATA_DIR, "STIN", "TER20.csv"))
    ts = du.build_stin_tasks(df, seed=0, val_frac=VAL_FRAC)
    model_kwargs = model_kwargs_for("MLP", ts.n_features)

    rows = []
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    global _TS, _MK
    _TS, _MK = ts, model_kwargs
    items = [dict(zip(keys, combo)) for combo in combos]
    jobs = get_jobs()
    print(f"Running {len(items)} hyperparameter combinations on STIN "
          f"(MLP, method=full, val_frac={VAL_FRAC}); {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_combo, items, jobs=jobs), 1):
        rows.append(row)
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)} done ({time.perf_counter()-t0:.1f}s elapsed)")

    res = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    res.to_csv(os.path.join(RESULTS_DIR, "hparam_sweep_stin.csv"), index=False)

    best = res.iloc[0].to_dict()
    best_hparams = {k: (int(best[k]) if k == "buffer_capacity" else float(best[k])) for k in keys}
    with open(os.path.join(RESULTS_DIR, "best_hparams.json"), "w") as f:
        json.dump(best_hparams, f, indent=2)

    print("\nTop 5 configurations:")
    print(res.head(5).to_string(index=False))
    print("\nSelected optimal hyperparameters (frozen for all further experiments):")
    print(json.dumps(best_hparams, indent=2))
    print(f"\nTotal sweep time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
