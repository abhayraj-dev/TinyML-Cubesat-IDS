"""Phase 2b: few-shot uplink sweep (RQ4).

Question: how many labelled samples per new attack must be uplinked for a
reliable onboard update? Varies the uplink budget K (samples/class for
T1..T3; T0 pre-launch corpus fixed at 700/class) on CuCD-ID, the primary
CubeSat dataset. Records Macro-F1, BWT, FM, BalancedAcc + detection AUROC +
exact uplink KB (from Task.uplink_bytes_int8) so the paper can draw the
KB-vs-accuracy Pareto front.

Design: MLP x [replay, full, icarl, bic] x 5 seeds, plus GRU x [replay, icarl] x 5 seeds.
Uses FROZEN per-method/arch hyperparameters from validation tuning (no test leakage).
val_frac=0.10 throughout, so K=150 column reproduces the main matrix.

Writes: results/fewshot.csv (one row per arch x method x K x seed).
"""
import sys
import os
import json
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import compute_cl_metrics, compute_macro_f1_balanced_acc
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS)
from run_main import final_detection_metrics

K_GRID = [5, 10, 25, 50, 100, 150, 300]
PLAN = [
    ("MLP", ["replay", "full", "icarl", "bic"], list(range(5))),
    ("GRU", ["replay", "icarl"], list(range(5))),
]
VAL_FRAC = 0.10
T0_N = 700


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frozen_hparams = load_frozen_hparams()
    cucd_df = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                              os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    rows = []
    t0 = time.perf_counter()
    total = sum(len(m) * len(K_GRID) * len(s) for _, m, s in PLAN)
    done = 0
    for arch, methods, seeds in PLAN:
        for K in K_GRID:
            for method in methods:
                # Get frozen hparams for this arch/method
                arch_hparams = frozen_hparams.get(arch, {}).get(method, {}).get("hparams", {})
                for seed in seeds:
                    ts = du.build_cucd_id_tasks(cucd_df, seed=seed, val_frac=VAL_FRAC,
                                                initial_n_per_class=T0_N,
                                                uplink_n_per_class=K)
                    mk = model_kwargs_for(arch, ts.n_features)
                    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
                    out = train_continual(MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed)
                    m = compute_cl_metrics(out["R"])
                    mf_ba = compute_macro_f1_balanced_acc(out["model"], ts, out.get("corrector"))
                    det = final_detection_metrics(ts, out)
                    uplink_kb = float(sum(t.uplink_bytes_int8 for t in ts.tasks[1:]) / 1024.0)
                    calib_kb = float(sum(t.uplink_calib_bytes_int8 for t in ts.tasks[1:]) / 1024.0)
                    rows.append(dict(arch=arch, method=method, K=K, seed=seed,
                                     uplink_KB_total=round(uplink_kb, 2),
                                     uplink_KB_calib=round(calib_kb, 2),
                                     uplink_KB_total_with_calib=round(uplink_kb + calib_kb, 2),
                                     **m, **mf_ba,
                                     DET_AUROC=det["DET_AUROC"]))
                    done += 1
                    if done % 20 == 0:
                        print(f"  [{done}/{total}] {arch} {method} K={K} seed={seed} "
                              f"ACC={m['ACC']:.3f} ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "fewshot.csv"), index=False)
    print("\nMean ACC per (arch, method, K):")
    print(df.groupby(["arch", "method", "K"])["ACC"].mean().round(3).to_string())
    print(f"\nWrote results/fewshot.csv ({len(df)} runs, {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
