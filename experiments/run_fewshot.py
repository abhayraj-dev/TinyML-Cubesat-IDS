"""Phase 2b: few-shot uplink sweep (RQ4), confirmatory protocol.

Question: how many labelled samples per new attack must be uplinked for a
reliable onboard update?  Varies the adaptation budget K (samples/class for
T1..T3; T0 pre-launch corpus fixed at 700/class) on CuCD-ID.

Two calibration protocols are reported:
  * fixed  -- 150 calibration rows/class, 150 disjoint selection rows/class
              (the deployment payload for calibration-based methods).
  * scaled -- K calibration rows/class, K disjoint selection rows/class,
              so calibration and adaptation budgets grow together.

Calibration payload is counted only for methods that consume it
(ewc/full/bic).  Replay and iCaRL are calibration-free and no longer carry a
phantom 10 KB calibration charge.  The selection half is used only for
hyperparameter scoring and is never counted as onboard payload.

Writes: results/fewshot.csv (one row per protocol x arch x method x K x seed).
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
                    BASE_TRAIN_HPARAMS, VAL_FRAC, VAL_CAP_PER_CLASS,
                    CALIB_CAP_PER_CLASS)
from run_main import final_detection_metrics

K_GRID = [5, 10, 25, 50, 100, 150, 300]
PROTOCOLS = ["fixed", "scaled"]
CALIB_METHODS = {"ewc", "full", "bic"}
PLAN = [
    ("MLP", ["replay", "full", "icarl", "bic"], list(range(5))),
    ("GRU", ["replay", "icarl"], list(range(5))),
]
T0_N = 700

_FROZEN = None
_TS = {}


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


def _build_tasks(cucd_df):
    """Pre-build all task sequences before forking so workers share them."""
    global _TS
    _TS = {}
    for protocol in PROTOCOLS:
        for K in K_GRID:
            val_cap = VAL_CAP_PER_CLASS if protocol == "fixed" else 2 * K
            for seed in range(5):
                _TS[(protocol, K, seed)] = du.build_cucd_id_tasks(
                    cucd_df, seed=seed, val_frac=VAL_FRAC,
                    initial_n_per_class=T0_N,
                    uplink_n_per_class=K,
                    val_cap_per_class=val_cap)


def _run_one(item):
    arch, method, protocol, K, seed = item
    ts = _TS[(protocol, K, seed)]
    mk = model_kwargs_for(arch, ts.n_features)
    hp = {**_BASE_HP, **_FROZEN.get(arch, {}).get(method, {}).get("hparams", {})}
    out = train_continual(MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    mf_ba = compute_macro_f1_balanced_acc(
        out["model"], ts, out.get("corrector"))
    det = final_detection_metrics(ts, out)
    adaptation_kb = float(sum(t.uplink_bytes_int8 for t in ts.tasks[1:]) / 1024.0)
    uses_calib = method in CALIB_METHODS
    calib_kb = (float(sum(t.uplink_calib_bytes_int8 for t in ts.tasks[1:]) / 1024.0)
                if uses_calib else 0.0)
    return dict(
        calibration_protocol=protocol,
        calib_cap_per_class=(CALIB_CAP_PER_CLASS if protocol == "fixed" else K),
        uses_calibration=uses_calib,
        arch=arch, method=method, K=K, seed=seed,
        uplink_KB_total=round(adaptation_kb, 2),
        uplink_KB_calib=round(calib_kb, 2),
        uplink_KB_total_with_calib=round(adaptation_kb + calib_kb, 2),
        **m, **mf_ba,
        DET_AUROC=det["DET_AUROC"],
    )


_BASE_HP = {}


def main():
    global _FROZEN, _BASE_HP
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams()
    _BASE_HP = dict(BASE_TRAIN_HPARAMS)

    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    _build_tasks(cucd_df)

    items = []
    for protocol in PROTOCOLS:
        for arch, methods, seeds in PLAN:
            for K in K_GRID:
                for method in methods:
                    for seed in seeds:
                        items.append((arch, method, protocol, K, seed))

    jobs = get_jobs()
    print(f"Few-shot sweep: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, row in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        if i % 20 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['calibration_protocol']} "
                  f"{row['arch']} {row['method']} K={row['K']} seed={row['seed']} "
                  f"ACC={row['ACC']:.3f} ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "fewshot.csv"), index=False)
    print("\nMean ACC per (protocol, arch, method, K):")
    print(df.groupby(["calibration_protocol", "arch", "method", "K"])["ACC"]
          .mean().round(3).to_string())
    print(f"\nWrote results/fewshot.csv ({len(df)} runs, "
          f"{time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
