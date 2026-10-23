"""Phase 2e: robustness + quantization + memory ledger (RQ6/RQ7, simulation-only).

Trains CuCD-ID/MLP x [replay, icarl, bic, joint_scratch] x 3 seeds once each
(random split, val_frac=0.20, disjoint calibration/selection, frozen hparams), then evaluates the FINAL head
(with BiC correction when fitted) on corrupted variants of the all-task test
pool. No MCU hardware is available, so every cost below is analytic or
host-simulated and labelled as such (see columns starting SIM_).

Corruptions (applied to scaled features):
  - gauss_0.1 / gauss_0.5 : additive N(0, s^2) telemetry noise
  - drop_0.1 / drop_0.3   : random feature zero-masking (packet loss proxy)
  - nowindow              : zero all SlidingWindow*/InWindow/MessageRate cols
    (tests reliance on the potentially-leaky aggregate features)
  - int8_sim              : per-tensor symmetric int8 round-trip on a model copy
    (simulated post-training INT8 quantization accuracy proxy; NO retraining;
    NOT actual INT8 hardware inference)

Memory ledger: deployed persistent state plus host-estimated peak update RAM
(batch activations, retained validation state, weights/gradients/Adam/teacher).
peak for forward and 2x-forward estimate for training step, per arch.

Writes: results/robustness.csv, results/activation_memory_ledger.csv,
        results/memory_ledger.csv
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
from metrics import compute_cl_metrics
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, ARCHS, VAL_FRAC)

METHODS = ["replay", "icarl", "bic", "joint_scratch"]
SEEDS = [0, 1, 2]
WINDOW_SUBSTR = ("Window", "RateInWindow")


def window_cols(feature_names):
    return [i for i, c in enumerate(feature_names)
            if any(s in c for s in WINDOW_SUBSTR)]


def corrupt(X, kind, rng, wcols):
    Xc = X.copy()
    if kind == "clean":
        return Xc
    if kind.startswith("gauss_"):
        return (Xc + rng.normal(0, float(kind.split("_")[1]), Xc.shape)).astype(np.float32)
    if kind.startswith("drop_"):
        m = rng.random(Xc.shape) < float(kind.split("_")[1])
        Xc[m] = 0.0
        return Xc
    if kind == "nowindow":
        Xc[:, wcols] = 0.0
        return Xc
    raise ValueError(kind)


def quantize_copy(model_cls, model_kwargs, model, seed=0):
    """Per-tensor symmetric int8 round-trip copy (simulates PTQ deployment)."""
    q = model_cls(**model_kwargs, seed=seed)
    for k, v in model.params.items():
        mx = float(np.abs(v).max())
        if mx == 0:
            q.params[k] = v.copy()
            continue
        scale = mx / 127.0
        qi = np.clip(np.round(v / scale), -127, 127).astype(np.float32)
        # grow head first so shapes match
        if k not in q.params or q.params[k].shape != v.shape:
            if k == "head.W":
                q.expand_head(v.shape[1])
            else:
                q.params[k] = np.zeros_like(v)
        q.params[k] = (qi * scale).astype(np.float32)
    return q


def activation_ledger(arch, n_features, n_classes_total=5):
    """Analytic activation floats for batch=1 forward (fp32 count). Documented
    approximations: GRU counts gate pre-activations + states per step; CNN
    counts full conv maps (no pooling reduction modelled — upper bound)."""
    if arch == "MLP":
        fwd = (32 + 32) + (16 + 16) + n_classes_total  # z1,a1,z2,a2,logits
        return {"arch": arch, "act_fwd_floats": fwd,
                "SIM_act_fwd_KB": round(fwd * 4 / 1024, 3),
                "SIM_act_train_KB": round(fwd * 2 * 4 / 1024, 3)}
    if arch == "GRU":
        from config import model_kwargs_for as mkf
        kw = mkf("GRU", n_features)
        S, H, F = kw["seq_len"], kw["hidden"], n_features // kw["seq_len"]
        per_step = 3 * H + 3 * H + H  # gate preact + gate act + state
        fwd = S * per_step + n_classes_total
        return {"arch": arch, "act_fwd_floats": fwd, "seq_len": S,
                "SIM_act_fwd_KB": round(fwd * 4 / 1024, 3),
                "SIM_act_train_KB": round(fwd * 2 * 4 / 1024, 3)}
    if arch == "CNN1D":
        kw = model_kwargs_for("CNN1D", n_features)
        L = n_features
        fwd = L * kw["c1"] * 2 + L * kw["c2"] * 2 + kw["c2"] + n_classes_total
        return {"arch": arch, "act_fwd_floats": fwd,
                "SIM_act_fwd_KB": round(fwd * 4 / 1024, 3),
                "SIM_act_train_KB": round(fwd * 2 * 4 / 1024, 3)}
    raise ValueError(arch)


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


_FROZEN = None
_TS = {}
CORRS = ["clean", "gauss_0.1", "gauss_0.5", "drop_0.1", "drop_0.3",
         "nowindow", "int8_sim"]


def _run_one(item):
    method, seed = item
    ts = _TS[seed]
    arch_hparams = _FROZEN.get("MLP", {}).get(method, {}).get("hparams", {})
    mk = model_kwargs_for("MLP", ts.n_features)
    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
    out = train_continual(MODEL_CLASSES["MLP"], mk, ts, method, hp, seed=seed)
    m = compute_cl_metrics(out["R"])
    Xs, ys, off = [], [], 0
    for task in ts.tasks:
        Xs.append(task.X_test)
        ys.append(task.y_test + off)
        off += task.n_new_classes
    X_all = np.concatenate(Xs)
    y_all = np.concatenate(ys)
    wcols = window_cols(ts.feature_names)
    rng = np.random.default_rng(1000 + seed)
    rows = []
    for corr in CORRS:
        if corr == "int8_sim":
            qmodel = quantize_copy(MODEL_CLASSES["MLP"], mk, out["model"], seed=seed)
            logits, _ = qmodel.forward(X_all)
        else:
            logits, _ = out["model"].forward(corrupt(X_all, corr, rng, wcols))
        corr_obj = out.get("corrector")
        if corr_obj is not None and getattr(corr_obj, "fitted", False):
            logits = corr_obj.apply(logits)
        acc = float((logits.argmax(1) == y_all).mean())
        rows.append(dict(method=method, seed=seed, corruption=corr,
                         ACC_final_pool=round(acc, 4),
                         n_window_cols=len(wcols)))
    return rows, method, seed, m["ACC"]


def main():
    global _FROZEN, _TS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams()
    cucd_df = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                              os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    _TS = {seed: du.build_cucd_id_tasks(
        cucd_df, seed=seed, val_frac=VAL_FRAC) for seed in SEEDS}

    items = [(method, seed) for method in METHODS for seed in SEEDS]
    jobs = get_jobs()
    print(f"Robustness: {len(items)} training runs x {len(CORRS)} corruptions; "
          f"{describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, (new_rows, method, seed, acc) in enumerate(
            parallel_map(_run_one, items, jobs=jobs), 1):
        rows.extend(new_rows)
        print(f"  [{i}/{len(items)}] {method} seed={seed} "
              f"clean-matrix ACC={acc:.3f} ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "robustness.csv"), index=False)
    print("\nMean pooled ACC by (method, corruption):")
    print(df.groupby(["method", "corruption"])["ACC_final_pool"].mean()
          .round(3).to_string())

    n_feat = len(du.build_cucd_id_tasks(cucd_df, seed=0).feature_names)
    led = pd.DataFrame([activation_ledger(a, n_feat) for a in ARCHS])
    led["SIM_act_fwd_KB_batch32"] = (led["act_fwd_floats"] * 4 * 32 / 1024).round(3)
    led["SIM_act_train_KB_batch32"] = (led["act_fwd_floats"] * 2 * 4 * 32 / 1024).round(3)
    led.to_csv(os.path.join(RESULTS_DIR, "activation_memory_ledger.csv"), index=False)
    print("\nActivation ledger (analytic, batch=1 and batch=32):")
    print(led.to_string(index=False))

    main_summary = os.path.join(RESULTS_DIR, "main_results_summary.csv")
    if os.path.exists(main_summary):
        ms = pd.read_csv(main_summary)
        keep = ["dataset", "arch", "method", "persistent_total_KB_mean",
                "retained_val_KB_mean", "peak_total_KB_mean"]
        keep = [c for c in keep if c in ms.columns]
        full = ms[keep].rename(columns={
            "persistent_total_KB_mean": "persistent_total_KB",
            "retained_val_KB_mean": "retained_update_state_KB",
            "peak_total_KB_mean": "SIM_peak_update_KB",
        })
        full.to_csv(os.path.join(RESULTS_DIR, "memory_ledger.csv"), index=False)
        print("\nFull persistent/update memory ledger:")
        print(full.to_string(index=False))
    else:
        print("\nWARNING: main_results_summary.csv missing; full memory_ledger.csv not written.")
    print(f"\nDone ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
