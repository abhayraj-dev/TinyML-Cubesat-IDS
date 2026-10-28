"""Budget-Aware Update Admission Control (BUAC) for CubeSat CIL updates.

The gate decides, at each task boundary, whether a candidate update is applied
onboard, deferred to a ground-assisted update, or rejected.  It uses only
signals available before test labels exist: selection-half gain/retention,
temporal proximity to previously seen training data, feature drift, and the
update's memory/uplink cost.

Protocols: IID, cold-start temporal (earliest-head train), and fair
random-past temporal (70/10/20; future test). The gate is trained on IID
tuning seeds and evaluated on held-out seeds; the fair protocol is the
non-trivial admission test because cold-start rejects every update.

Outputs (per run prefix):
  buac_stage_signals.csv
  buac_gate_eval.csv
  buac_gate_by_method.csv
  buac_gate_coefficients.csv
  buac_budget_sensitivity.csv
  buac_decisions.csv
"""
from __future__ import annotations

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
from metrics import model_peak_training_ram
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, VAL_FRAC, load_frozen_hparams_strict)

METHODS = ["finetune", "replay", "ewc", "full", "lwf", "bic"]
PROTOCOLS = ["iid", "temporal_purged", "temporal_purged_fair"]
PROTOCOL_CFG = {
    "iid": dict(builder="iid", test_frac=0.30, val_frac=VAL_FRAC),
    "temporal_purged": dict(
        builder="temporal", test_frac=0.30, val_frac=VAL_FRAC,
        train_sample_mode="earliest", val_mode="block"),
    "temporal_purged_fair": dict(
        builder="temporal", test_frac=0.20, val_frac=0.10,
        train_sample_mode="random_past", val_mode="random_past"),
}
CALIB_METHODS = {"ewc", "full", "bic"}
PURGE_TIME_RAD = 0.02
SAFE_ACC = 0.50
SAFE_FORGET = 0.15
MEM_BUDGET_KB = 256.0
UP_BUDGET_KB = 64.0
GATE_FEATURES = ["g_new_select", "g_old_forget_select", "r_fwd_flag", "r_drift_cur_log"]

_FROZEN = None
_TS = {}


def _raw_features(X: np.ndarray, ts) -> np.ndarray:
    return X * ts.scaler_std[None, :] + ts.scaler_mean[None, :]


def _time_idx(ts) -> int:
    return ts.feature_names.index("TimeRadians")


def _circular_median_nearest(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d = np.abs(a[:, None] - b[None, :])
    d = np.minimum(d, 2.0 * np.pi - d)
    return float(np.median(d.min(axis=1)))


def _proximity(ts, t: int, task) -> float:
    if t == 0 or task.X_select is None:
        return 0.0
    raw_sel = _raw_features(task.X_select, ts)
    old = np.concatenate([tt.X_train for tt in ts.tasks[:t]], axis=0)
    raw_old = _raw_features(old, ts)
    ti = _time_idx(ts)
    return _circular_median_nearest(raw_sel[:, ti], raw_old[:, ti]) / np.pi


def _current_temporal_gap(ts, task) -> tuple[float, float]:
    if task.X_select is None:
        return 0.0, 0.0
    raw_train = _raw_features(task.X_train, ts)
    raw_sel = _raw_features(task.X_select, ts)
    ti = _time_idx(ts)
    gap = _circular_median_nearest(raw_sel[:, ti], raw_train[:, ti]) / np.pi
    max_train = float(np.max(raw_train[:, ti]))
    d = (raw_sel[:, ti] - max_train) % (2.0 * np.pi)
    fwd = float(np.mean((d > 1e-6) & (d < np.pi)))
    return gap, fwd


def _drift(ts, task, reference: str) -> tuple[float, float]:
    if task.X_select is None:
        return 0.0, 0.0
    raw_sel = _raw_features(task.X_select, ts)
    if reference == "current":
        raw_ref = _raw_features(task.X_train, ts)
    else:
        raise ValueError(reference)
    skip = {"TimeRadians", "MsgId", "CmdCode", "ApId"}
    vals = []
    for i, name in enumerate(ts.feature_names):
        if name in skip:
            continue
        sd = float(np.std(raw_ref[:, i]))
        if sd < 1e-9:
            continue
        vals.append(abs(float(np.mean(raw_sel[:, i]) - np.mean(raw_ref[:, i]))) / sd)
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.max(vals))


def _safe_label(R: np.ndarray, t: int) -> tuple[float, float, int]:
    post = float(np.nanmean(R[t, :t + 1]))
    if t == 0:
        return post, 0.0, int(post >= SAFE_ACC)
    drops = [R[t - 1, j] - R[t, j] for j in range(t)]
    forget = float(np.mean(np.maximum(0.0, drops)))
    return post, forget, int(post >= SAFE_ACC and forget <= SAFE_FORGET)


def _stage_callback(ctx):
    t = int(ctx["t"])
    task = ctx["task"]
    ts = ctx["task_seq"]
    V = ctx["V"]
    R = ctx["R"]
    method = ctx["method"]
    hp = ctx["hp"]

    g_new = float(V[t, t])
    g_old = float(np.mean(V[t, :t])) if t > 0 else 0.0
    g_old_min = float(np.min(V[t, :t])) if t > 0 else 0.0
    if t > 0:
        drops = [V[t - 1, j] - V[t, j] for j in range(t)]
        g_forget = float(np.mean(np.maximum(0.0, drops)))
    else:
        g_forget = 0.0

    r_prox = _proximity(ts, t, task)
    r_gap_cur, r_fwd_frac = _current_temporal_gap(ts, task)
    r_drift_cur_mean, r_drift_cur_max = _drift(ts, task, "current")
    r_drift_cur_log = float(np.log1p(r_drift_cur_mean))

    buffer = ctx["buffer"]
    n_replay = 0
    if buffer is not None:
        n_replay = max(1, int(hp["batch_size"] * hp.get("replay_frac", 0.0)))
    retained = 0
    for tt in ts.tasks[:t + 1]:
        if tt.X_calib is not None:
            retained += int(tt.X_calib.nbytes)
        if tt.y_calib is not None:
            retained += int(tt.y_calib.nbytes)
    peak = model_peak_training_ram(
        ctx["model"], buffer, ctx["ewc"], teacher=ctx["teacher"],
        batch_size=int(hp["batch_size"]), n_features=ts.n_features,
        n_replay=n_replay, retained_state_bytes=retained, ncm=ctx["ncm"])

    up = int(task.uplink_bytes_int8)
    if method in CALIB_METHODS:
        up += int(task.uplink_calib_bytes_int8)

    post_acc, forget_test, safe = _safe_label(R, t)
    return {
        "t": t,
        "stage": f"T{t}",
        "task_name": task.name,
        "n_train": int(len(task.y_train)),
        "n_select": int(len(task.y_select)) if task.y_select is not None else 0,
        "g_new_select": g_new,
        "g_old_select": g_old,
        "g_old_min_select": g_old_min,
        "g_old_forget_select": g_forget,
        "r_prox": r_prox,
        "r_gap_cur": r_gap_cur,
        "r_fwd_frac": r_fwd_frac,
        "r_fwd_flag": float(r_fwd_frac > 0.9),
        "r_drift_cur_mean": r_drift_cur_mean,
        "r_drift_cur_max": r_drift_cur_max,
        "r_drift_cur_log": r_drift_cur_log,
        "c_mem_KB": float(peak["peak_total_KB"]),
        "c_up_KB": up / 1024.0,
        "post_acc_seen": post_acc,
        "forget_test": forget_test,
        "safe": safe,
    }


def _run_one(item):
    arch, method, protocol, seed = item
    ts = _TS[(protocol, seed)]
    mk = model_kwargs_for(arch, ts.n_features)
    hp = {**BASE_TRAIN_HPARAMS,
          **_FROZEN.get(arch, {}).get(method, {}).get("hparams", {})}
    out = train_continual(
        MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed,
        stage_callback=_stage_callback)
    rows = []
    for t, rec in enumerate(out["stage_records"]):
        rec = dict(rec)
        rec.update(arch=arch, method=method, protocol=protocol, seed=seed,
                   train_seconds=float(out["per_task_train_seconds"][t]))
        rows.append(rec)
    return rows


def _sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic(X, y, l2=1.0, lr=0.1, n_iter=4000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    Xb = np.column_stack([np.ones(n), X])
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(n - y.sum()))
    sw = np.where(y == 1.0, n / (2.0 * pos), n / (2.0 * neg))
    w = np.zeros(Xb.shape[1], dtype=float)
    for _ in range(n_iter):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ ((p - y) * sw) / n
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def _standardize_fit(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def _standardize_apply(X, mu, sd):
    return (np.asarray(X, dtype=float) - mu) / sd


def _logistic_score(X, w):
    Xb = np.column_stack([np.ones(len(X)), X])
    return _sigmoid(Xb @ w)


def _auroc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and score[order[j + 1]] == score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _tune_threshold(y, score, mask=None, max_unsafe=0.05):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    grid = np.unique(np.round(score, 6))
    best = None
    for tau in grid:
        pred = (score >= tau) & mask
        sa = float(pred[y == 1].mean()) if np.any(y == 1) else 0.0
        ua = float(pred[y == 0].mean()) if np.any(y == 0) else 0.0
        if ua <= max_unsafe:
            key = (1, sa, -ua)
        else:
            key = (0, -ua, sa)
        if best is None or key > best[0]:
            best = (key, float(tau), sa, ua)
    return best[1], best[2], best[3]


def _eval_pred(y, pred, score, prefix):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=bool)
    safe = y == 1
    unsafe = ~safe
    sa = float(pred[safe].mean()) if safe.any() else float("nan")
    ua = float(pred[unsafe].mean()) if unsafe.any() else float("nan")
    tp = int((pred & safe).sum())
    fp = int((pred & unsafe).sum())
    return {
        f"{prefix}_n": int(len(y)),
        f"{prefix}_safe_n": int(safe.sum()),
        f"{prefix}_unsafe_n": int(unsafe.sum()),
        f"{prefix}_safe_accept": sa,
        f"{prefix}_unsafe_accept": ua,
        f"{prefix}_balanced_acc": 0.5 * (sa + (1.0 - ua)),
        f"{prefix}_precision": tp / max(1, tp + fp),
        f"{prefix}_auroc": _auroc(y, score),
    }


def _gate_tables(rows):
    df = rows.copy()
    df["c_up_KB"] = df["c_up_KB"].astype(float)
    df["budget_ok"] = ((df["c_mem_KB"] <= MEM_BUDGET_KB)
                       & (df["c_up_KB"] <= UP_BUDGET_KB))

    tune = df[(df["seed"] < 10) & (df["t"] >= 1)].copy()
    test = df[(df["seed"] >= 10) & (df["t"] >= 1)].copy()

    mu, sd = _standardize_fit(tune[GATE_FEATURES].to_numpy())
    Xt = _standardize_apply(tune[GATE_FEATURES].to_numpy(), mu, sd)
    Xe = _standardize_apply(test[GATE_FEATURES].to_numpy(), mu, sd)
    w = _fit_logistic(Xt, tune["safe"].to_numpy())
    s_tune = _logistic_score(Xt, w)
    s_test = _logistic_score(Xe, w)

    tau, _, _ = _tune_threshold(
        tune["safe"].to_numpy(), s_tune, mask=tune["budget_ok"].to_numpy(),
        max_unsafe=0.05)
    pred_buac = (s_test >= tau) & test["budget_ok"].to_numpy()
    pred_buac_nobudget = s_test >= tau

    # Gate trained only on IID tuning updates, evaluated on temporal test rows.
    tune_iid = tune[tune["protocol"] == "iid"]
    mu_i, sd_i = _standardize_fit(tune_iid[GATE_FEATURES].to_numpy())
    Xi = _standardize_apply(tune_iid[GATE_FEATURES].to_numpy(), mu_i, sd_i)
    w_i = _fit_logistic(Xi, tune_iid["safe"].to_numpy())
    s_i_tune = _logistic_score(Xi, w_i)
    tau_i, _, _ = _tune_threshold(
        tune_iid["safe"].to_numpy(), s_i_tune,
        mask=tune_iid["budget_ok"].to_numpy(), max_unsafe=0.05)
    s_i_test = _logistic_score(
        _standardize_apply(test[GATE_FEATURES].to_numpy(), mu_i, sd_i), w_i)
    pred_iid_trained = (s_i_test >= tau_i) & test["budget_ok"].to_numpy()

    g_new = test["g_new_select"].to_numpy().astype(float)
    forget = test["g_old_forget_select"].to_numpy().astype(float)
    fwd = test["r_fwd_flag"].to_numpy().astype(float)
    drift = test["r_drift_cur_log"].to_numpy().astype(float)
    rule_core = ((g_new >= 0.50) & (forget <= 0.10) & (fwd == 0.0)
                 & (drift <= 1.0))
    rule_risk = (np.maximum(0.0, 0.50 - g_new) / 0.50 + forget / 0.10
                 + 2.0 * fwd + np.maximum(0.0, drift - 1.0))
    pred_rule = rule_core & test["budget_ok"].to_numpy()
    pred_rule_nobudget = rule_core

    baselines = {
        "always_onboard": (np.ones(len(test), dtype=bool), np.ones(len(test))),
        "never_onboard": (np.zeros(len(test), dtype=bool), np.zeros(len(test))),
        "oracle": (test["safe"].to_numpy() == 1, test["safe"].to_numpy().astype(float)),
        "buac": (pred_buac, s_test),
        "buac_no_budget": (pred_buac_nobudget, s_test),
        "buac_iid_trained": (pred_iid_trained, s_i_test),
        "rule_buac": (pred_rule, -rule_risk),
        "rule_buac_no_budget": (pred_rule_nobudget, -rule_risk),
    }

    signal_specs = [
        ("iid_gain_only", "g_new_select", 1.0),
        ("temporal_gap_only", "r_fwd_frac", -1.0),
        ("drift_only", "r_drift_cur_log", -1.0),
        ("retention_only", "g_old_forget_select", -1.0),
    ]
    for name, col, sign in signal_specs:
        score = sign * test[col].to_numpy().astype(float)
        tune_score = sign * tune[col].to_numpy().astype(float)
        tau_b, _, _ = _tune_threshold(
            tune["safe"].to_numpy(), tune_score, mask=None, max_unsafe=0.05)
        baselines[name] = (score >= tau_b, score)

    eval_rows = []
    for name, (pred, score) in baselines.items():
        for scope in ["all"] + PROTOCOLS:
            if scope == "all":
                m = np.ones(len(test), dtype=bool)
            else:
                m = test["protocol"].to_numpy() == scope
            if not np.any(m):
                continue
            out = _eval_pred(test["safe"].to_numpy()[m], pred[m], score[m], "gate")
            eval_rows.append({"gate": name, "scope": scope, "tau": tau if name.startswith("buac") else np.nan,
                              "mem_budget_KB": MEM_BUDGET_KB, "up_budget_KB": UP_BUDGET_KB, **out})
    eval_df = pd.DataFrame(eval_rows)

    by_method = []
    for name, (pred, _) in baselines.items():
        for (protocol, method), g in test.groupby(["protocol", "method"]):
            idx = g.index
            p = pred[test.index.get_indexer(idx)]
            by_method.append({
                "gate": name, "protocol": protocol, "method": method,
                "n": int(len(g)), "safe_rate": float(g["safe"].mean()),
                "onboard_rate": float(np.mean(p)),
                "unsafe_onboard": float(np.mean(p[g["safe"].to_numpy() == 0]))
                if np.any(g["safe"].to_numpy() == 0) else float("nan"),
                "safe_onboard": float(np.mean(p[g["safe"].to_numpy() == 1]))
                if np.any(g["safe"].to_numpy() == 1) else float("nan"),
            })
    by_method_df = pd.DataFrame(by_method)

    coef = pd.DataFrame({
        "feature": ["intercept"] + GATE_FEATURES,
        "weight": w,
    })
    coef["abs_weight"] = coef["weight"].abs()

    decisions = test.copy()
    decisions["p_safe"] = s_test
    decisions["tau"] = tau
    decisions["buac_onboard"] = pred_buac
    decisions["buac_no_budget_onboard"] = pred_buac_nobudget
    decisions["buac_iid_trained_onboard"] = pred_iid_trained
    decisions["rule_buac_onboard"] = pred_rule
    decisions["rule_buac_risk"] = rule_risk
    decisions["buac_mode"] = np.where(
        pred_buac, "onboard",
        np.where(s_test >= tau, "ground_assisted", "reject"))

    sens = []
    for mem in (64.0, 128.0, 256.0, 512.0):
        for up in (16.0, 32.0, 64.0, 128.0):
            ok = (test["c_mem_KB"] <= mem) & (test["c_up_KB"] <= up)
            pred = (s_test >= tau) & ok
            out = _eval_pred(test["safe"].to_numpy(), pred, s_test, "gate")
            sens.append({"mem_budget_KB": mem, "up_budget_KB": up, **out})
    sens_df = pd.DataFrame(sens)

    by_seed = []
    for seed, g in test.groupby("seed"):
        pos = test.index.get_indexer(g.index)
        out = _eval_pred(g["safe"].to_numpy(), pred_buac[pos], s_test[pos], "gate")
        by_seed.append({"seed": int(seed), **out})
    by_seed_df = pd.DataFrame(by_seed)

    return (eval_df, by_method_df, coef, decisions, sens_df, by_seed_df,
            float(tau))


def _write_gate_outputs(df, prefix):
    if not ((df["seed"] < 10).any() and (df["seed"] >= 10).any()):
        print("Need tuning seeds <10 and held-out seeds >=10 for gate evaluation.")
        return
    eval_df, by_method, coef, decisions, sens, by_seed, tau = _gate_tables(df)
    eval_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_gate_eval.csv"), index=False)
    by_method.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_gate_by_method.csv"), index=False)
    coef.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_gate_coefficients.csv"), index=False)
    decisions.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_decisions.csv"), index=False)
    sens.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_budget_sensitivity.csv"), index=False)
    by_seed.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_gate_by_seed.csv"), index=False)

    print(f"\nBUAC gate threshold tau={tau:.4f}")
    print("\nHeld-out gate evaluation (all):")
    print(eval_df[eval_df.scope == "all"][
        ["gate", "gate_safe_accept", "gate_unsafe_accept",
         "gate_balanced_acc", "gate_auroc"]].round(3).to_string(index=False))
    for protocol in PROTOCOLS:
        print(f"\nHeld-out gate evaluation ({protocol}):")
        print(eval_df[eval_df.scope == protocol][
            ["gate", "gate_safe_accept", "gate_unsafe_accept",
             "gate_balanced_acc"]].round(3).to_string(index=False))
    print("\nSafe rate by protocol/method (held-out):")
    print(df[df.seed >= 10].groupby(["protocol", "method"])["safe"]
          .agg(["mean", "count"]).round(3).to_string())
    print(f"\nWrote {prefix}_gate_eval.csv, _gate_by_method.csv, "
          f"_gate_coefficients.csv, _decisions.csv, _budget_sensitivity.csv, "
          f"_gate_by_seed.csv")


def main():
    global _FROZEN, _TS
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="MLP")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--prefix", default="buac")
    ap.add_argument("--stage-csv", default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.stage_csv:
        _write_gate_outputs(pd.read_csv(args.stage_csv), args.prefix)
        return

    lo, hi = [int(x) for x in args.seeds.split("-")]
    seeds = list(range(lo, hi + 1))
    arch = args.arch
    methods = [m for m in args.methods if m in METHODS]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams_strict()
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))

    for protocol in PROTOCOLS:
        cfg = PROTOCOL_CFG[protocol]
        for seed in seeds:
            if cfg["builder"] == "iid":
                _TS[(protocol, seed)] = du.build_cucd_id_tasks(
                    cucd_df, seed=seed, test_frac=cfg["test_frac"],
                    val_frac=cfg["val_frac"])
            else:
                _TS[(protocol, seed)] = du.build_cucd_id_tasks_purged_temporal(
                    cucd_df, seed=seed, test_frac=cfg["test_frac"],
                    val_frac=cfg["val_frac"],
                    purge_time_rad=PURGE_TIME_RAD,
                    train_sample_mode=cfg["train_sample_mode"],
                    val_mode=cfg["val_mode"])

    items = [(arch, method, protocol, seed)
             for method in methods
             for protocol in PROTOCOLS
             for seed in seeds]
    jobs = get_jobs(args.jobs)
    print(f"BUAC: {len(items)} runs; {describe(jobs)}")
    t0 = time.perf_counter()
    rows = []
    for i, recs in enumerate(parallel_map(_run_one, items, jobs=jobs), 1):
        rows.extend(recs)
        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] ({time.perf_counter()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    sig_path = os.path.join(RESULTS_DIR, f"{args.prefix}_stage_signals.csv")
    df.to_csv(sig_path, index=False)
    print(f"Wrote {sig_path} ({len(df)} stage rows)")

    _write_gate_outputs(df, args.prefix)


if __name__ == "__main__":
    main()
