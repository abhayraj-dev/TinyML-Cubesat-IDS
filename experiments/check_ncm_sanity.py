"""Six-cell NCM sanity/control diagnostics.

The original ``icarl_ncm`` comparison changes two things at once relative to
the main matrix: exemplars are selected by embedding-space herding, and the
deployed head becomes cosine NCM.  This script separates the effects.

Part 1 (90 runs): train iCaRL with embedding-space herding and evaluate
  * the same model's softmax head (embedding-herding softmax control),
  * NCM over the stored buffer,
  * NCM over all accumulated training data (final model),
  * 1-nearest-buffer-exemplar cosine baseline.
This also reproduces the existing NCM buffer result for regression checking.

Part 2 (18 runs): joint_scratch on all accumulated training data, comparing
softmax against NCM fitted on the same all-data training pool.

Writes:
  results/ncm_sanity_runs.csv
  results/ncm_sanity_summary.csv
  results/ncm_sanity_vs_main.csv
  results/ncm_sanity_vs_main_runs.csv
  results/ncm_sanity_per_class.csv
  results/accuracy_matrices_ncm_sanity.json
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from parallel import configure_single_thread, describe, parallel_map, get_jobs
configure_single_thread()

import numpy as np
import pandas as pd

from train import train_continual
from ncm import NCMClassifier
from inference import concatenate_test
from metrics import per_class_prf1
from config import (MODEL_CLASSES, ARCHS, SEEDS, model_kwargs_for,
                    RESULTS_DIR, BASE_TRAIN_HPARAMS,
                    load_frozen_hparams_strict)
from run_main import build_datasets

DEFAULT_DATASETS = ["CuCD-ID", "STIN"]
DEFAULT_ARCHS = list(ARCHS)
JOINT_SEEDS = [0, 1, 2]
_FROZEN = None
_DATASETS = {}


def _all_train(ts):
    Xs, ys, offset = [], [], 0
    for task in ts.tasks:
        Xs.append(task.X_train)
        ys.append(task.y_train + offset)
        offset += task.n_new_classes
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _task_mean_acc(ts, pred):
    sample_off, class_off, accs = 0, 0, []
    for task in ts.tasks:
        n = len(task.y_test)
        y = pred[sample_off:sample_off + n]
        accs.append(float((y == (task.y_test + class_off)).mean()))
        sample_off += n
        class_off += task.n_new_classes
    return float(np.mean(accs))


def _class_metrics(y_true, logits, ts=None):
    pred = np.asarray(logits).argmax(axis=1)
    n_classes = int(np.max(y_true)) + 1
    prf = per_class_prf1(y_true, pred, n_classes)
    sample_acc = float((pred == y_true).mean())
    return {
        "ACC": _task_mean_acc(ts, pred) if ts is not None else sample_acc,
        "SampleACC": sample_acc,
        "MacroF1": float(np.mean(prf["f1"])),
        "BalancedAccuracy": float(np.mean(prf["recall"])),
    }, prf, pred


def _cosine_margin(ncm, model, X, y):
    feat = NCMClassifier._l2_normalize(NCMClassifier._embed(model, X))
    means = ncm.class_means.astype(np.float64)
    cos = feat @ means.T
    true_cos = cos[np.arange(len(y)), y]
    other = cos.copy()
    other[np.arange(len(y)), y] = -np.inf
    return float(np.mean(true_cos - other.max(axis=1)))


def _one_nn_buffer(model, buffer, X, y):
    feat = NCMClassifier._l2_normalize(NCMClassifier._embed(model, X))
    buf = NCMClassifier._l2_normalize(
        NCMClassifier._embed(model, buffer.X[:buffer.size]))
    sim = feat @ buf.T
    pred = buffer.y[:buffer.size][sim.argmax(axis=1)]
    return float((pred == y).mean())


def _base_row(dataset, arch, seed, stage):
    return dict(dataset=dataset, arch=arch, seed=seed, stage=stage)


def _add_predictor(row, prefix, metrics, ncm=None, model=None, X=None, y=None):
    row[f"{prefix}_ACC"] = metrics["ACC"]
    row[f"{prefix}_SampleACC"] = metrics["SampleACC"]
    row[f"{prefix}_MacroF1"] = metrics["MacroF1"]
    row[f"{prefix}_BalancedAccuracy"] = metrics["BalancedAccuracy"]
    if ncm is not None and model is not None and X is not None and y is not None:
        row[f"{prefix}_cosine_margin"] = _cosine_margin(ncm, model, X, y)


def _run_embedding_icarl(dset_name, arch, seed):
    ts = _DATASETS[(dset_name, seed)]
    model_kwargs = model_kwargs_for(arch, ts.n_features)
    arch_hp = _FROZEN.get(arch, {}).get("icarl", {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hp}
    out = train_continual(
        MODEL_CLASSES[arch], model_kwargs, ts, "icarl", hp, seed=seed,
        ncm_eval=True, embedding_herding=True)
    model, buffer = out["model"], out["buffer"]
    X_test, y_test = concatenate_test(ts)
    X_train, y_train = _all_train(ts)

    soft_logits, _ = model.forward(X_test)
    soft_m, _, _ = _class_metrics(y_test, soft_logits, ts)
    buf_m, buf_prf, _ = _class_metrics(
        y_test, out["ncm"].logits(model, X_test), ts)
    buf_train_m, _, _ = _class_metrics(
        y_train, out["ncm"].logits(model, X_train))

    ncm_all = NCMClassifier(n_classes=ts.n_classes_total, temperature=10.0)
    ncm_all.fit_from_arrays(model, X_train, y_train, n_classes=ts.n_classes_total)
    all_m, all_prf, _ = _class_metrics(
        y_test, ncm_all.logits(model, X_test), ts)
    all_train_m, _, _ = _class_metrics(y_train, ncm_all.logits(model, X_train))

    row = _base_row(dset_name, arch, seed, "icarl_embedding")
    _add_predictor(row, "softmax", soft_m)
    _add_predictor(row, "ncm_buffer", buf_m, out["ncm"], model, X_test, y_test)
    _add_predictor(row, "ncm_all", all_m, ncm_all, model, X_test, y_test)
    row.update({
        "one_nn_buffer_ACC": _one_nn_buffer(model, buffer, X_test, y_test),
        "ncm_buffer_train_ACC": buf_train_m["ACC"],
        "ncm_all_train_ACC": all_train_m["ACC"],
        "ncm_buffer_count_min": int(out["ncm"].class_counts.min()),
        "ncm_buffer_count_max": int(out["ncm"].class_counts.max()),
        "buffer_size": int(buffer.size),
        "n_classes": int(ts.n_classes_total),
    })
    return row, out["R"].tolist(), buf_prf, all_prf


def _run_joint(dset_name, arch, seed):
    ts = _DATASETS[(dset_name, seed)]
    model_kwargs = model_kwargs_for(arch, ts.n_features)
    arch_hp = _FROZEN.get(arch, {}).get("joint_scratch", {}).get("hparams", {})
    hp = {**BASE_TRAIN_HPARAMS, **arch_hp}
    out = train_continual(
        MODEL_CLASSES[arch], model_kwargs, ts, "joint_scratch", hp, seed=seed)
    model = out["model"]
    X_test, y_test = concatenate_test(ts)
    X_train, y_train = _all_train(ts)

    soft_logits, _ = model.forward(X_test)
    soft_m, _, _ = _class_metrics(y_test, soft_logits, ts)
    ncm = NCMClassifier(n_classes=ts.n_classes_total, temperature=10.0)
    ncm.fit_from_arrays(model, X_train, y_train, n_classes=ts.n_classes_total)
    ncm_m, ncm_prf, _ = _class_metrics(y_test, ncm.logits(model, X_test), ts)
    ncm_train_m, _, _ = _class_metrics(y_train, ncm.logits(model, X_train))

    row = _base_row(dset_name, arch, seed, "joint_scratch")
    _add_predictor(row, "softmax", soft_m)
    _add_predictor(row, "ncm_all", ncm_m, ncm, model, X_test, y_test)
    row.update({
        "ncm_all_train_ACC": ncm_train_m["ACC"],
        "buffer_size": 0,
        "n_classes": int(ts.n_classes_total),
    })
    return row, out["R"].tolist(), None, ncm_prf


def _run_one(item):
    dset_name, arch, seed, stage = item
    if stage == "icarl_embedding":
        return _run_embedding_icarl(dset_name, arch, seed)
    return _run_joint(dset_name, arch, seed)


def _summary(df):
    rows = []
    for (stage, dset, arch), g in df.groupby(["stage", "dataset", "arch"]):
        row = dict(stage=stage, dataset=dset, arch=arch, n=len(g))
        for col in [
            "softmax_ACC", "ncm_buffer_ACC", "ncm_all_ACC",
            "softmax_SampleACC", "ncm_buffer_SampleACC", "ncm_all_SampleACC",
            "one_nn_buffer_ACC", "softmax_MacroF1", "ncm_buffer_MacroF1",
            "ncm_all_MacroF1", "ncm_buffer_train_ACC", "ncm_all_train_ACC",
            "ncm_buffer_cosine_margin", "ncm_all_cosine_margin",
        ]:
            if col in g and g[col].notna().any():
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _vs_main(df):
    runs = df[df.stage == "icarl_embedding"].copy()
    base_path = os.path.join(RESULTS_DIR, "main_results_runs.csv")
    ncm_path = os.path.join(RESULTS_DIR, "icarl_ncm_runs.csv")
    if os.path.exists(base_path):
        base = pd.read_csv(base_path)
        base = base[base.method == "icarl"][
            ["dataset", "arch", "seed", "ACC", "MacroF1", "BalancedAccuracy"]
        ].rename(columns={"ACC": "input_softmax_ACC",
                          "MacroF1": "input_softmax_MacroF1",
                          "BalancedAccuracy": "input_softmax_BalancedAccuracy"})
        runs = runs.merge(base, on=["dataset", "arch", "seed"], how="left")
    if os.path.exists(ncm_path):
        old = pd.read_csv(ncm_path)[
            ["dataset", "arch", "seed", "ACC", "MacroF1", "BalancedAccuracy"]
        ].rename(columns={"ACC": "existing_ncm_ACC",
                          "MacroF1": "existing_ncm_MacroF1",
                          "BalancedAccuracy": "existing_ncm_BalancedAccuracy"})
        runs = runs.merge(old, on=["dataset", "arch", "seed"], how="left")

    if "input_softmax_ACC" in runs:
        runs["embedding_minus_input_softmax_ACC"] = (
            runs["softmax_ACC"] - runs["input_softmax_ACC"])
    if "existing_ncm_ACC" in runs:
        runs["ncm_buffer_minus_existing_ncm_ACC"] = (
            runs["ncm_buffer_ACC"] - runs["existing_ncm_ACC"])
    runs["ncm_buffer_minus_embedding_softmax_ACC"] = (
        runs["ncm_buffer_ACC"] - runs["softmax_ACC"])
    runs["ncm_all_minus_embedding_softmax_ACC"] = (
        runs["ncm_all_ACC"] - runs["softmax_ACC"])
    runs["ncm_buffer_minus_ncm_all_ACC"] = (
        runs["ncm_buffer_ACC"] - runs["ncm_all_ACC"])

    rows = []
    for (dset, arch), g in runs.groupby(["dataset", "arch"]):
        row = dict(dataset=dset, arch=arch, stage="icarl_embedding", n=len(g))
        for col in [
            "input_softmax_ACC", "softmax_ACC", "ncm_buffer_ACC", "ncm_all_ACC",
            "existing_ncm_ACC", "embedding_minus_input_softmax_ACC",
            "ncm_buffer_minus_embedding_softmax_ACC",
            "ncm_all_minus_embedding_softmax_ACC",
            "ncm_buffer_minus_ncm_all_ACC",
            "ncm_buffer_minus_existing_ncm_ACC",
        ]:
            if col in g and g[col].notna().any():
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)

    joint = df[df.stage == "joint_scratch"]
    for (dset, arch), g in joint.groupby(["dataset", "arch"]):
        row = dict(dataset=dset, arch=arch, stage="joint_scratch", n=len(g))
        for col in ["softmax_ACC", "ncm_all_ACC"]:
            if col in g:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        row["ncm_all_minus_softmax_ACC_mean"] = float(
            (g["ncm_all_ACC"] - g["softmax_ACC"]).mean())
        rows.append(row)
    return runs, pd.DataFrame(rows)


def main():
    global _FROZEN, _DATASETS
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--joint-seeds", nargs="+", type=int, default=JOINT_SEEDS)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.datasets = ["CuCD-ID"]
        args.archs = ["MLP"]
        args.seeds = [0]
        args.joint_seeds = [0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams_strict()
    _DATASETS = build_datasets()

    items = [(d, a, s, "icarl_embedding")
             for d in args.datasets for a in args.archs for s in args.seeds]
    items += [(d, a, s, "joint_scratch")
              for d in args.datasets for a in args.archs for s in args.joint_seeds]
    jobs = get_jobs()
    print(f"NCM sanity/control: {len(items)} runs; {describe(jobs)}")

    rows, matrices, per_class = [], {}, []
    t0 = time.perf_counter()
    for i, (row, matrix, buf_prf, all_prf) in enumerate(
            parallel_map(_run_one, items, jobs=jobs), 1):
        rows.append(row)
        matrices[f"{row['stage']}|{row['dataset']}|{row['arch']}|{row['seed']}"] = matrix
        ts = _DATASETS[(row["dataset"], row["seed"])]
        names = ts.class_names_global
        for prefix, prf in (("ncm_buffer", buf_prf), ("ncm_all", all_prf)):
            if prf is None:
                continue
            for c in range(len(names)):
                per_class.append(dict(
                    stage=row["stage"], dataset=row["dataset"], arch=row["arch"],
                    seed=row["seed"], predictor=prefix, class_index=c,
                    class_name=names[c],
                    precision=float(prf["precision"][c]),
                    recall=float(prf["recall"][c]),
                    f1=float(prf["f1"][c]),
                    support=int(prf["support"][c]),
                ))
        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {row['stage']} {row['dataset']} "
                  f"{row['arch']} seed={row['seed']} "
                  f"softmax={row.get('softmax_ACC', float('nan')):.3f} "
                  f"ncm_buffer={row.get('ncm_buffer_ACC', float('nan')):.3f} "
                  f"ncm_all={row.get('ncm_all_ACC', float('nan')):.3f} "
                  f"({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ncm_sanity_runs.csv"), index=False)
    _summary(df).to_csv(os.path.join(RESULTS_DIR, "ncm_sanity_summary.csv"),
                        index=False)
    runs_vs_main, vs_main = _vs_main(df)
    runs_vs_main.to_csv(os.path.join(RESULTS_DIR, "ncm_sanity_vs_main_runs.csv"),
                        index=False)
    vs_main.to_csv(os.path.join(RESULTS_DIR, "ncm_sanity_vs_main.csv"), index=False)
    pd.DataFrame(per_class).to_csv(
        os.path.join(RESULTS_DIR, "ncm_sanity_per_class.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "accuracy_matrices_ncm_sanity.json"), "w") as f:
        import json
        json.dump(matrices, f)

    print("\nNCM control summary (mean ACC):")
    print(df.pivot_table(index=["dataset", "arch"], columns="stage",
                         values=["softmax_ACC", "ncm_buffer_ACC", "ncm_all_ACC"]
                         ).round(3).to_string())
    print(f"\nWrote NCM sanity outputs ({len(df)} runs, "
          f"{time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
