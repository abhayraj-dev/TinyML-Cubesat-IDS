"""
Per-class precision/recall/F1 and confusion-matrix analysis.

The main experiment matrix (run_main.py) reports task-level ACC/BWT/FM,
which only *imply* that old classes get forgotten under naive fine-tuning.
This script makes that failure mode directly visible at the level of
individual attack classes: for a small set of representative configs, the
final trained model is evaluated on the full (all-tasks-seen) test set and
a per-class precision/recall/F1 breakdown + confusion matrix is computed.

Configs analysed (10 seeds each, frozen per-method/arch hyperparameters):
  * STIN,    GRU -- across all 9 methods (the secondary/stress-test dataset).
  * CuCD-ID, MLP -- across all 9 methods (the primary dataset's
    best-performing architecture, for a second, cleaner illustration).

Writes:
  results/per_class_f1_runs.csv       (one row per class x method x seed)
  results/per_class_f1_summary.csv    (mean over seeds)
  results/confusion_matrices.json     (per (dataset,arch,method,seed) confusion matrix)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import per_class_prf1
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    METHODS, SEEDS, BASE_TRAIN_HPARAMS)


def load_frozen_hparams():
    """Fail loudly if tuning was skipped: never train on defaults."""
    from config import load_frozen_hparams_strict
    return load_frozen_hparams_strict()


def evaluate_final_model_all_classes(model, task_seq, corrector=None):
    """Recomputes the same class_offset resolution train.py uses (canonical
    task order), builds the full concatenated test set across every task,
    and returns (y_true_global, y_pred_global, class_names_in_global_order,
    task_name_per_global_class). Applies the BiC corrector when fitted."""
    class_offset = 0
    Xs, ys = [], []
    task_of_class = []
    for task in task_seq.tasks:
        y_test_global = task.y_test + class_offset
        Xs.append(task.X_test)
        ys.append(y_test_global)
        task_of_class.extend([task.name] * task.n_new_classes)
        class_offset += task.n_new_classes
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    logits, _ = model.forward(X_all)
    if corrector is not None and getattr(corrector, "fitted", False):
        logits = corrector.apply(logits)
    preds = logits.argmax(axis=1)
    return y_all, preds, task_seq.class_names_global, task_of_class


def run_config(dataset_name, arch, df, builder, frozen_hparams, rows_out, cm_out):
    for method in METHODS:
        arch_hparams = frozen_hparams.get(arch, {}).get(method, {}).get("hparams", {})
        for seed in SEEDS:
            ts = builder(df, seed=seed, val_frac=0.10)
            mk = model_kwargs_for(arch, ts.n_features)
            hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
            out = train_continual(MODEL_CLASSES[arch], mk, ts, method, hp, seed=seed)
            y_true, y_pred, class_names, task_of_class = evaluate_final_model_all_classes(
                out["model"], ts, out.get("corrector"))
            report = per_class_prf1(y_true, y_pred, len(class_names))

            for c, cname in enumerate(class_names):
                rows_out.append(dict(
                    dataset=dataset_name, arch=arch, method=method, seed=seed,
                    class_id=c, class_name=cname, task_name=task_of_class[c],
                    precision=report["precision"][c], recall=report["recall"][c],
                    f1=report["f1"][c], support=int(report["support"][c]),
                ))

            key = f"{dataset_name}|{arch}|{method}|{seed}"
            cm_out[key] = {
                "class_names": class_names,
                "confusion_matrix": report["confusion_matrix"].tolist(),
            }
            print(f"  {key}  mean-F1={report['f1'].mean():.3f}")


def main():
    frozen_hparams = load_frozen_hparams()
    cucd_df = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                               os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    stin_df = du.load_stin(os.path.join(DATA_DIR, "STIN", "SAT20.csv"),
                            os.path.join(DATA_DIR, "STIN", "TER20.csv"))

    rows, cms = [], {}

    print("STIN + GRU (all methods):")
    run_config("STIN", "GRU", stin_df, du.build_stin_tasks, frozen_hparams, rows, cms)

    print("CuCD-ID + MLP (all methods):")
    run_config("CuCD-ID", "MLP", cucd_df, du.build_cucd_id_tasks, frozen_hparams, rows, cms)

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(os.path.join(RESULTS_DIR, "per_class_f1_runs.csv"), index=False)

    summary = runs_df.groupby(["dataset", "arch", "method", "class_name", "task_name"]).agg(
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        support_mean=("support", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "per_class_f1_summary.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "confusion_matrices.json"), "w") as f:
        json.dump(cms, f)

    print(f"\n{len(runs_df)} per-class rows written to results/per_class_f1_runs.csv")
    print("\nMean F1 by (dataset, arch, method, task_name):")
    print(summary.groupby(["dataset", "method", "task_name"]).f1_mean.mean().unstack("method").round(3))


if __name__ == "__main__":
    main()
