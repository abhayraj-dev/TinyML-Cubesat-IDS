"""
Main experiment matrix (Phase 1).

For each of the 2 datasets (CuCD-ID primary, STIN secondary) x 3
architectures (MLP, GRU, CNN1D) x 9 methods (finetune, replay, ewc, full,
lwf, bic, icarl, joint-legacy-warm-start, joint_scratch) x 10 seeds, trains
sequentially using frozen per-method/arch hyperparameters from
best_hparams_per_method_arch.json (validation-tuned, never test-tuned),
sequentially through the class-incremental task stream and records:
  - the full per-task accuracy matrix R
  - ACC / BWT / FM continual-learning metrics
  - CuCD-ID-only binary detection metrics (AUROC/AUPR/FPR95 from 1-p(normal))
  - per-task wall-clock training time
  - deployed memory footprint (model + CMMU) in bytes
  - estimated MACs per forward pass

Phase-1 protocol note: tasks are built with val_frac=0.10 (test locked first;
val carved from train pool for BiC fitting + val-balanced Fisher). This
changes absolute numbers vs the pre-fix CSVs (random split, no val) -- see
DATALOG.md; do not mix old/new numbers in one table.

Writes:
  results/main_results_runs.csv     (one row per individual run)
  results/main_results_summary.csv  (mean +/- std, grouped by dataset/arch/method)
  results/accuracy_matrices.json    (raw R matrices for plotting)
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import data_utils as du
from train import train_continual
from metrics import (compute_cl_metrics, compute_detection_metrics,
                      compute_macro_f1_balanced_acc,
                      model_persistent_footprint, model_peak_training_ram,
                      estimate_macs_per_sample, estimate_macs_per_update)
from config import (MODEL_CLASSES, ARCHS, METHODS, SEEDS, model_kwargs_for,
                      DATA_DIR, RESULTS_DIR, BASE_TRAIN_HPARAMS,
                      load_frozen_hparams_strict)


VAL_FRAC = 0.10  # Phase 1: locked test + held-out val for BiC/val-Fisher


def final_detection_metrics(ts, out):
    """CuCD-ID only: 1-p(normal) detector on the concatenated all-task test
    pool through the final head (with BiC correction applied when fitted).
    Returns NaNs for STIN (no benign class)."""
    if ts.dataset_name != "CuCD-ID":
        return {"DET_AUROC": float("nan"), "DET_AUPR": float("nan"),
                "DET_FPR95": float("nan"), "DET_PREV": float("nan")}
    try:
        normal_idx = list(ts.class_names_global).index("normal")
    except ValueError:
        return {"DET_AUROC": float("nan"), "DET_AUPR": float("nan"),
                "DET_FPR95": float("nan"), "DET_PREV": float("nan")}
    # Canonical processing order => global offsets are cumulative.
    Xs, ys, off = [], [], 0
    for task in ts.tasks:
        Xs.append(task.X_test)
        ys.append(task.y_test + off)
        off += task.n_new_classes
    import numpy as np
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    logits, _ = out["model"].forward(X_all)
    corr = out.get("corrector")
    if corr is not None and getattr(corr, "fitted", False):
        logits = corr.apply(logits)
    return compute_detection_metrics(y_all, logits, normal_idx)


def load_frozen_hparams():
    """Load per-method/arch frozen hyperparameters from validation tuning."""
    path = os.path.join(RESULTS_DIR, "best_hparams_per_method_arch.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_best_hparams():
    """Legacy: loads global best_hparams.json for backward compatibility."""
    path = os.path.join(RESULTS_DIR, "best_hparams.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(lr=3e-3, ewc_lambda=1000.0, buffer_capacity=128, replay_frac=0.5)


def build_datasets():
    cucd_df = du.load_cucd_id(os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
                               os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    stin_df = du.load_stin(os.path.join(DATA_DIR, "STIN", "SAT20.csv"),
                            os.path.join(DATA_DIR, "STIN", "TER20.csv"))
    datasets = {}
    for seed in SEEDS:
        datasets[("CuCD-ID", seed)] = du.build_cucd_id_tasks(cucd_df, seed=seed, val_frac=VAL_FRAC)
        datasets[("STIN", seed)] = du.build_stin_tasks(stin_df, seed=seed, val_frac=VAL_FRAC)
    return datasets


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Fail loudly if tuning was skipped: never fall back to legacy/defaults.
    frozen_hparams = load_frozen_hparams_strict()
    print(f"Using per-method/arch frozen hyperparameters from validation tuning")

    datasets = build_datasets()

    run_rows = []
    matrices = {}
    t_start = time.perf_counter()

    total = len(["CuCD-ID", "STIN"]) * len(ARCHS) * len(METHODS) * len(SEEDS)
    done = 0
    for dset_name in ["CuCD-ID", "STIN"]:
        for arch in ARCHS:
            for method in METHODS:
                for seed in SEEDS:
                    ts = datasets[(dset_name, seed)]
                    model_kwargs = model_kwargs_for(arch, ts.n_features)
                    
                    arch_hparams = frozen_hparams.get(arch, {}).get(method, {}).get("hparams", {})
                    hp = {**BASE_TRAIN_HPARAMS, **arch_hparams}
                    # Log the exact frozen hyperparameters for this run
                    if done == 0 or (dset_name == "CuCD-ID" and arch == "MLP" and method == "replay" and seed == 0):
                        print(f"  [Frozen hparams] Dataset={dset_name} Arch={arch} Method={method}")
                        for k, v in hp.items():
                            print(f"    {k}={v}")

                    out = train_continual(MODEL_CLASSES[arch], model_kwargs, ts, method, hp, seed=seed)
                    m = compute_cl_metrics(out["R"])
                    det = final_detection_metrics(ts, out)
                    mf_ba = compute_macro_f1_balanced_acc(out["model"], ts, out.get("corrector"))

                    # Persistent footprint (deployed storage)
                    persist = model_persistent_footprint(out["model"], out["buffer"], out["ewc"])

                    # Peak training RAM (per-step resident state, not full buffer)
                    if out["buffer"] is not None:
                        n_replay_step = max(1, int(hp["batch_size"] * hp.get("replay_frac", 0.0)))
                    else:
                        n_replay_step = 0
                    peak = model_peak_training_ram(out["model"], out["buffer"], out["ewc"],
                                                    teacher=out.get("teacher"), batch_size=hp["batch_size"],
                                                    n_features=ts.n_features, n_replay=n_replay_step)

                    # Estimated/conservative MAC cost (student fwd+bwd + teacher fwd,
                    # using final-head-width per-sample proxy for all tasks)
                    macs_fwd = estimate_macs_per_sample(out["model"], ts.n_features)
                    macs_update = int(macs_fwd * 3 * int(out.get("train_samples_total", 0))
                                      + macs_fwd * int(out.get("teacher_samples_total", 0)))

                    row = dict(
                        dataset=dset_name, arch=arch, method=method, seed=seed,
                        n_tasks=len(ts.tasks), **m, **det, **mf_ba,
                        train_seconds_total=float(sum(out["per_task_train_seconds"])),
                        train_seconds_per_task=float(np.mean(out["per_task_train_seconds"])),
                        **persist,
                        peak_total_KB=peak["peak_total_KB"],
                        peak_adam_KB=peak["peak_adam_bytes_fp32"] / 1024.0,
                        macs_per_forward=macs_fwd,
                        macs_per_update=macs_update,
                    )
                    run_rows.append(row)
                    key = f"{dset_name}|{arch}|{method}|{seed}"
                    matrices[key] = out["R"].tolist()

                    done += 1
                    if done % 10 == 0:
                        print(f"  [{done}/{total}] {dset_name} {arch} {method} seed={seed} "
                              f"ACC={m['ACC']:.3f} BWT={m['BWT']:.3f} "
                              f"({time.perf_counter()-t_start:.1f}s elapsed)")

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(os.path.join(RESULTS_DIR, "main_results_runs.csv"), index=False)

    summary = runs_df.groupby(["dataset", "arch", "method"]).agg(
        ACC_mean=("ACC", "mean"), ACC_std=("ACC", "std"),
        BWT_mean=("BWT", "mean"), BWT_std=("BWT", "std"),
        FM_mean=("FM", "mean"), FM_std=("FM", "std"),
        MacroF1_mean=("MacroF1", "mean"), MacroF1_std=("MacroF1", "std"),
        BalancedAccuracy_mean=("BalancedAccuracy", "mean"), BalancedAccuracy_std=("BalancedAccuracy", "std"),
        DET_AUROC_mean=("DET_AUROC", "mean"), DET_AUROC_std=("DET_AUROC", "std"),
        DET_AUPR_mean=("DET_AUPR", "mean"),
        DET_FPR95_mean=("DET_FPR95", "mean"),
        train_seconds_per_task_mean=("train_seconds_per_task", "mean"),
        persistent_total_KB_mean=("persistent_total_KB", "mean"),
        persistent_model_bytes_mean=("persistent_model_bytes_int8", "mean"),
        persistent_cmmu_bytes_mean=("persistent_cmmu_bytes", "mean"),
        peak_total_KB_mean=("peak_total_KB", "mean"),
        peak_adam_KB_mean=("peak_adam_KB", "mean"),
        macs_per_forward_mean=("macs_per_forward", "mean"),
        macs_per_update_mean=("macs_per_update", "mean"),
    ).reset_index()
    summary["persistent_model_KB_mean"] = summary["persistent_model_bytes_mean"] / 1024.0
    summary["persistent_cmmu_KB_mean"] = summary["persistent_cmmu_bytes_mean"] / 1024.0
    summary = summary.drop(columns=["persistent_model_bytes_mean", "persistent_cmmu_bytes_mean"])
    summary.to_csv(os.path.join(RESULTS_DIR, "main_results_summary.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "accuracy_matrices.json"), "w") as f:
        json.dump(matrices, f)

    print(f"\nDone. Total time: {time.perf_counter()-t_start:.1f}s")
    print(f"\n{len(runs_df)} runs written to results/main_results_runs.csv")
    print("\nSummary (ACC_mean):")
    print(summary.pivot_table(index=["dataset", "arch"], columns="method", values="ACC_mean").round(3))


if __name__ == "__main__":
    main()
