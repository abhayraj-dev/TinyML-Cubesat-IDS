"""Generates every figure used in the results section from the CSV/JSON
files written by experiments/run_hparam_sweep.py, run_main.py and
run_ablation.py. Run after all three have completed:

    python3 analysis/plots.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import numpy as np
import pandas as pd
from config import SEEDS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

METHOD_ORDER = ["finetune", "ewc", "lwf", "replay", "full", "bic", "icarl",
                "joint", "joint_scratch"]
METHOD_LABELS = {"finetune": "Fine-tune (no CMMU)", "ewc": "EWC only",
                 "lwf": "LwF (distill only)",
                 "replay": "Replay only", "full": "Full CMMU (Replay+EWC)",
                 "bic": "BiC (herding+distill+bias-corr)",
                 "icarl": "iCaRL-lite (herding+distill)",
                 "joint": "Joint warm-start (legacy)",
                 "joint_scratch": "Joint from-scratch (offline ref.)"}
COLORS = {"finetune": "#B0413E", "ewc": "#E1A94C", "lwf": "#9E9E9E",
          "replay": "#4C72B0", "full": "#2E7D32",
          "bic": "#AB47BC", "icarl": "#00897B",
          "joint": "#7B7B7B", "joint_scratch": "#424242"}
SEEDS_ALL = SEEDS


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print("wrote", path)


# Two-sided 95% Student-t critical values for small samples. Using 1.96 with
# n=10 would understate the interval by roughly 15%; the table is intentionally
# explicit instead of adding a SciPy dependency.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def ci95(std, n):
    """Half-width of the 95% CI of a mean (t critical * SE)."""
    std = np.asarray(std, dtype=float)
    n = np.asarray(n, dtype=float)
    crit = np.array([_T_CRIT_95.get(int(x), 1.96) for x in np.atleast_1d(n)], dtype=float)
    crit = crit.reshape(np.shape(n))
    return crit * std / np.sqrt(np.maximum(n, 1))


# --------------------------------------------------------------------------- #
def fig_method_comparison_bars():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "main_results_runs.csv"))
    cols = ["ACC", "MacroF1", "BWT", "FM"]
    cols = [c for c in cols if c in df.columns]
    archs = [a for a in ["MLP", "GRU", "CNN1D"] if a in set(df.arch)]
    for dset in df.dataset.unique():
        sub = df[df.dataset == dset]
        fig, axes = plt.subplots(len(archs), len(cols),
                                 figsize=(4 * len(cols) + 1, 3.0 * len(archs)),
                                 squeeze=False)
        for ai, arch in enumerate(archs):
            arch_sub = sub[sub.arch == arch]
            agg = arch_sub.groupby("method")[cols].agg(["mean", "std", "count"])
            methods = [m for m in METHOD_ORDER if m in agg.index]
            n_seeds = int(arch_sub.seed.nunique()) if "seed" in arch_sub.columns else 0
            for mi, metric in enumerate(cols):
                ax = axes[ai, mi]
                means = [agg.loc[m, (metric, "mean")] for m in methods]
                errs = [ci95(agg.loc[m, (metric, "std")], agg.loc[m, (metric, "count")])
                        for m in methods]
                colors = [COLORS[m] for m in methods]
                ax.bar(range(len(methods)), means, yerr=errs, color=colors, capsize=4)
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels([METHOD_LABELS[m].split(" (")[0] for m in methods],
                                   rotation=35, ha="right", fontsize=8)
                if ai == 0:
                    ax.set_title(metric)
                if mi == 0:
                    ax.set_ylabel(f"{arch}\nmean")
                ax.axhline(0, color="black", linewidth=0.6)
                ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"CL method comparison on {dset} "
                     f"(mean +/- 95% t CI over {n_seeds} seeds; faceted by architecture)")
        savefig(f"method_comparison_{dset}.png")


def fig_forgetting_curves():
    with open(os.path.join(RESULTS_DIR, "accuracy_matrices.json")) as f:
        matrices = json.load(f)
    for dset in ["CuCD-ID", "STIN"]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for method in METHOD_ORDER:
            key0 = f"{dset}|MLP|{method}|0"
            if key0 not in matrices:
                continue
            # average task-0 accuracy trajectory across seeds
            traj = []
            T = len(matrices[key0])
            for seed in SEEDS_ALL:
                k = f"{dset}|MLP|{method}|{seed}"
                if k not in matrices:
                    continue
                R = np.array(matrices[k])
                traj.append(R[:, 0])  # task-0 accuracy after each subsequent task
            traj = np.nanmean(np.array(traj), axis=0)
            ax.plot(range(1, T + 1), traj, marker="o", label=METHOD_LABELS[method], color=COLORS[method])
        ax.set_xlabel("Tasks learned so far")
        ax.set_ylabel("Accuracy on Task 0 (first-learned task)")
        ax.set_title(f"Retention of the first task while learning new ones -- {dset} (MLP)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        savefig(f"forgetting_curve_task0_{dset}.png")


def fig_heatmaps():
    with open(os.path.join(RESULTS_DIR, "accuracy_matrices.json")) as f:
        matrices = json.load(f)
    for dset in ["CuCD-ID", "STIN"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, method in zip(axes, ["finetune", "full"]):
            k = f"{dset}|MLP|{method}|0"
            R = np.array(matrices[k])
            T = R.shape[0]
            im = ax.imshow(R, vmin=0, vmax=1, cmap="viridis")
            for i in range(T):
                for j in range(T):
                    if not np.isnan(R[i, j]):
                        ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center",
                                color="white" if R[i, j] < 0.6 else "black", fontsize=8)
            ax.set_title(METHOD_LABELS[method])
            ax.set_xlabel("Evaluated task")
            ax.set_ylabel("After training task #")
            ax.set_xticks(range(T)); ax.set_yticks(range(T))
        fig.colorbar(im, ax=axes[-1], shrink=0.8, label="Test accuracy")
        fig.suptitle(f"Per-task accuracy matrix -- {dset} (MLP, seed 0)")
        savefig(f"accuracy_heatmap_{dset}.png")


def fig_memory_footprint():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "main_results_runs.csv"))
    sub = df[df.method == "full"].groupby(["dataset", "arch"]).agg(
        model_bytes_int8=("persistent_model_bytes_int8", "mean"),
        cmmu_bytes=("persistent_cmmu_bytes", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [f"{r.dataset}\n{r.arch}" for r in sub.itertuples()]
    x = np.arange(len(labels))
    ax.bar(x, sub.model_bytes_int8, label="Model weights (int8)", color="#4C72B0")
    ax.bar(x, sub.cmmu_bytes, bottom=sub.model_bytes_int8, label="CMMU (replay buffer + EWC, fp16)", color="#DD8452")
    ax.axhline(60 * 1024, color="red", linestyle="--", linewidth=1.2, label="60 KB budget")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Bytes")
    ax.set_title("Deployed memory footprint: model + CMMU vs. 60 KB budget")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    savefig("memory_footprint.png")


def fig_hparam_sweep_heatmap():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "hparam_sweep_stin.csv"))
    piv = df.groupby(["ewc_lambda", "buffer_capacity"])["ACC"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("Replay buffer capacity"); ax.set_ylabel("EWC lambda")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.3f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_title("Hyperparameter sweep (STIN, MLP): mean ACC")
    fig.colorbar(im, ax=ax, label="ACC")
    savefig("hparam_sweep_heatmap.png")


def fig_ewc_lambda_sensitivity():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_ewc_lambda.csv"))
    arch_colors = {"MLP": "#4C72B0", "GRU": "#B0413E", "CNN1D": "#55A868"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dset in zip(axes, ["CuCD-ID", "STIN"]):
        for arch, color in arch_colors.items():
            sub = df[(df.dataset == dset) & (df.arch == arch)].groupby("ewc_lambda")["ACC"].mean()
            ax.plot(sub.index.astype(str), sub.values, marker="o", label=arch, color=color)
        ax.set_xlabel("EWC lambda")
        ax.set_title(dset)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ACC")
    axes[0].legend()
    fig.suptitle("EWC lambda transfers poorly across architectures (method=full)\n"
                 "-- MLP is robust; GRU and CNN1D are both harmed by higher lambda, on both datasets")
    savefig("ewc_lambda_sensitivity.png")


def fig_buffer_size_sensitivity():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_buffer_size.csv"))
    arch_colors = {"MLP": "#4C72B0", "GRU": "#B0413E", "CNN1D": "#55A868"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dset in zip(axes, ["CuCD-ID", "STIN"]):
        for arch, color in arch_colors.items():
            sub = df[(df.dataset == dset) & (df.arch == arch)].groupby("buffer_capacity")["ACC"].mean()
            ax.plot(sub.index.astype(str), sub.values, marker="o", label=arch, color=color)
        ax.set_xlabel("Replay buffer capacity (samples)")
        ax.set_title(dset)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ACC")
    axes[0].legend()
    fig.suptitle("Replay buffer capacity sensitivity, all 3 architectures (method=full)")
    savefig("buffer_size_sensitivity.png")


def fig_per_class_f1():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "per_class_f1_summary.csv"))
    method_order = [m for m in METHOD_ORDER if m in df.method.unique()]

    # canonical class introduction order (not alphabetical) straight from the
    # dataset builders, so bars read left-to-right in the order classes were
    # actually learned
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import data_utils as du
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    cucd_df = du.load_cucd_id(os.path.join(data_dir, "CuCD-ID", "noised_dataset.csv"),
                               os.path.join(data_dir, "CuCD-ID", "label_schema.json"))
    stin_df = du.load_stin(os.path.join(data_dir, "STIN", "SAT20.csv"),
                            os.path.join(data_dir, "STIN", "TER20.csv"))
    canonical_order = {
        "CuCD-ID": du.build_cucd_id_tasks(cucd_df, seed=0).class_names_global,
        "STIN": du.build_stin_tasks(stin_df, seed=0).class_names_global,
    }

    for dset, arch in [("STIN", "GRU"), ("CuCD-ID", "MLP")]:
        sub = df[(df.dataset == dset) & (df.arch == arch)]
        class_order = [c for c in canonical_order[dset] if c in set(sub.class_name)]
        n_classes = len(class_order)
        n_methods = len(method_order)
        x = np.arange(n_classes)
        width = 0.8 / n_methods

        fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.3), 4.8))
        for i, method in enumerate(method_order):
            vals = [sub[(sub.class_name == c) & (sub.method == method)].f1_mean.values[0]
                    if len(sub[(sub.class_name == c) & (sub.method == method)]) else 0.0
                    for c in class_order]
            ax.bar(x + i * width, vals, width, label=METHOD_LABELS[method], color=COLORS[method])
        ax.set_xticks(x + width * (n_methods - 1) / 2)
        ax.set_xticklabels(class_order, rotation=30, ha="right")
        n_seeds_pc = int(sub.seed.nunique()) if "seed" in sub.columns else 0
        ax.set_ylabel(f"F1 (mean over {n_seeds_pc} seeds)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-class F1 -- {dset} / {arch}\n"
                     "(classes ordered left-to-right by introduction order; leftmost classes are learned first)")
        ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        ax.grid(axis="y", alpha=0.3)
        savefig(f"per_class_f1_{dset}_{arch}.png")


def fig_confusion_matrices():
    with open(os.path.join(RESULTS_DIR, "confusion_matrices.json")) as f:
        cms = json.load(f)
    for dset, arch in [("STIN", "GRU"), ("CuCD-ID", "MLP")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, method in zip(axes, ["finetune", "icarl"]):
            key = f"{dset}|{arch}|{method}|0"
            if key not in cms:
                ax.set_visible(False)
                continue
            entry = cms[key]
            cm = np.array(entry["confusion_matrix"], dtype=float)
            names = entry["class_names"]
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
            im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
            ax.set_title(METHOD_LABELS.get(method, method))
        handles = [a for a in axes if a.get_visible()]
        if handles:
            fig.colorbar(im, ax=handles[-1], shrink=0.8, label="Row-normalised fraction")
        fig.suptitle(f"Confusion matrix, final model -- {dset} / {arch} (seed 0)\n"
                     "Naive fine-tuning collapses everything into the most-recently-learned class(es)")
        savefig(f"confusion_matrix_{dset}_{arch}.png")


def fig_task_order():
    # Phase-2 disentangled study supersedes the legacy 3-perm confounded one.
    new_path = os.path.join(RESULTS_DIR, "task_order_disentangled.csv")
    if os.path.exists(new_path):
        df = pd.read_csv(new_path)
        if "degenerate_single_class_first" in df.columns:
            df = df[~df["degenerate_single_class_first"]].copy()
        agg = df.groupby(["budget_mode", "method"])["ACC"].agg(["mean", "std", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        modes = agg.budget_mode.unique()
        methods = [m for m in ["replay", "icarl"] if m in set(agg.method)]
        x = np.arange(len(modes))
        width = 0.35
        for i, m in enumerate(methods):
            s = agg[agg.method == m].set_index("budget_mode").loc[modes]
            ax.bar(x + i * width, s["mean"], width, yerr=ci95(s["std"], s["count"]),
                   label=METHOD_LABELS.get(m, m), color=COLORS.get(m, "gray"), capsize=4)
        ax.set_xticks(x + width / 2); ax.set_xticklabels(modes)
        ax.set_ylabel("ACC (mean +/- 95% t CI)")
        mode_means = df.groupby("budget_mode")["ACC"].mean()
        budget_gap = float(mode_means.max() - mode_means.min())
        order_means = df.groupby("order")["ACC"].mean()
        order_spread = float(order_means.max() - order_means.min())
        ax.set_title("Task-order study, primary two-class-first permutations "
                     "(CuCD-ID, MLP, "
                     f"{df.order.nunique()} perms x 2 budget modes)\n"
                     f"Budget-mode gap={budget_gap:.4f}; per-order mean spread={order_spread:.4f}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        savefig("task_order_sensitivity.png")
        return
    df = pd.read_csv(os.path.join(RESULTS_DIR, "legacy", "ablation_task_order.csv"))
    agg = df.groupby("order")[["ACC", "BWT"]].agg(["mean", "std", "count"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(agg))
    ax.bar(x, agg[("ACC", "mean")],
           yerr=ci95(agg[("ACC", "std")], agg[("ACC", "count")]),
           color="#2E7D32", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=20)
    ax.set_ylabel("ACC"); ax.set_ylim(0.9, 1.01)
    ax.set_title("Task-order sensitivity (CuCD-ID, MLP, full CMMU)")
    ax.grid(axis="y", alpha=0.3)
    savefig("task_order_sensitivity.png")


def fig_arch_comparison():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "main_results_runs.csv"))
    sub = df[df.method == "full"]
    agg = sub.groupby(["dataset", "arch"])["ACC"].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dsets = agg.dataset.unique()
    archs = ["MLP", "GRU", "CNN1D"]
    width = 0.35
    x = np.arange(len(archs))
    for i, dset in enumerate(dsets):
        s = agg[agg.dataset == dset].set_index("arch").loc[archs]
        ax.bar(x + i * width, s["mean"], width, yerr=ci95(s["std"], s["count"]), label=dset, capsize=4)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(archs)
    ax.set_ylabel("ACC (mean +/- 95% CI)"); ax.set_title("Architecture comparison (method = full CMMU)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig("architecture_comparison.png")


# ---------------- Phase-2 figures (each guarded by its CSV) ---------------- #
def fig_fewshot_pareto():
    path = os.path.join(RESULTS_DIR, "fewshot.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, arch in zip(axes, ["MLP", "GRU"]):
        sub = df[(df.arch == arch) & (df.calibration_protocol == "scaled")]
        if sub.empty:
            ax.set_visible(False)
            continue
        for method in [m for m in ["icarl", "bic", "replay", "full"] if m in set(sub.method)]:
            g = sub[sub.method == method].groupby("K").agg(
                uplink_KB=("uplink_KB_total_with_calib", "mean"),
                macro_f1=("MacroF1", "mean"))
            ax.plot(g.uplink_KB.values, g.macro_f1.values, marker="o",
                    label=METHOD_LABELS.get(method, method), color=COLORS.get(method, "gray"))
            for k, row in g.iterrows():
                ax.annotate(str(k), (row.uplink_KB, row.macro_f1), fontsize=6,
                            textcoords="offset points", xytext=(3, 3))
        ax.set_xlabel("Uplink KB per new class (adaptation + calibration)")
        ax.set_title(f"{arch} (CuCD-ID)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(fontsize=8)
    fig.suptitle("Few-shot uplink Pareto (scaled calibration): Macro-F1 vs uplink KB per new attack\n"
                 "point labels are K (samples/class); calibration data included in KB")
    savefig("fewshot_pareto.png")


def fig_arch_tune():
    path = os.path.join(RESULTS_DIR, "arch_tune.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    agg = df.groupby(["method", "lr", "epochs"])["ACC"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cfgs = sorted(set(zip(agg.lr, agg.epochs)))
    x = np.arange(len(cfgs))
    methods = [m for m in ["replay", "icarl", "joint_scratch"] if m in set(agg.method)]
    width = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        vals = [agg[(agg.method == m) & (agg.lr == lr) & (agg.epochs == ep)]["ACC"].mean()
                for lr, ep in cfgs]
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS.get(m, m), color=COLORS.get(m, "gray"))
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"lr={lr}, ep={ep}" for lr, ep in cfgs], rotation=20, ha="right")
    ax.set_ylabel("ACC (mean over 3 seeds)")
    ax.set_title("Per-arch tune, CuCD-ID/CNN: frozen hparams starve the conv trunk\n"
                 "lr=0.01 + 50 epochs closes most of the joint gap")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    savefig("arch_tune.png")


def fig_temporal_bracket():
    path = os.path.join(RESULTS_DIR, "temporal_purged_compare.csv")
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, "temporal_compare.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    group_col = "protocol" if "protocol" in df.columns else "split"
    cats = ["iid", "temporal_purged"] if group_col == "protocol" else ["random", "temporal"]
    agg = df.groupby(["method", group_col])["ACC"].mean().unstack()
    methods = [m for m in ["replay", "icarl", "bic", "joint_scratch"] if m in agg.index]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, split in enumerate(cats):
        vals = [agg.loc[m, split] if split in agg.columns else 0.0 for m in methods]
        ax.bar(x + i * width, vals, width, label=f"{split}",
               color="#4C72B0" if i == 0 else "#B0413E")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([METHOD_LABELS.get(m, m).split(" (")[0] for m in methods], rotation=15, ha="right")
    n_seeds = int(df.seed.nunique()) if "seed" in df.columns else 0
    ax.set_ylabel(f"ACC (mean over {n_seeds} seeds)")
    ax.set_title("IID vs forward-temporal-purged, CuCD-ID/MLP\n"
                 "separate estimands: benchmark vs generalization under shift (purge 0.02 rad TimeRadians)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig("temporal_bracket.png")


def fig_temporal_leakage_audit():
    summary_path = os.path.join(RESULTS_DIR, "temporal_leakage_audit_summary.csv")
    hist_path = os.path.join(RESULTS_DIR, "temporal_leakage_histogram.csv")
    ranges_path = os.path.join(RESULTS_DIR, "temporal_class_block_ranges.csv")
    blocks_path = os.path.join(RESULTS_DIR, "temporal_block_coverage.csv")
    if not all(os.path.exists(p) for p in
               (summary_path, hist_path, ranges_path, blocks_path)):
        return
    summary = pd.read_csv(summary_path)
    hist = pd.read_csv(hist_path)
    ranges = pd.read_csv(ranges_path)
    blocks = pd.read_csv(blocks_path)
    classes = list(summary.label)
    colors = dict(zip(classes, ["#4C72B0", "#DD8452", "#55A868",
                                "#C44E52", "#8172B3"]))
    purge = 0.02

    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 4.9),
        gridspec_kw={"width_ratios": [1.12, 1.0]})
    ax = axes[0]
    for label in classes:
        sub = hist[hist.label == label].sort_values("bin_left_rad")
        centers = np.sqrt(sub.bin_left_rad.to_numpy() * sub.bin_right_rad.to_numpy())
        frac = float(summary.loc[summary.label == label, "frac_within_purge"].iloc[0])
        ax.step(centers, sub.fraction.to_numpy(), where="mid",
                color=colors[label], linewidth=1.5,
                label=f"{label} ({100*frac:.1f}% ≤ purge)")
    ax.axvline(purge, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(purge, ax.get_ylim()[1] * 0.96, " 0.02 rad purge",
            rotation=90, va="top", ha="left", fontsize=8, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("nearest training-row |Δt| (rad, log scale)")
    ax.set_ylabel("fraction of test rows per bin")
    ax.set_title("Random split is temporal interpolation\n"
                 "pooled over 15 splits; distance to rows actually trained on")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    for i, (_, block) in enumerate(blocks.iterrows()):
        ax.axvspan(block.time_min_rad, block.time_max_rad,
                   color="#BDBDBD" if i % 2 == 0 else "#E8E8E8",
                   alpha=0.28, linewidth=0)
        ax.text((block.time_min_rad + block.time_max_rad) / 2,
                len(classes) + 0.08 + 0.22 * (i % 2), block.block_id,
                ha="center", va="bottom", fontsize=8, color="#555555")
    for yi, label in enumerate(classes):
        sub = ranges[ranges.label == label]
        for _, row in sub.iterrows():
            lw = 1.0 + 3.0 * np.log1p(row.n_rows) / np.log1p(
                ranges.n_rows.max())
            ax.plot([row.time_min_rad, row.time_max_rad], [yi, yi],
                    color=colors[label], linewidth=lw, solid_capstyle="butt")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    x_pad = 0.10
    ax.set_xlim(blocks.time_min_rad.min() - x_pad,
                blocks.time_max_rad.max() + x_pad)
    ax.set_ylim(-0.6, len(classes) + 0.5)
    ax.set_xlabel("unwrapped TimeRadians (rad; origin after largest gap)")
    ax.set_title("Natural time blocks are class-incomplete\n"
                 "segments show each class's occupied time within a block")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Temporal leakage/validity audit (read-only; no models trained)",
                 fontsize=11)
    savefig("temporal_leakage_audit.png")


def fig_robustness():
    path = os.path.join(RESULTS_DIR, "robustness.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    order = ["clean", "gauss_0.1", "gauss_0.5", "drop_0.1", "drop_0.3", "nowindow", "int8_sim", "int8"]
    order = [c for c in order if c in set(df.corruption)]
    methods = [m for m in ["replay", "icarl", "bic", "joint_scratch"] if m in set(df.method)]
    x = np.arange(len(order))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(methods):
        vals = [df[(df.method == m) & (df.corruption == c)]["ACC_final_pool"].mean() for c in order]
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS.get(m, m), color=COLORS.get(m, "gray"))
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(order, rotation=20, ha="right")
    n_seeds_rb = int(df.seed.nunique()) if "seed" in df.columns else 0
    ax.set_ylabel(f"Pooled ACC (mean over {n_seeds_rb} seeds)")
    ax.set_title("Robustness ledger, CuCD-ID/MLP: simulated corruptions + simulated INT8 (int8_sim)\n"
                 "window-masking costs ~0.3 (aggregate-feature reliance — see leakage limit)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    savefig("robustness.png")


def fig_leakage_ablation():
    path = os.path.join(RESULTS_DIR, "leakage_ablation_summary.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    variant_order = ["all", "no_time", "no_ids", "no_window", "no_time_ids_window"]
    variants = [v for v in variant_order if v in set(df.variant)]
    methods = [m for m in ["replay", "bic", "joint_scratch"] if m in set(df.method)]
    x = np.arange(len(variants))
    width = 0.8 / max(1, len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, metric, label in [(axes[0], "ACC_mean", "ACC"),
                              (axes[1], "MacroF1_mean", "Macro-F1")]:
        for i, m in enumerate(methods):
            vals = [df[(df.variant == v) & (df.method == m)][metric].mean()
                    for v in variants]
            ax.bar(x + i * width, vals, width, label=METHOD_LABELS.get(m, m),
                   color=COLORS.get(m, "gray"))
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(variants, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    raw_path = os.path.join(RESULTS_DIR, "leakage_ablation.csv")
    n_seeds = (int(pd.read_csv(raw_path)["seed"].nunique())
               if os.path.exists(raw_path) else 10)
    fig.suptitle("Feature-provenance leakage ablation, CuCD-ID/MLP "
                 f"(retrained, mean over {n_seeds} seeds)\n"
                 "removing time / identifiers / 20-s window aggregates")
    savefig("leakage_ablation.png")


def main():
    fig_method_comparison_bars()
    fig_forgetting_curves()
    fig_heatmaps()
    fig_memory_footprint()
    fig_hparam_sweep_heatmap()
    fig_ewc_lambda_sensitivity()
    fig_buffer_size_sensitivity()
    fig_task_order()
    fig_arch_comparison()
    fig_per_class_f1()
    fig_confusion_matrices()
    # Phase-2 (no-ops when their CSVs are absent)
    fig_fewshot_pareto()
    fig_arch_tune()
    fig_temporal_bracket()
    fig_temporal_leakage_audit()
    fig_robustness()
    fig_leakage_ablation()


if __name__ == "__main__":
    main()
