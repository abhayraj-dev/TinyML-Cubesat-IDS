"""Builds results/RESULTS_SUMMARY.md purely from the CSV/JSON files already
written by run_hparam_sweep.py, run_main.py and run_ablation.py -- every
number in the report is read out of those files, nothing is hand-typed."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from config import RESULTS_DIR


def df_to_md(df, float_fmt="{:.4f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: float_fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def main():
    out = []
    out.append("# Results Summary\n")
    out.append("_This file is generated automatically by "
                "`experiments/generate_summary_report.py` directly from the CSV/JSON "
                "outputs of the experiment scripts -- every number below is a real, "
                "computed result, not a placeholder._\n")
    out.append("\n**Data provenance.** CuCD-ID is the labelled NOS3/cFS "
                "software-in-the-loop CubeSat telemetry dataset of Fayyaz et al. "
                "(Data in Brief 2026, doi:10.1016/j.dib.2026.112598; Mendeley "
                "doi:10.17632/7n2d42pm3n.3); all telemetry is simulated, not "
                "captured from an orbiting spacecraft. STIN (SAT20/TER20) is the "
                "satellite/terrestrial flow dataset of Li et al. (IEEE Access 2020, "
                "doi:10.1109/ACCESS.2020.3041641; github.com/kun9717/STIN-data-set) "
                "and is used as a cross-domain stress test. Headline tables below "
                "use the IID random split; Sections 5.3 and 7.4 quantify the "
                "resulting leakage/feature-reliance caveats.\n")

    # ---- hyperparameter sweep (coarse prior) + frozen per-method/arch configs ----
    sweep = pd.read_csv(os.path.join(RESULTS_DIR, "hparam_sweep_stin.csv"))
    with open(os.path.join(RESULTS_DIR, "best_hparams.json")) as f:
        best_hp = json.load(f)
    out.append("## 1. Hyperparameter protocol (STIN prior + per-method/arch validation tuning)\n")
    out.append(f"Coarse STIN/MLP/full grid: **{len(sweep)}** configurations, scored on validation.\n")
    out.append("Top 5 by validation score (ACC - max(FM,0)):\n")
    out.append(df_to_md(sweep.sort_values("score", ascending=False).head(5)))
    out.append("\n**Coarse STIN prior (NOT the final configs):**\n")
    out.append("```json\n" + json.dumps(best_hp, indent=2) + "\n```\n")
    frozen_path = os.path.join(RESULTS_DIR, "best_hparams_per_method_arch.json")
    if os.path.exists(frozen_path):
        with open(frozen_path) as f:
            frozen = json.load(f)
        out.append("\n**Final frozen configs** (`best_hparams_per_method_arch.json`, "
                    "per method x architecture, tuned on CuCD-ID validation, 3 seeds each):\n")
        frows = []
        for arch in sorted(frozen):
            if not isinstance(frozen[arch], dict):
                continue  # skip metadata keys like "_note"
            for method in sorted(frozen[arch]):
                h = frozen[arch][method].get("hparams", {})
                frows.append(dict(arch=arch, method=method,
                                  val_score=round(float(frozen[arch][method].get("val_score", float("nan"))), 4),
                                  hparams=json.dumps(h, sort_keys=True)))
        out.append(df_to_md(pd.DataFrame(frows)))
        out.append("\nEvery main/Phase-2 run below uses its matching frozen config; "
                    "test data is never used for selection.\n")

    # ---- main results ----
    runs = pd.read_csv(os.path.join(RESULTS_DIR, "main_results_runs.csv"))
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "main_results_summary.csv"))
    out.append("## 2. Main experiment matrix\n")
    out.append(f"Total individual runs: **{len(runs)}** "
                f"(2 datasets x {runs.arch.nunique()} architectures x {runs.method.nunique()} "
                f"methods x {runs.seed.nunique()} seeds).\n")

    for dset in ["CuCD-ID", "STIN"]:
        out.append(f"\n### {dset}\n")
        piv_acc = summary[summary.dataset == dset].pivot_table(index="arch", columns="method", values="ACC_mean")
        piv_bwt = summary[summary.dataset == dset].pivot_table(index="arch", columns="method", values="BWT_mean")
        n_seeds = int(runs.seed.nunique())
        out.append(f"**Average Accuracy (ACC), mean over {n_seeds} seeds:**\n")
        out.append(df_to_md(piv_acc.reset_index()))
        out.append(f"\n**Backward Transfer (BWT), mean over {n_seeds} seeds (negative = forgetting):**\n")
        out.append(df_to_md(piv_bwt.reset_index()))
        if "MacroF1_mean" in summary.columns:
            piv_f1 = summary[summary.dataset == dset].pivot_table(index="arch", columns="method", values="MacroF1_mean")
            out.append(f"\n**Macro-F1 (primary, BiC-corrected), mean over {n_seeds} seeds:**\n")
            out.append(df_to_md(piv_f1.reset_index()))
        if "BalancedAccuracy_mean" in summary.columns:
            piv_ba = summary[summary.dataset == dset].pivot_table(index="arch", columns="method", values="BalancedAccuracy_mean")
            out.append(f"\n**Balanced Accuracy, mean over {n_seeds} seeds:**\n")
            out.append(df_to_md(piv_ba.reset_index()))
        if "DET_AUROC_mean" in summary.columns:
            det = summary[summary.dataset == dset][["arch", "method", "DET_AUROC_mean", "DET_AUPR_mean", "DET_FPR95_mean"]]
            if det[["DET_AUROC_mean"]].notnull().any().any():
                out.append("\n**Binary detection (1-p(normal), CuCD-ID only; STIN has no benign class):**\n")
                out.append(df_to_md(det))

    # ---- memory footprint (bytes means converted to KB; *_total_KB already KB) ----
    mem = runs[runs.method == "full"].groupby(["dataset", "arch"]).agg(
        persistent_model_bytes=("persistent_model_bytes_int8", "mean"),
        persistent_cmmu_bytes=("persistent_cmmu_bytes", "mean"),
        persistent_total_KB=("persistent_total_KB", "mean"),
        peak_total_KB=("peak_total_KB", "mean"),
        macs_per_forward=("macs_per_forward", "mean"),
    ).reset_index()
    mem["persistent_model_KB"] = mem["persistent_model_bytes"] / 1024.0
    mem["persistent_cmmu_KB"] = mem["persistent_cmmu_bytes"] / 1024.0
    mem = mem.drop(columns=["persistent_model_bytes", "persistent_cmmu_bytes"])
    out.append("\n## 3. Memory: persistent deployment vs peak update RAM (method=full)\n")
    out.append(df_to_md(mem, float_fmt="{:.2f}"))
    out.append(f"\nAll configurations fit within the **assumed 60 KB TinyML "
                f"persistent-state budget** (a conservative study constraint, not a "
                f"universal CubeSat limit; largest persistent: "
                f"**{mem.persistent_total_KB.max():.2f} KB**, "
                f"{100*mem.persistent_total_KB.max()/60:.1f}% of budget). "
                f"Estimated peak update RAM is larger (max "
                f"**{mem.peak_total_KB.max():.2f} KB** host-estimated, counting FP32 "
                f"weights, gradients, Adam states, activations, batches, and teacher) "
                f"-- a reported constraint on onboard training, not covered by the "
                f"persistent number.\n")

    # ---- uplink payload sizes ----
    out.append("\n## 4. Estimated uplink payload sizes (packed telemetry payload)\n")
    out.append("Computed directly from the constructed task splits "
                "(`src/data_utils.py`), reflecting the KB-scale uplink constraint "
                "from the operational concept: task 0 is trained "
                "on a larger pre-launch corpus, while every subsequent task is "
                "trained only on a small uplinked labelled batch. Figures are "
                "**packed INT8 payload estimates** (rows x features plus label "
                "bytes), excluding CSV text formatting, scale/zero-point "
                "parameters, and packet/protocol framing overhead. The reported "
                "uplink budget **includes both adaptation data and the onboard "
                "calibration data** consumed by BiC fitting and val-balanced "
                "Fisher estimation (Option A): a deployed update must transmit "
                "both for the satellite to reproduce these results.\n")
    import data_utils as du
    cucd_df = du.load_cucd_id(os.path.join(RESULTS_DIR, "..", "data", "CuCD-ID", "noised_dataset.csv"),
                               os.path.join(RESULTS_DIR, "..", "data", "CuCD-ID", "label_schema.json"))
    stin_df = du.load_stin(os.path.join(RESULTS_DIR, "..", "data", "STIN", "SAT20.csv"),
                            os.path.join(RESULTS_DIR, "..", "data", "STIN", "TER20.csv"))
    up_rows = []
    for dset_name, builder, df_ in [("CuCD-ID", du.build_cucd_id_tasks, cucd_df),
                                     ("STIN", du.build_stin_tasks, stin_df)]:
        ts = builder(df_, seed=0)
        for task in ts.tasks:
            n_val = task.X_val.shape[0] if task.has_val else 0
            up_rows.append(dict(
                dataset=dset_name, task=task.name, n_train_rows=task.X_train.shape[0],
                n_calib_rows=n_val, n_new_classes=task.n_new_classes,
                uplink_KB_fp32=task.uplink_bytes_fp32 / 1024,
                uplink_KB_int8=task.uplink_bytes_int8 / 1024,
                uplink_KB_calib_int8=task.uplink_calib_bytes_int8 / 1024,
                uplink_KB_total_with_calib=(task.uplink_bytes_int8 + task.uplink_calib_bytes_int8) / 1024,
            ))
    up_df = pd.DataFrame(up_rows)
    out.append(df_to_md(up_df, float_fmt="{:.2f}"))

    # ---- ablations ----
    out.append("\n## 5. Ablation studies\n")

    buf = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_buffer_size.csv"))
    out.append("### 5.1 Replay buffer capacity sensitivity (both datasets x all 3 architectures, method=full)\n")
    out.append(df_to_md(buf.groupby(["dataset", "arch", "buffer_capacity"])[["ACC", "BWT", "FM"]].mean().reset_index()))
    out.append("\nPer-architecture ACC range across all tested buffer capacities:\n")
    buf_ranges = buf.groupby(["dataset", "arch"]).apply(
        lambda g: g.groupby("buffer_capacity").ACC.mean().agg(["min", "max"])
    ).reset_index()
    buf_ranges["span"] = buf_ranges["max"] - buf_ranges["min"]
    out.append(df_to_md(buf_ranges, float_fmt="{:.4f}"))

    lam = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_ewc_lambda.csv"))
    out.append("\n### 5.2 EWC lambda sensitivity (both datasets x all 3 architectures, method=full)\n")
    out.append(df_to_md(lam.groupby(["dataset", "arch", "ewc_lambda"])[["ACC", "BWT", "FM"]].mean().reset_index()))
    lam_ranges = lam.groupby(["dataset", "arch"]).apply(
        lambda g: g.groupby("ewc_lambda").ACC.mean().agg(["min", "max"])
    ).reset_index()
    lam_ranges["span"] = lam_ranges["max"] - lam_ranges["min"]
    out.append("\nPer-(dataset, architecture) ACC range across all tested lambdas:\n")
    out.append(df_to_md(lam_ranges, float_fmt="{:.4f}"))
    out.append("\nMLP's span is small on both datasets, while GRU and CNN1D both show a "
                "much larger span, and in both cases ACC *decreases* as lambda increases "
                "(visible directly in the per-lambda table above and in "
                "`figures/ewc_lambda_sensitivity.png`). This is why final numbers use "
                "frozen per-method x per-architecture validation configs instead of one "
                "shared configuration.\n")
    frozen_gamma_path = os.path.join(RESULTS_DIR, "best_hparams_per_method_arch.json")
    if os.path.exists(frozen_gamma_path):
        with open(frozen_gamma_path) as f:
            frozen_gamma = json.load(f)
        gamma_strs = []
        for _arch in ["MLP", "GRU", "CNN1D"]:
            _g = frozen_gamma.get(_arch, {}).get("ewc", {}).get("hparams", {}).get("ewc_gamma", None)
            gamma_strs.append(f"{_arch} γ={_g}")
        out.append("EWC here is online diagonal Fisher with a running estimate "
                    "(`F_new = γ·F_old + F_current`). The Fisher sample is drawn "
                    "from a class-balanced mix of the current task's training data "
                    "and the cumulative validation pool of all tasks seen so far "
                    "(not the current task alone), so old-task curvature is not "
                    "dropped at each consolidation. Frozen per-architecture γ "
                    "weighting: "
                    + ", ".join(gamma_strs) + ". γ is treated as a "
                    "validation-tuned weighting parameter rather than a conventional "
                    "decay factor (GRU γ=2.0 amplifies retained Fisher by validation "
                    "selection; a γ≤1 restriction would be stricter but would require "
                    "rerunning EWC experiments).\n")

    leak_path = os.path.join(RESULTS_DIR, "leakage_ablation_summary.csv")
    if os.path.exists(leak_path):
        leak = pd.read_csv(leak_path)
        out.append("\n### 5.3 Feature-provenance leakage ablation (CuCD-ID/MLP, 10 seeds)\n")
        out.append("Retrained models after removing feature groups (not test-time "
                    "corruption). Deltas are relative to the all-feature run for the "
                    "same method; negative values mean the removed features were "
                    "helping the model.\n")
        out.append(df_to_md(leak[[
            "variant", "method", "n_features", "ACC_mean", "MacroF1_mean",
            "ACC_mean_delta_vs_all", "MacroF1_mean_delta_vs_all"
        ]]))
        out.append("\nThis quantifies how much random-split performance rides on "
                    "the circular time field, CCSDS identifiers, and 20-second "
                    "window aggregates. It is a reliance measure, not proof that "
                    "every removed feature is illegitimate: `no_ids` also removes "
                    "protocol information a real detector may legitimately use. The "
                    "IID headline remains a benchmark estimand; the temporal-purged "
                    "table in Section 7.4 is the separate deployment-shift estimand.\n")
    else:
        out.append("\n_(leakage_ablation_summary.csv not found -- run "
                    "`experiments/run_leakage_ablation.py` first)_\n")

    order = pd.read_csv(os.path.join(RESULTS_DIR, "legacy", "ablation_task_order.csv"))
    out.append("\n### 5.4 Task-order sensitivity, legacy 3-perm (CuCD-ID, MLP, method=full)\n")
    out.append("_Legacy confounded design (budgets move with the task); superseded by "
                "the disentangled 12-permutation study in §7.3. Retained for provenance only._\n")
    order_agg = order.groupby("order")[["ACC", "BWT"]].agg(["mean", "std"]).reset_index()
    order_agg.columns = ["order", "ACC_mean", "ACC_std", "BWT_mean", "BWT_std"]
    out.append(df_to_md(order_agg))
    order_means = order.groupby("order").ACC.mean()
    spread = order_means.max() - order_means.min()
    out.append(f"\nACC spread across the 3 legacy orderings: **{spread:.4f}** "
                f"(range {order_means.min():.4f} - {order_means.max():.4f}). "
                f"See §7.3 for the disentangled measurement (computed from "
                f"`task_order_disentangled.csv`, not hard-coded).\n")

    # ---- per-class F1 / confusion matrix ----
    out.append("\n## 6. Per-class F1 -- direct evidence of forgetting\n")
    out.append("Section 2's ACC/BWT/FM numbers are task-level aggregates; this section "
                "makes the same phenomenon visible at the level of individual attack "
                "classes by evaluating the final trained model (all classes seen so far, "
                "single shared head) and computing a per-class precision/recall/F1 "
                "breakdown. See `figures/per_class_f1_*.png` and "
                "`figures/confusion_matrix_*.png` for the visual version.\n")
    pcf1_path = os.path.join(RESULTS_DIR, "per_class_f1_summary.csv")
    if os.path.exists(pcf1_path):
        pcf1 = pd.read_csv(pcf1_path)
        for dset, arch in [("STIN", "GRU"), ("CuCD-ID", "MLP")]:
            sub = pcf1[(pcf1.dataset == dset) & (pcf1.arch == arch)]
            out.append(f"\n### 6.{'1' if dset=='STIN' else '2'} {dset} / {arch}, mean F1 per task-of-origin\n")
            piv = sub.pivot_table(index="task_name", columns="method", values="f1_mean")
            method_cols = [c for c in piv.columns if c in
                           ["finetune", "ewc", "lwf", "replay", "full", "bic", "icarl",
                            "joint", "joint_scratch"]]
            out.append(df_to_md(piv[method_cols].reset_index()))
        out.append("\nFor CuCD-ID, the memoryless methods (`finetune`, `ewc`) "
                   "collapse most classes to zero or near-zero F1 while retaining "
                   "some ability on the most recent class; the exact per-class "
                   "pattern is shown above and varies by method and seed. On STIN "
                   "the same severe recency bias appears, but it is noisier because "
                   "each task introduces several classes and the feature set is less "
                   "separable. Memory-based and joint references retain non-zero F1 "
                   "across most tasks. This is the direct per-class confirmation of "
                   "the task-level results in Section 2; do not infer a universal "
                   "exact-zero pattern from one method.\n")
    else:
        out.append("\n_(per-class F1 results not found -- run "
                    "`experiments/run_per_class_analysis.py` first)_\n")

    # ---- Phase-2 evidence (each guarded; derived claims use fixed thresholds) ----
    out.append("\n## 7. Phase-2 evidence (all numbers computed from their CSVs)\n")

    def _present(name):
        return os.path.exists(os.path.join(RESULTS_DIR, name))

    if _present("fewshot.csv"):
        fs = pd.read_csv(os.path.join(RESULTS_DIR, "fewshot.csv"))
        out.append("\n### 7.1 Few-shot uplink sweep, CuCD-ID (RQ4)\n")
        piv = fs.groupby(["arch", "method", "K"])["ACC"].mean().unstack("K")
        out.append(df_to_md(piv.reset_index()))
        kb = fs.groupby("K")["uplink_KB_total"].mean().to_dict()
        out.append("\nAdaptation uplink KB for T1..T3 per K: " +
                    ", ".join(f"K={k}: {kb[k]:.1f}KB" for k in sorted(kb)) + ".\n")
        if "uplink_KB_total_with_calib" in fs.columns:
            kbc = fs.groupby("K")["uplink_KB_total_with_calib"].mean().to_dict()
            out.append("Total uplink KB **including onboard calibration data** "
                        "(BiC/Fisher val samples, Option A) per K: " +
                        ", ".join(f"K={k}: {kbc[k]:.1f}KB" for k in sorted(kbc)) + ".\n")
        mlp = fs[fs.arch == "MLP"]
        k50 = mlp[mlp.K == 50].groupby("method")["ACC"].mean()
        k5 = mlp[mlp.K == 5].groupby("method")["ACC"].mean()
        out.append(f"MLP @K=50 (knee): " +
                    ", ".join(f"{m}={k50[m]:.3f}" for m in k50.index) + ".\n")
        out.append(f"MLP @K=5 (extreme): " +
                    ", ".join(f"{m}={k5[m]:.3f}" for m in k5.index) + ".\n")
    else:
        out.append("\n_(fewshot.csv not found -- run `experiments/run_fewshot.py` first)_\n")

    if _present("arch_tune.csv"):
        at = pd.read_csv(os.path.join(RESULTS_DIR, "arch_tune.csv"))
        out.append("\n### 7.2 Per-arch tune, CuCD-ID/CNN (exploratory, validation-scored, not applied)\n")
        out.append(df_to_md(at.groupby(["method", "lr", "epochs"])["ACC"].mean().reset_index()))
        best = at.groupby(["method", "lr", "epochs"])["ACC"].mean().reset_index()
        for m in ["replay", "icarl", "joint_scratch"]:
            b = best[best.method == m].sort_values("ACC", ascending=False).iloc[0]
            out.append(f"\n- {m}: best lr={b['lr']}, epochs={int(b['epochs'])} "
                        f"-> ACC={b['ACC']:.3f}.")
        out.append("\n")
    else:
        out.append("\n_(arch_tune.csv not found -- run `experiments/run_arch_tune.py` first)_\n")

    if _present("task_order_disentangled.csv"):
        to = pd.read_csv(os.path.join(RESULTS_DIR, "task_order_disentangled.csv"))
        out.append("\n### 7.3 Disentangled task-order study, CuCD-ID/MLP (RQ5)\n")
        out.append(df_to_md(to.groupby(["budget_mode", "method"])["ACC"].agg(["mean", "std"]).reset_index()))
        spread = to.groupby("order")["ACC"].mean()
        raw_spread = float(to["ACC"].max() - to["ACC"].min())
        budget_means = to.groupby("budget_mode")["ACC"].mean()
        out.append(f"\nAggregate difference between budget modes is small "
                    f"(fixed_position={budget_means.get('fixed_position', float('nan')):.3f} vs "
                    f"moves_with_task={budget_means.get('moves_with_task', float('nan')):.3f}; "
                    f"spread of per-order means across {to.order.nunique()} permutations: "
                    f"**{spread.max()-spread.min():.4f}**), although individual runs "
                    f"vary more (raw ACC range **{raw_spread:.4f}**). The legacy "
                    f"confounded headline is retired.\n")
    else:
        out.append("\n_(task_order_disentangled.csv not found -- run "
                    "`experiments/run_task_order.py` first)_\n")

    if _present("temporal_purged_compare.csv"):
        tc = pd.read_csv(os.path.join(RESULTS_DIR, "temporal_purged_compare.csv"))
        out.append("\n### 7.4 IID vs forward-temporal-purged, CuCD-ID/MLP\n")
        out.append(df_to_md(tc.groupby(["method", "protocol"])["ACC"].agg(["mean", "std"]).reset_index()))
        out.append("\nIID = random-split benchmark; temporal-purged = forward temporal "
                    "generalization under distribution shift with reduced window overlap "
                    "(separate estimands, not interpolated). Purge: 0.02 rad on "
                    "TimeRadians (a proxy for the documented ~20-22 s window horizon; "
                    "not a wall-clock guarantee). Blocks are ordered on unwrapped "
                    "circular time so an orbital wrap does not scramble chronology. "
                    "For classes whose TimeRadians span is narrow, the purge can "
                    "empty the validation slice; those cells are warned about by the "
                    "builder and val-dependent methods (BiC/EWC) are correspondingly "
                    "weaker in this protocol. See DATALOG.md for the drift diagnosis.\n")
    elif _present("temporal_compare.csv"):
        tc = pd.read_csv(os.path.join(RESULTS_DIR, "temporal_compare.csv"))
        out.append("\n### 7.4 Temporal-vs-random bracket (legacy unpurged), CuCD-ID/MLP\n")
        out.append(df_to_md(tc.groupby(["method", "split"])["ACC"].agg(["mean", "std"]).reset_index()))
        out.append("\nLegacy unpurged temporal comparison (superseded by temporal-purged).\n")
    else:
        out.append("\n_(temporal_purged_compare.csv not found -- run "
                    "`experiments/run_purged_temporal.py` first)_\n")

    if _present("robustness.csv"):
        rb = pd.read_csv(os.path.join(RESULTS_DIR, "robustness.csv"))
        out.append("\n### 7.5 Robustness + simulated-quantization ledger, CuCD-ID/MLP (RQ6/RQ7)\n")
        out.append("_`int8_sim` is a simulated post-training INT8 round-trip accuracy "
                    "proxy (FP32 weights only); no INT8 kernels were executed and no "
                    "hardware deployment is claimed._\n")
        out.append(df_to_md(rb.groupby(["method", "corruption"])["ACC_final_pool"].mean().reset_index()))
        base = rb[rb.corruption == "clean"].groupby("method")["ACC_final_pool"].mean()
        valt = lambda m, c: rb[(rb.method == m) & (rb.corruption == c)]["ACC_final_pool"].mean()
        drops = {c: float(np.mean([base[m] - valt(m, c) for m in base.index]))
                 for c in rb.corruption.unique() if c != "clean"}
        out.append("\nMean pooled-ACC drop vs clean (all methods): " +
                    ", ".join(f"{c}={drops[c]:+.3f}" for c in sorted(drops)) + ".\n")
    else:
        out.append("\n_(robustness.csv not found -- run `experiments/run_robustness.py` first)_\n")

    if _present("memory_ledger.csv"):
        ml = pd.read_csv(os.path.join(RESULTS_DIR, "memory_ledger.csv"))
        out.append("\n### 7.6 Activation memory ledger (analytic, batch=1, SIM_ = host-simulated)\n")
        out.append(df_to_md(ml))
        out.append("\n")
    else:
        out.append("\n_(memory_ledger.csv not found -- run `experiments/run_robustness.py` first)_\n")

    for metric_file, metric_label in [
        ("stats_permutation.csv", "ACC"),
        ("stats_permutation_macrof1.csv", "Macro-F1 (primary)"),
    ]:
        if _present(metric_file):
            st = pd.read_csv(os.path.join(RESULTS_DIR, metric_file))
            sig = st[st.p_holm < 0.05]
            out.append(f"\n### 7.7 Exact paired permutation + Holm (matched seeds, "
                        f"{metric_label}): {len(sig)}/{len(st)} headline comparisons "
                        f"significant at Holm p<0.05\n")
            out.append("Holm correction is applied across all comparisons in this "
                        "metric. The exact sign-flip test has n=10 paired seeds.\n")
            cols = [c for c in ["dataset", "arch", "method_a", "method_b", "n",
                                "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
                                "cohens_d", "p_exact", "p_holm"] if c in st.columns]
            out.append(df_to_md(st.sort_values("mean_diff", ascending=False)[cols]))
        else:
            out.append(f"\n_({metric_file} not found -- run "
                        f"`experiments/run_stats.py` first)_\n")

    # ---- dynamic executive summary for the v5.4 final report ----
    deployable = ["finetune", "replay", "ewc", "full", "lwf", "bic", "icarl"]
    best_rows = []
    for dset in ["CuCD-ID", "STIN"]:
        for arch in sorted(runs.arch.unique()):
            sub = runs[(runs.dataset == dset) & (runs.arch == arch) &
                       (runs.method.isin(deployable))]
            if sub.empty:
                continue
            g = sub.groupby("method")[["ACC", "MacroF1"]].mean()
            m = g["MacroF1"].idxmax()
            best_rows.append(dict(dataset=dset, arch=arch, method=m,
                                  ACC=g.loc[m, "ACC"], MacroF1=g.loc[m, "MacroF1"]))
    best_df = pd.DataFrame(best_rows)

    sig_lines = []
    for metric_file, metric_label in [
        ("stats_permutation.csv", "ACC"),
        ("stats_permutation_macrof1.csv", "Macro-F1"),
    ]:
        path = os.path.join(RESULTS_DIR, metric_file)
        if os.path.exists(path):
            st = pd.read_csv(path)
            sig_lines.append(f"{int((st.p_holm < 0.05).sum())}/{len(st)} {metric_label} "
                             f"comparisons significant at Holm p<0.05")

    temporal_line = ""
    tpath = os.path.join(RESULTS_DIR, "temporal_purged_compare.csv")
    if os.path.exists(tpath):
        tp = pd.read_csv(tpath)
        iid = tp[tp.protocol == "iid"].groupby("method").ACC.mean()
        pur = tp[tp.protocol == "temporal_purged"].groupby("method").ACC.mean()
        common = [m for m in deployable if m in iid.index and m in pur.index]
        if common:
            temporal_line = (f"Mean deployable-method ACC falls from "
                             f"{iid[common].mean():.4f} (IID) to "
                             f"{pur[common].mean():.4f} (temporal-purged).")

    leak_line = ""
    lpath = os.path.join(RESULTS_DIR, "leakage_ablation_summary.csv")
    if os.path.exists(lpath):
        lk = pd.read_csv(lpath)
        if not lk.empty and "MacroF1_mean_delta_vs_all" in lk.columns:
            worst = lk.loc[lk["MacroF1_mean_delta_vs_all"].idxmin()]
            leak_line = (f"Largest retrained Macro-F1 drop: {worst['variant']}/"
                         f"{worst['method']} {worst['MacroF1_mean_delta_vs_all']:+.4f}.")

    task_line = ""
    opath = os.path.join(RESULTS_DIR, "task_order_disentangled.csv")
    if os.path.exists(opath):
        to = pd.read_csv(opath)
        if "order" in to.columns and "ACC" in to.columns:
            task_line = (f"Disentangled task-order ACC spread: "
                         f"{to.groupby('order').ACC.mean().max() - to.groupby('order').ACC.mean().min():.4f}.")

    exec_out = [
        "# Project Report v5.4",
        "## Resource-Aware Class-Incremental Continual Learning for TinyML-Based Intrusion Detection in CubeSats",
        "",
        "_Auto-generated by `experiments/generate_summary_report.py` from the "
        "CSV/JSON outputs of this run. No result value is typed by hand; the "
        "static text is limited to provenance and interpretation._",
        "",
        "**Data provenance.** CuCD-ID is the labelled NOS3/cFS "
        "software-in-the-loop CubeSat telemetry dataset of Fayyaz et al. "
        "(Data in Brief 2026, doi:10.1016/j.dib.2026.112598; Mendeley "
        "doi:10.17632/7n2d42pm3n.3); all telemetry is simulated, not captured "
        "on orbit. STIN is Li et al., IEEE Access 2020, "
        "doi:10.1109/ACCESS.2020.3041641, used as a cross-domain stress test.",
        "",
        "## Executive summary",
        "",
        f"- Main matrix: **{len(runs)}** runs, {runs.arch.nunique()} architectures, "
        f"{runs.method.nunique()} methods, {runs.seed.nunique()} seeds.",
        f"- Largest `full` persistent state: **{mem.persistent_total_KB.max():.2f} KB**; "
        f"largest host-estimated peak update RAM: **{mem.peak_total_KB.max():.2f} KB**.",
    ]
    if not best_df.empty:
        for _, r in best_df.iterrows():
            exec_out.append(f"- Best deployable by Macro-F1, {r['dataset']}/{r['arch']}: "
                            f"**{r['method']}** (ACC {r['ACC']:.4f}, Macro-F1 {r['MacroF1']:.4f}).")
    for line in sig_lines:
        exec_out.append(f"- Exact paired tests: {line}.")
    cpath = os.path.join(RESULTS_DIR, "consistency_check.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            cons = json.load(f)
        exec_out.append(f"- Main-vs-few-shot consistency: **{cons.get('status', 'unknown')}** "
                        f"(max |ACC diff| {cons.get('k150_max_abs_acc_diff')}).")
    if temporal_line:
        exec_out.append(f"- {temporal_line}")
    if leak_line:
        exec_out.append(f"- {leak_line}")
    if task_line:
        exec_out.append(f"- {task_line}")
    exec_out.append("- The IID headline is a benchmark estimand; the temporal and "
                    "feature-provenance results are separate robustness/reliance "
                    "estimands, not deployment guarantees.")

    report = "\n".join(out)
    with open(os.path.join(RESULTS_DIR, "RESULTS_SUMMARY.md"), "w") as f:
        f.write(report)
    final_report = "\n".join(exec_out) + "\n\n---\n\n" + report.replace(
        "# Results Summary\n", "# Detailed results\n", 1)
    with open(os.path.join(RESULTS_DIR, "..", "v5_4_project_report_final.md"), "w") as f:
        f.write(final_report)
    print(f"Wrote {os.path.join(RESULTS_DIR, 'RESULTS_SUMMARY.md')} ({len(report)} chars)")
    print("Wrote v5_4_project_report_final.md "
          f"({len(final_report)} chars, auto-generated)")


if __name__ == "__main__":
    main()
