"""XAI study: SHAP vs IG, agreement and faithfulness across CIL methods.

This extends the base paper's XAI protocol (DeepSHAP + IG, Top5Ratio and
Avg.Rel.) in three ways:

  * SHAP is implemented from scratch as KernelSHAP (no ``shap`` dependency);
  * explanations are evaluated with deletion/insertion curves, not only with
    cross-method agreement;
  * agreement and faithfulness are reported across seeds and across three
    CIL predictors: BiC, iCaRL-lite + NCM, and the offline Scratch reference.

The same class-stratified test rows are used for every method on a given
dataset/architecture/seed, so cross-method comparisons are not confounded by
different test samples.  Explanations use a real training-medoid background
because the zero vector is degenerate for cosine/NCM paths, and every run is
repeated with a class-balanced mean-of-class-medoids background as a
sensitivity check.  IG defaults to 512 steps; KernelSHAP uses 256 sampled
coalitions.  KernelSHAP's efficiency constraint makes completeness exact by
construction, so completeness is an implementation sanity check rather than
evidence of attribution accuracy.  Correct-only rows are the main-text subset,
while all/error rows are retained for the supplement.

Writes:
  results/xai_metrics.csv
  results/xai_examples.csv
  results/xai_feature_importance.csv
  results/xai_summary.csv
  results/xai_baseline_sensitivity.csv
  results/xai_feature_summary.csv
  results/xai_metrics_by_correctness.csv
  results/xai_summary_by_correctness.csv
  results/xai_group_attribution_runs.csv
  results/xai_group_attribution_summary.csv
  results/xai_group_deletion_runs.csv
  results/xai_group_deletion_summary.csv
  results/xai_group_summary.csv
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

import data_utils as du
from train import train_continual
from inference import concatenate_test, eval_logits
from xai import (integrated_gradients, kernel_shap, normalize_attributions,
                 topk_overlap, avg_relative, spearman_rank,
                 faithfulness_metrics, completeness_errors,
                 group_attribution_shares, group_deletion_insertion,
                 select_stratified_indices)
from feature_groups import get_feature_groups, feature_group_indices
from config import (MODEL_CLASSES, model_kwargs_for, DATA_DIR, RESULTS_DIR,
                    BASE_TRAIN_HPARAMS, VAL_FRAC, VAL_CAP_PER_CLASS,
                    load_frozen_hparams_strict)

DEFAULT_DATASETS = ["CuCD-ID", "STIN"]
DEFAULT_ARCHS = ["MLP", "GRU"]
DEFAULT_METHODS = ["bic", "icarl_ncm", "joint_scratch"]
DEFAULT_SEEDS = [0, 1, 2]

PRIMARY_BASELINE = "training_medoid"
SECONDARY_BASELINE = "class_balanced_medoid_mean"

_FROZEN = None
_DATASETS = {}
_ARGS = None


def _training_baselines(ts):
    """Return the primary and sensitivity backgrounds for one task sequence.

    Primary: the real training row closest to the global training mean.
    Sensitivity: the mean of the per-class medoids, which is class-balanced
    and does not reduce to a single potentially atypical row.
    """
    X_train = np.concatenate([task.X_train for task in ts.tasks], axis=0)
    train_mean = X_train.mean(axis=0)
    primary = X_train[int(np.argmin(
        ((X_train - train_mean) ** 2).sum(axis=1)))]

    class_medoids = []
    for task in ts.tasks:
        for local_class in range(task.n_new_classes):
            Xc = task.X_train[task.y_train == local_class]
            if len(Xc) == 0:
                continue
            class_mean = Xc.mean(axis=0)
            class_medoids.append(Xc[int(np.argmin(
                ((Xc - class_mean) ** 2).sum(axis=1)))])
    secondary = (np.mean(np.stack(class_medoids, axis=0), axis=0)
                 if class_medoids else primary.copy())
    return {PRIMARY_BASELINE: primary, SECONDARY_BASELINE: secondary}


def _train_method(method):
    return "icarl" if method == "icarl_ncm" else method


def _build_tasks(datasets, seeds):
    cucd_df = du.load_cucd_id(
        os.path.join(DATA_DIR, "CuCD-ID", "noised_dataset.csv"),
        os.path.join(DATA_DIR, "CuCD-ID", "label_schema.json"))
    stin_df = du.load_stin(
        os.path.join(DATA_DIR, "STIN", "SAT20.csv"),
        os.path.join(DATA_DIR, "STIN", "TER20.csv"))
    out = {}
    for dset in datasets:
        for seed in seeds:
            if dset == "CuCD-ID":
                out[(dset, seed)] = du.build_cucd_id_tasks(
                    cucd_df, seed=seed, val_frac=VAL_FRAC,
                    val_cap_per_class=VAL_CAP_PER_CLASS)
            elif dset == "STIN":
                out[(dset, seed)] = du.build_stin_tasks(
                    stin_df, seed=seed, val_frac=VAL_FRAC,
                    val_cap_per_class=VAL_CAP_PER_CLASS)
            else:
                raise ValueError(dset)
    return out


def _predictor(out):
    """Return (score_fn, exact input-gradient fn) for the deployed predictor."""
    model = out["model"]
    ncm = out.get("ncm")
    corrector = out.get("corrector")
    if ncm is not None:
        return (lambda X: ncm.logits(model, X),
                lambda X, t: ncm.input_grad(model, X, t))

    if corrector is not None and getattr(corrector, "fitted", False):
        scales = np.ones(model.n_classes_total, dtype=np.float64)
        for stage in corrector.stages:
            lo = stage["n_old"]
            hi = lo + stage["n_new"]
            scales[lo:hi] *= float(stage["alpha"])

        def score_fn(X):
            return corrector.apply(model.forward(X)[0])

        def grad_fn(X, targets):
            logits, cache = model.forward(X)
            dlogits = np.zeros_like(logits)
            dlogits[np.arange(len(X)), targets] = scales[targets]
            return model.backward_input(cache, dlogits)
        return score_fn, grad_fn

    def score_fn(X):
        return model.forward(X)[0]

    def grad_fn(X, targets):
        logits, cache = model.forward(X)
        dlogits = np.zeros_like(logits)
        dlogits[np.arange(len(X)), targets] = 1.0
        return model.backward_input(cache, dlogits)
    return score_fn, grad_fn


METRIC_KEYS = [
    "top5_ratio", "avg_relative", "spearman",
    "ig_deletion_auc", "ig_insertion_auc", "ig_comprehensiveness",
    "ig_sufficiency", "shap_deletion_auc", "shap_insertion_auc",
    "shap_comprehensiveness", "shap_sufficiency", "base_prob",
    "ig_completeness_mean_abs", "ig_completeness_max_abs",
    "ig_completeness_mean_rel", "ig_completeness_frac_1pct",
    "ig_completeness_frac_5pct", "shap_completeness_mean_abs",
    "shap_completeness_max_abs", "shap_completeness_mean_rel",
    "shap_completeness_frac_1pct", "shap_completeness_frac_5pct",
]


def _subset_metrics(score_fn, Xs, ys, targets, ig, shap, mask, subset,
                    dset_name, arch, method, seed, min_n=1, baseline=None,
                    baseline_name=PRIMARY_BASELINE):
    idx = np.flatnonzero(mask)
    row = dict(dataset=dset_name, arch=arch, method=method, seed=seed,
               baseline=baseline_name, subset=subset,
               n_samples=int(len(idx)))
    if len(idx) < int(min_n):
        row.update({k: float("nan") for k in METRIC_KEYS})
        return row
    X, y, t = Xs[idx], ys[idx], targets[idx]
    ig_s, shap_s = ig[idx], shap[idx]
    ig_n = normalize_attributions(ig_s)
    shap_n = normalize_attributions(shap_s)
    faith_ig = faithfulness_metrics(score_fn, X, t, ig_s, baseline=baseline)
    faith_shap = faithfulness_metrics(score_fn, X, t, shap_s, baseline=baseline)
    comp_ig = completeness_errors(score_fn, X, t, ig_s, baseline=baseline)
    comp_shap = completeness_errors(score_fn, X, t, shap_s, baseline=baseline)
    row.update(dict(
        top5_ratio=topk_overlap(ig_n, shap_n, k=5),
        avg_relative=avg_relative(ig_n, shap_n),
        spearman=spearman_rank(ig_n, shap_n),
        ig_deletion_auc=faith_ig["deletion_auc"],
        ig_insertion_auc=faith_ig["insertion_auc"],
        ig_comprehensiveness=faith_ig["comprehensiveness"],
        ig_sufficiency=faith_ig["sufficiency"],
        shap_deletion_auc=faith_shap["deletion_auc"],
        shap_insertion_auc=faith_shap["insertion_auc"],
        shap_comprehensiveness=faith_shap["comprehensiveness"],
        shap_sufficiency=faith_shap["sufficiency"],
        base_prob=faith_ig["base_prob"],
        ig_completeness_mean_abs=comp_ig["mean_abs_error"],
        ig_completeness_max_abs=comp_ig["max_abs_error"],
        ig_completeness_mean_rel=comp_ig["mean_rel_error"],
        ig_completeness_frac_1pct=comp_ig["frac_within_1pct"],
        ig_completeness_frac_5pct=comp_ig["frac_within_5pct"],
        shap_completeness_mean_abs=comp_shap["mean_abs_error"],
        shap_completeness_max_abs=comp_shap["max_abs_error"],
        shap_completeness_mean_rel=comp_shap["mean_rel_error"],
        shap_completeness_frac_1pct=comp_shap["frac_within_1pct"],
        shap_completeness_frac_5pct=comp_shap["frac_within_5pct"],
    ))
    return row


def _run_one(item):
    dset_name, arch, method, seed = item
    ts = _DATASETS[(dset_name, seed)]
    model_kwargs = model_kwargs_for(arch, ts.n_features)
    train_method = _train_method(method)
    hp = {**BASE_TRAIN_HPARAMS,
          **_FROZEN.get(arch, {}).get(train_method, {}).get("hparams", {})}
    out = train_continual(
        MODEL_CLASSES[arch], model_kwargs, ts, train_method, hp, seed=seed,
        ncm_eval=(method == "icarl_ncm"),
        embedding_herding=(method == "icarl_ncm"))

    score_fn, grad_fn = _predictor(out)
    X, y = concatenate_test(ts)
    sel = select_stratified_indices(y, max_samples=_ARGS.max_samples,
                                    seed=seed)
    Xs, ys = X[sel], y[sel]
    logits = np.asarray(score_fn(Xs))
    targets = logits.argmax(axis=1)
    correct = targets == ys

    features = list(ts.feature_names)
    groups = get_feature_groups(dset_name, features)
    group_idx = feature_group_indices(features, groups)
    subsets = [
        ("all", np.ones(len(sel), dtype=bool), 1),
        ("correct", correct, 1),
        ("error", ~correct, 3),
    ]

    baselines = _training_baselines(ts)
    metric = None
    feature_rows = []
    example_rows = []
    correctness_rows = []
    group_attr_rows = []
    group_del_rows = []
    for baseline_name, baseline in baselines.items():
        ig = integrated_gradients(score_fn, Xs, targets, baseline=baseline,
                                  steps=_ARGS.ig_steps, grad_fn=grad_fn)
        shap = kernel_shap(score_fn, Xs, targets, baseline=baseline,
                           n_coalitions=_ARGS.coalitions, seed=seed)
        ig_n = normalize_attributions(ig)
        shap_n = normalize_attributions(shap)

        baseline_correctness = [
            _subset_metrics(score_fn, Xs, ys, targets, ig, shap, mask, subset,
                            dset_name, arch, method, seed, min_n=min_n,
                            baseline=baseline, baseline_name=baseline_name)
            for subset, mask, min_n in subsets
        ]
        for row in baseline_correctness:
            if row["subset"] == "all":
                row["n_correct"] = int(correct.sum())
                row["n_error"] = int((~correct).sum())
        correctness_rows.extend(baseline_correctness)
        if baseline_name == PRIMARY_BASELINE:
            metric = dict(baseline_correctness[0])

        for xai_method, attr in (("ig", ig_n), ("shap", shap_n)):
            mean_abs = np.abs(attr).mean(axis=0)
            mean_signed = attr.mean(axis=0)
            order = np.argsort(-mean_abs)
            for rank, j in enumerate(order, 1):
                feature_rows.append(dict(
                    dataset=dset_name, arch=arch, method=method, seed=seed,
                    baseline=baseline_name, xai_method=xai_method,
                    feature=features[j], rank=int(rank),
                    mean_abs_attribution=float(mean_abs[j]),
                    mean_signed_attribution=float(mean_signed[j]),
                ))

        for xai_method, attr in (("ig", ig_n), ("shap", shap_n)):
            abs_share, signed_share = group_attribution_shares(attr, group_idx)
            for gi, name in enumerate(groups):
                group_attr_rows.append(dict(
                    dataset=dset_name, arch=arch, method=method, seed=seed,
                    baseline=baseline_name, xai_method=xai_method, group=name,
                    n_features=len(groups[name]),
                    mean_abs_share=float(abs_share[:, gi].mean()),
                    mean_signed_share=float(signed_share[:, gi].mean()),
                ))

        for r in group_deletion_insertion(score_fn, Xs, targets, group_idx,
                                          baseline=baseline):
            group_del_rows.append(dict(
                dataset=dset_name, arch=arch, method=method, seed=seed,
                baseline=baseline_name, **r))

        for xai_method, attr in (("ig", ig_n), ("shap", shap_n)):
            top = np.argsort(-np.abs(attr), axis=1)[:, :5]
            for row_i, sample_i in enumerate(sel):
                for rank, j in enumerate(top[row_i], 1):
                    example_rows.append(dict(
                        dataset=dset_name, arch=arch, method=method, seed=seed,
                        baseline=baseline_name, sample_index=int(sample_i),
                        true_class=ts.class_names_global[int(ys[row_i])],
                        pred_class=ts.class_names_global[int(targets[row_i])],
                        correct=bool(correct[row_i]),
                        xai_method=xai_method, rank=int(rank),
                        feature=features[j],
                        attribution=float(attr[row_i, j]),
                        abs_attribution=float(abs(attr[row_i, j])),
                    ))
    return (metric, feature_rows, example_rows, correctness_rows,
            group_attr_rows, group_del_rows)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--max-samples", type=int, default=48)
    p.add_argument("--coalitions", type=int, default=256)
    p.add_argument("--ig-steps", type=int, default=512)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.datasets = ["CuCD-ID"]
        args.archs = ["MLP"]
        args.methods = ["icarl_ncm"]
        args.seeds = [0]
        args.max_samples = 16
        args.coalitions = 128
        args.ig_steps = 128
    return args


def main():
    global _FROZEN, _DATASETS, _ARGS
    _ARGS = _parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _FROZEN = load_frozen_hparams_strict()
    _DATASETS = _build_tasks(_ARGS.datasets, _ARGS.seeds)

    items = [(dset, arch, method, seed)
             for dset in _ARGS.datasets
             for arch in _ARGS.archs
             for method in _ARGS.methods
             for seed in _ARGS.seeds]
    jobs = get_jobs()
    print(f"XAI: {len(items)} runs; {describe(jobs)}; "
          f"{_ARGS.max_samples} samples/run, {_ARGS.coalitions} coalitions, "
          f"{_ARGS.ig_steps} IG steps")

    metrics, feature_rows, example_rows = [], [], []
    correctness_rows, group_attr_rows, group_del_rows = [], [], []
    t0 = time.perf_counter()
    for i, (m, fr, er, cr, gar, gdr) in enumerate(
            parallel_map(_run_one, items, jobs=jobs), 1):
        metrics.append(m)
        feature_rows.extend(fr)
        example_rows.extend(er)
        correctness_rows.extend(cr)
        group_attr_rows.extend(gar)
        group_del_rows.extend(gdr)
        if i % 3 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {m['dataset']} {m['arch']} "
                  f"{m['method']} seed={m['seed']} "
                  f"Top5={m['top5_ratio']:.3f} "
                  f"IG_del={m['ig_deletion_auc']:.3f} "
                  f"SHAP_del={m['shap_deletion_auc']:.3f} "
                  f"IG_comp={m['ig_completeness_mean_rel']:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)")

    cdf = pd.DataFrame(correctness_rows)
    mdf = cdf[cdf.subset == "all"].copy()
    fdf = pd.DataFrame(feature_rows)
    edf = pd.DataFrame(example_rows)
    gadf = pd.DataFrame(group_attr_rows)
    gddf = pd.DataFrame(group_del_rows)
    mdf.to_csv(os.path.join(RESULTS_DIR, "xai_metrics.csv"), index=False)
    cdf.to_csv(os.path.join(RESULTS_DIR,
                            "xai_metrics_by_correctness.csv"), index=False)
    fdf.to_csv(os.path.join(RESULTS_DIR, "xai_feature_importance.csv"), index=False)
    edf.to_csv(os.path.join(RESULTS_DIR, "xai_examples.csv"), index=False)
    gadf.to_csv(os.path.join(RESULTS_DIR,
                             "xai_group_attribution_runs.csv"), index=False)
    gddf.to_csv(os.path.join(RESULTS_DIR,
                             "xai_group_deletion_runs.csv"), index=False)

    def summarise(df, group_cols):
        agg = {"seed": "count", "n_samples": "mean"}
        for col in METRIC_KEYS:
            agg[col] = ["mean", "std"]
        out = df.groupby(group_cols, dropna=False).agg(agg)
        out.columns = [
            ("n_runs" if c[0] == "seed" else
             "n_samples_mean" if c[0] == "n_samples" and c[1] == "mean" else
             f"{c[0]}_{c[1]}") for c in out.columns
        ]
        return out.reset_index()

    summary = summarise(mdf, ["dataset", "arch", "method", "baseline"])
    summary.to_csv(os.path.join(RESULTS_DIR, "xai_summary.csv"), index=False)

    csummary = summarise(cdf, ["dataset", "arch", "method", "baseline", "subset"])
    csummary.to_csv(os.path.join(RESULTS_DIR,
                                 "xai_summary_by_correctness.csv"), index=False)

    fsum = fdf.groupby(
        ["dataset", "arch", "method", "baseline", "xai_method", "feature"]).agg(
        mean_abs_attribution_mean=("mean_abs_attribution", "mean"),
        mean_abs_attribution_std=("mean_abs_attribution", "std"),
        mean_signed_attribution_mean=("mean_signed_attribution", "mean"),
    ).reset_index()
    fsum.to_csv(os.path.join(RESULTS_DIR, "xai_feature_summary.csv"), index=False)

    ga_summary = gadf.groupby(
        ["dataset", "arch", "method", "baseline", "group", "n_features",
         "xai_method"]
    ).agg(
        mean_abs_share_mean=("mean_abs_share", "mean"),
        mean_abs_share_std=("mean_abs_share", "std"),
        mean_signed_share_mean=("mean_signed_share", "mean"),
    ).reset_index()
    ga_summary.to_csv(
        os.path.join(RESULTS_DIR, "xai_group_attribution_summary.csv"),
        index=False)

    gd_summary = gddf.groupby(
        ["dataset", "arch", "method", "baseline", "group", "n_features"]
    ).agg(
        deletion_drop_mean=("deletion_drop", "mean"),
        deletion_drop_std=("deletion_drop", "std"),
        insertion_gain_mean=("insertion_gain", "mean"),
        insertion_gain_std=("insertion_gain", "std"),
        insertion_gap_vs_full_mean=("insertion_gap_vs_full", "mean"),
        base_prob_mean=("base_prob", "mean"),
        baseline_prob_mean=("baseline_prob", "mean"),
        deleted_prob_mean=("deleted_prob", "mean"),
        inserted_prob_mean=("inserted_prob", "mean"),
    ).reset_index()
    gd_summary.to_csv(
        os.path.join(RESULTS_DIR, "xai_group_deletion_summary.csv"),
        index=False)

    share_pivot = ga_summary.pivot_table(
        index=["dataset", "arch", "method", "baseline", "group", "n_features"],
        columns="xai_method", values="mean_abs_share_mean").reset_index()
    share_pivot = share_pivot.rename(columns={
        "ig": "ig_share_mean", "shap": "shap_share_mean"})
    group_summary = gd_summary.merge(
        share_pivot,
        on=["dataset", "arch", "method", "baseline", "group", "n_features"],
        how="left")
    group_summary.to_csv(
        os.path.join(RESULTS_DIR, "xai_group_summary.csv"), index=False)

    sensitivity_keys = [
        "top5_ratio_mean", "avg_relative_mean", "spearman_mean",
        "ig_deletion_auc_mean", "ig_insertion_auc_mean",
        "shap_deletion_auc_mean", "shap_insertion_auc_mean",
        "ig_comprehensiveness_mean", "shap_comprehensiveness_mean",
        "ig_completeness_mean_rel_mean", "shap_completeness_mean_rel_mean",
        "base_prob_mean",
    ]
    primary = summary[summary.baseline == PRIMARY_BASELINE].set_index(
        ["dataset", "arch", "method"])
    secondary = summary[summary.baseline == SECONDARY_BASELINE].set_index(
        ["dataset", "arch", "method"])
    sensitivity = primary[sensitivity_keys].add_suffix("_primary").join(
        secondary[sensitivity_keys].add_suffix("_secondary"), how="inner")
    for key in sensitivity_keys:
        sensitivity[f"{key}_delta"] = (sensitivity[f"{key}_secondary"] -
                                       sensitivity[f"{key}_primary"])
    sensitivity = sensitivity.reset_index()
    sensitivity.to_csv(
        os.path.join(RESULTS_DIR, "xai_baseline_sensitivity.csv"), index=False)

    primary_summary = summary[summary.baseline == PRIMARY_BASELINE]
    primary_csummary = csummary[
        (csummary.baseline == PRIMARY_BASELINE) & (csummary.subset == "correct")]
    print("\nXAI summary (primary training-medoid baseline, mean over seeds):")
    print(primary_summary[["dataset", "arch", "method", "top5_ratio_mean",
                           "avg_relative_mean", "ig_deletion_auc_mean",
                           "shap_deletion_auc_mean",
                           "ig_completeness_mean_rel_mean"]].round(3).to_string(
                               index=False))
    print("\nXAI correct-only summary (primary baseline, mean over seeds):")
    print(primary_csummary[
        ["dataset", "arch", "method", "n_samples_mean", "top5_ratio_mean",
         "ig_deletion_auc_mean", "shap_deletion_auc_mean"]
    ].round(3).to_string(index=False))
    print("\nBaseline sensitivity (secondary minus primary; top5/AUC):")
    print(sensitivity[
        ["dataset", "arch", "method", "top5_ratio_mean_delta",
         "ig_deletion_auc_mean_delta", "shap_deletion_auc_mean_delta"]
    ].round(3).to_string(index=False))
    n_models = len(cdf[["dataset", "arch", "method", "seed"]].drop_duplicates())
    print(f"\nWrote results/xai_*.csv ({n_models} model runs, "
          f"{len(mdf)} baseline evaluations, "
          f"{time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
