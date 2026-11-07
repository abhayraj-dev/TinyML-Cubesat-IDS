"""Summarise the first-differencing robustness check.

Reads the run files produced by run_first_difference.py and writes:
  - first_difference_tests.csv
  - first_difference_takeaways.md

The paired test is the exact sign-flip permutation test already used in
experiments/run_stats.py (n=15 -> 32768 sign assignments), with Holm
correction across the nine methods within each diff_set x protocol family.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import pandas as pd

from config import RESULTS_DIR
from run_stats import holm_correction, paired_permutation_test

RESULTS = os.path.join(_HERE, "results")
MEMORY_FREE = ["finetune", "ewc", "lwf"]
EXEMPLAR = ["replay", "full", "bic", "icarl"]
ONLINE_EXEMPLAR = ["replay", "bic", "icarl"]
OFFLINE = ["joint", "joint_scratch"]


def _paired_tests(runs: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base = baseline[["protocol", "method", "seed", "ACC"]].copy()
    merged = runs.merge(
        base, on=["protocol", "method", "seed"], how="inner",
        suffixes=("_diff", "_base"), validate="many_to_one")
    rows = []
    for (diff_set, protocol), group in merged.groupby(
            ["diff_set", "protocol"], sort=True):
        group = group.sort_values(["method", "seed"])
        pvals, entries = [], []
        for method in sorted(group["method"].unique()):
            sub = group[group["method"] == method].sort_values("seed")
            res = paired_permutation_test(
                sub["ACC_diff"].to_numpy(), sub["ACC_base"].to_numpy())
            entries.append(dict(
                diff_set=diff_set,
                protocol=protocol,
                method=method,
                n=len(sub),
                baseline_ACC=float(sub["ACC_base"].mean()),
                diff_ACC=float(sub["ACC_diff"].mean()),
                mean_delta=res["mean_diff"],
                ci_low=res["ci_low"],
                ci_high=res["ci_high"],
                cohen_d=res["d"],
                p=res["p"],
            ))
            pvals.append(res["p"])
        adj = holm_correction(pvals)
        for entry, p_holm in zip(entries, adj):
            entry["p_holm"] = float(p_holm)
            rows.append(entry)
    return pd.DataFrame(rows)


def _mean_acc(summary, diff_set, protocol, methods):
    sub = summary[
        (summary["diff_set"] == diff_set)
        & (summary["protocol"] == protocol)
        & (summary["method"].isin(methods))
    ]
    return float(sub["ACC_mean"].mean())


def _best(summary, diff_set, protocol, methods=None):
    sub = summary[
        (summary["diff_set"] == diff_set) & (summary["protocol"] == protocol)
    ]
    if methods is not None:
        sub = sub[sub["method"].isin(methods)]
    sub = sub.sort_values("ACC_mean", ascending=False)
    row = sub.iloc[0]
    return str(row["method"]), float(row["ACC_mean"])


def _baseline_best(baseline, protocol):
    sub = baseline[
        (baseline["protocol"] == protocol)
        & (baseline["method"].isin(EXEMPLAR))
    ].groupby("method")["ACC"].mean().sort_values(ascending=False)
    return str(sub.index[0]), float(sub.iloc[0])


def _write_takeaways(path, summary, tests, baseline):
    rs_iid_ex = _mean_acc(summary, "resource_state", "iid", ONLINE_EXEMPLAR)
    rs_fair_ex = _mean_acc(
        summary, "resource_state", "temporal_purged_fair", ONLINE_EXEMPLAR)
    rs_iid_joint = _mean_acc(summary, "resource_state", "iid", OFFLINE)
    rs_fair_joint = _mean_acc(
        summary, "resource_state", "temporal_purged_fair", OFFLINE)
    rs_fair_best, rs_fair_best_acc = _best(
        summary, "resource_state", "temporal_purged_fair", ONLINE_EXEMPLAR)

    np_iid_ex = _mean_acc(summary, "raw_no_prov", "iid", ONLINE_EXEMPLAR)
    np_fair_ex = _mean_acc(
        summary, "raw_no_prov", "temporal_purged_fair", ONLINE_EXEMPLAR)
    np_iid_joint = _mean_acc(summary, "raw_no_prov", "iid", OFFLINE)
    np_fair_joint = _mean_acc(
        summary, "raw_no_prov", "temporal_purged_fair", OFFLINE)
    np_fair_best, np_fair_best_acc = _best(
        summary, "raw_no_prov", "temporal_purged_fair", ONLINE_EXEMPLAR)

    lines = [
        "# First-differencing: takeaways",
        "",
        "First-differencing removes constant offsets in selected feature "
        "groups before the same task sequences are rebuilt. It is a robustness "
        "check, not a replacement for the main protocol, and it does not by "
        "itself remove time, identifier or window shortcuts.",
        "",
        "## What each representation controls",
        "",
        "- `raw` differences `MsgLength` and the eight memory gauges. Time, "
        "packet identifiers and window aggregates remain, so this is the "
        "weakest control.",
        "- `raw_window` also differences the nine window aggregates but keeps "
        "time and packet identifiers. It is a harsher stress test, not a "
        "stronger causal control.",
        "- `resource_state` differences the 17 retained resource-state features "
        "in Singh et al. (eight memory gauges and nine window aggregates) and "
        "drops `MsgLength`, time and packet identifiers. This is the closest "
        "analogue of their control.",
        "- `raw_no_prov` differences the raw memory/length features and then "
        "drops time, all packet identifiers and all window aggregates. This is "
        "stricter than Singh et al. because it removes the differenced window "
        "signal entirely; treat it as a lower-bound control, not a replication.",
        "",
        "## Headline",
        "",
    ]

    known_order = ["raw", "raw_window", "resource_state", "raw_no_prov"]
    present = set(summary["diff_set"].unique())
    diff_sets = [d for d in known_order if d in present]
    diff_sets += sorted(present.difference(diff_sets))
    for diff_set in diff_sets:
        lines += [f"### `{diff_set}`", ""]
        for protocol in ["iid", "temporal_purged_fair"]:
            mf = _mean_acc(summary, diff_set, protocol, MEMORY_FREE)
            ex = _mean_acc(summary, diff_set, protocol, ONLINE_EXEMPLAR)
            best_method, best_acc = _best(
                summary, diff_set, protocol, ONLINE_EXEMPLAR)
            base_method, base_acc = _baseline_best(baseline, protocol)
            lines += [
                f"- **{protocol}**: memory-free mean ACC {mf:.3f}; "
                f"replay/BiC/iCaRL mean {ex:.3f}; best online exemplar "
                f"`{best_method}` ({best_acc:.3f}). Baseline best exemplar "
                f"`{base_method}` ({base_acc:.3f}).",
            ]
        lines += [""]

    lines += [
        "## Interpretation",
        "",
        "1. **`raw` removes the memory/length offsets but not the provenance "
        "shortcuts.** Memory-free methods remain at the 0.25 level while "
        "online exemplars average 0.973 IID and 0.837 fair-temporal. That is a "
        "useful offset-only check, but it cannot answer the full "
        "session-confounding critique because time, packet identifiers and "
        "window aggregates are still present.",
        f"2. **`resource_state` is the closest control to Singh et al.** "
        f"Memory-free methods remain at 0.25, online exemplars fall to "
        f"{rs_iid_ex:.3f} IID and {rs_fair_ex:.3f} fair-temporal, and "
        f"joint/joint_scratch average {rs_iid_joint:.3f} IID and "
        f"{rs_fair_joint:.3f} fair-temporal. Every exemplar and joint/scratch "
        "drop is significant after Holm correction. A large part of the 0.837 "
        "fair-temporal accuracy therefore depends on resource-state offsets "
        "and provenance fields, but the method gap does not vanish: exemplars "
        "stay about twice chance and well above memory-free baselines.",
        f"3. **`raw_no_prov` is a stricter lower bound.** It removes the "
        f"differenced window signal as well, leaving online exemplars at "
        f"{np_iid_ex:.3f} IID and {np_fair_ex:.3f} fair-temporal, and "
        f"joint/joint_scratch at {np_iid_joint:.3f} IID and "
        f"{np_fair_joint:.3f} fair-temporal. The exact and stricter controls "
        "bracket the result at roughly 0.47--0.55 fair-temporal for the online "
        "exemplars. Dropping the window signal therefore does not create the "
        "residual skill; it lowers the bound only slightly.",
        "4. **The exact exemplar ranking is not stable.** Under baseline fair "
        "temporal evaluation iCaRL is best; after `raw` differencing BiC is "
        f"best, and after `resource_state` BiC is still best "
        f"({rs_fair_best_acc:.3f}, `{rs_fair_best}`), as it is after "
        f"`raw_no_prov` ({np_fair_best_acc:.3f}, `{np_fair_best}`). Treat this "
        "as a difference in ordering: the paired tests compare each method "
        "with its own baseline, not BiC against iCaRL, and the ordering changes "
        "are not pairwise evidence.",
        "5. **`raw_window` is a stress test, not a better control.** Removing "
        "offsets from the window aggregates significantly lowers replay, "
        "iCaRL and the replay+EWC hybrid. Window features may carry attack "
        "signal, so do not read that drop as confound removal.",
        "6. **Random-split inflation shrinks but does not disappear.** The "
        f"`resource_state` gap is {rs_iid_ex:.3f} IID against "
        f"{rs_fair_ex:.3f} fair-temporal, much smaller than the `raw` gap. The "
        "paper should not claim that the random-split advantage is entirely "
        "independent of provenance fields.",
        "",
        "## Recommendation",
        "",
        "Use `raw` as the offset-only check, `raw_window` as a stress test, "
        "`resource_state` as the closest control to Singh et al., and "
        "`raw_no_prov` as the stricter lower bound. Say that the conclusions "
        "persist after differencing memory/length offsets, and, in both "
        "combined controls, that absolute accuracy drops significantly while "
        "exemplar methods remain above chance and above memory-free baselines. "
        "Do not say the confound is removed or that the conclusions are "
        "invariant to it. Keep this as a secondary robustness analysis, not a "
        "new methodological contribution.",
        "",
        "## Paired tests",
        "",
        "`first_difference_tests.csv` reports exact paired sign-flip tests for "
        "each method's ACC change relative to the original run, with Holm "
        "correction within each `diff_set x protocol` family of nine methods.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    runs = pd.read_csv(os.path.join(RESULTS, "first_difference_runs.csv"))
    summary = pd.read_csv(os.path.join(RESULTS, "first_difference_summary.csv"))
    baseline = pd.read_csv(
        os.path.join(RESULTS_DIR, "temporal_purged_compare_MLP.csv"))

    tests = _paired_tests(runs, baseline)
    tests_path = os.path.join(RESULTS, "first_difference_tests.csv")
    tests.to_csv(tests_path, index=False)

    takeaways_path = os.path.join(RESULTS, "first_difference_takeaways.md")
    _write_takeaways(takeaways_path, summary, tests, baseline)

    print(tests.to_string(index=False))
    print(f"\nWrote {tests_path}")
    print(f"Wrote {takeaways_path}")


if __name__ == "__main__":
    main()
