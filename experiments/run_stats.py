"""Paired statistical comparisons (exact paired permutation test + Holm).

Reads results/main_results_runs.csv and writes:
  results/stats_permutation.csv          — ACC all-pairwise comparisons
  results/stats_permutation_macrof1.csv  — Macro-F1 (primary metric) comparisons

The family is every unordered method pair within each dataset x architecture
(216 comparisons per metric), fixed before looking at results. Both files
cover CuCD-ID and STIN across matched seeds (n=15), with mean diff, 95% CI,
Cohen's d, exact permutation p-value, and Holm-adjusted p-value. Holm is
applied across the whole metric-specific family.

The p-value is computed by EXACT enumeration over all 2^n sign-flip
permutations (n=15 -> 32768). No SciPy dependency, no Monte Carlo.
"""
import sys
import os
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import RESULTS_DIR


def paired_permutation_test(x, y):
    """Exact paired permutation test (sign-flip test for paired differences).
    Enumerates all 2^n sign assignments (exact for n <= 20; falls back to
    full enumeration regardless since n=15 -> 32768 permutations).
    Returns p-value, 95% CI for mean difference, Cohen's d."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    diff = x - y
    mean_diff = float(diff.mean())

    if n == 0:
        return {"p": 1.0, "ci_low": 0.0, "ci_high": 0.0, "d": 0.0, "mean_diff": 0.0}

    # Cohen's d for paired samples
    d = mean_diff / (diff.std(ddof=1) + 1e-12)

    # Exact two-sided p: fraction of the 2^n sign assignments whose mean
    # absolute permuted difference meets/exceeds the observed one.
    n_perms = 2 ** n
    count = 0
    abs_obs = abs(mean_diff)
    # Exact enumeration is still floating point. Without a tolerance the
    # all-positive/all-negative assignments can miss the observed statistic by
    # one ulp, producing the impossible p=0. The minimum exact p is 2/2^n.
    tol = 1e-12 * max(1.0, abs_obs)
    for mask in range(n_perms):
        s = 0.0
        for i in range(n):
            s += diff[i] if (mask >> i) & 1 else -diff[i]
        if abs(s / n) >= abs_obs - tol:
            count += 1
    p = float(count) / float(n_perms)

    # 95% CI via percentile bootstrap
    boot_means = np.random.default_rng(123).choice(diff, size=(10000, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {"p": p, "ci_low": float(ci_low), "ci_high": float(ci_high),
            "d": float(d), "mean_diff": mean_diff}


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(m)
    for i, p in enumerate(sorted_p):
        adjusted[i] = min(1.0, p * (m - i))
    # Ensure monotonicity
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i-1])
    # Map back to original order
    result = np.zeros(m)
    result[sorted_idx] = adjusted
    return result


COMPARISON_FAMILY = "all_pairwise_within_dataset_arch"


def build_comparisons(df):
    """Every unordered method pair within each dataset x architecture.

    The confirmatory family is fixed by design, not selected after looking at
    which differences were largest.  Holm correction is applied across the
    whole metric-specific family.
    """
    comparisons = []
    for dset in ["CuCD-ID", "STIN"]:
        for arch in sorted(df[df.dataset == dset].arch.unique()):
            methods = [m for m in [
                "finetune", "replay", "ewc", "full", "lwf",
                "bic", "icarl", "joint", "joint_scratch",
            ] if m in set(df[(df.dataset == dset) & (df.arch == arch)].method)]
            for method_a, method_b in itertools.combinations(methods, 2):
                comparisons.append((dset, arch, method_a, method_b))
    return comparisons


def run_metric(df, metric, suffix):
    rows = []
    all_p_values = []
    comparisons = build_comparisons(df)

    for dset, arch, method_a, method_b in comparisons:
        g = df[(df.dataset == dset) & (df.arch == arch)]
        if len(g) == 0:
            continue
        ga = g[g.method == method_a].sort_values("seed")
        gb = g[g.method == method_b].sort_values("seed")
        seeds = sorted(set(ga.seed) & set(gb.seed))
        if len(seeds) < 2:
            continue
        xa = ga.set_index("seed").loc[seeds, metric].to_numpy()
        xb = gb.set_index("seed").loc[seeds, metric].to_numpy()

        res = paired_permutation_test(xa, xb)
        all_p_values.append(res["p"])

        rows.append(dict(
            comparison_family=COMPARISON_FAMILY,
            dataset=dset, arch=arch, method_a=method_a, method_b=method_b,
            metric=metric, n=len(seeds),
            mean_a=float(xa.mean()), mean_b=float(xb.mean()),
            mean_diff=res["mean_diff"],
            ci_low=res["ci_low"], ci_high=res["ci_high"],
            cohens_d=res["d"],
            p_exact=res["p"],
        ))

    holm_p = holm_correction(all_p_values)
    for i, row in enumerate(rows):
        row["p_holm"] = holm_p[i]

    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, f"stats_permutation{suffix}.csv")
    out.to_csv(out_path, index=False)

    n_sig = int((out["p_holm"] < 0.05).sum()) if len(out) else 0
    print(f"\nWrote {len(out)} pre-specified pairwise comparisons to {out_path}")
    print(f"  {n_sig}/{len(out)} survive Holm p<0.05")
    top = out.reindex(out["mean_diff"].abs().sort_values(ascending=False).index).head(12)
    for _, row in top.iterrows():
        sig = "**" if row["p_holm"] < 0.05 else ("*" if row["p_holm"] < 0.1 else "")
        print(f"  {row['dataset']:7s} {row['arch']:5s} {row['method_a']:8s} vs "
              f"{row['method_b']:8s}: Δ={row['mean_diff']:+.3f} "
              f"95%CI=[{row['ci_low']:+.3f},{row['ci_high']:+.3f}] "
              f"d={row['cohens_d']:+.2f} p={row['p_exact']:.4f} "
              f"p_holm={row['p_holm']:.4f} {sig}")
    return out


def main():
    path = os.path.join(RESULTS_DIR, "main_results_runs.csv")
    if not os.path.exists(path):
        print(f"No {path} -- run run_main.py first.")
        return
    df = pd.read_csv(path)

    run_metric(df, "ACC", "")
    if "MacroF1" in df.columns:
        run_metric(df, "MacroF1", "_macrof1")


if __name__ == "__main__":
    main()
