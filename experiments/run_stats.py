"""Phase 1 paired statistical comparisons (exact paired permutation test + Holm).

Reads results/main_results_runs.csv (ACC per seed) and writes
results/stats_permutation.csv: for each (dataset, arch) pair, headline method
comparisons on ACC across matched seeds (n=10): mean diff, 95% CI, Cohen's d,
exact permutation p-value, Holm-adjusted p-value.

The p-value is computed by EXACT enumeration over all 2^n sign-flip
permutations (n=10 -> 1024, trivial). No SciPy dependency, no Monte Carlo.
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
    full enumeration regardless since headline n=10 -> 1024 permutations).
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
    for mask in range(n_perms):
        s = 0.0
        for i in range(n):
            s += diff[i] if (mask >> i) & 1 else -diff[i]
        if abs(s / n) >= abs_obs:
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


def main():
    path = os.path.join(RESULTS_DIR, "main_results_runs.csv")
    if not os.path.exists(path):
        print(f"No {path} -- run run_main.py first.")
        return
    df = pd.read_csv(path)

    # Headline comparisons: best vs rest on CuCD-ID per arch
    headline_comparisons = [
        ("CuCD-ID", "MLP", "icarl", "replay"),
        ("CuCD-ID", "MLP", "icarl", "full"),
        ("CuCD-ID", "MLP", "icarl", "bic"),
        ("CuCD-ID", "MLP", "replay", "finetune"),
        ("CuCD-ID", "MLP", "full", "finetune"),
        ("CuCD-ID", "GRU", "icarl", "replay"),
        ("CuCD-ID", "GRU", "icarl", "full"),
        ("CuCD-ID", "CNN1D", "bic", "replay"),
        ("CuCD-ID", "CNN1D", "bic", "icarl"),
    ]

    rows = []
    all_p_values = []

    for dset, arch, method_a, method_b in headline_comparisons:
        g = df[(df.dataset == dset) & (df.arch == arch)]
        if len(g) == 0:
            continue
        ga = g[g.method == method_a].sort_values("seed")
        gb = g[g.method == method_b].sort_values("seed")
        seeds = sorted(set(ga.seed) & set(gb.seed))
        if len(seeds) < 2:
            continue
        xa = ga.set_index("seed").loc[seeds, "ACC"].to_numpy()
        xb = gb.set_index("seed").loc[seeds, "ACC"].to_numpy()

        res = paired_permutation_test(xa, xb)
        all_p_values.append(res["p"])

        rows.append(dict(
            dataset=dset, arch=arch, method_a=method_a, method_b=method_b,
            n=len(seeds),
            mean_a=float(xa.mean()), mean_b=float(xb.mean()),
            mean_diff=res["mean_diff"],
            ci_low=res["ci_low"], ci_high=res["ci_high"],
            cohens_d=res["d"],
            p_exact=res["p"],
        ))

    # Holm correction
    holm_p = holm_correction(all_p_values)
    for i, row in enumerate(rows):
        row["p_holm"] = holm_p[i]

    out = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(RESULTS_DIR, "stats_permutation.csv"), index=False)

    print(f"Wrote {len(out)} headline comparisons to results/stats_permutation.csv")
    print("\nHeadline results (CuCD-ID):")
    for _, row in out.iterrows():
        sig = "**" if row["p_holm"] < 0.05 else ("*" if row["p_holm"] < 0.1 else "")
        print(f"  {row['arch']:5s} {row['method_a']:8s} vs {row['method_b']:8s}: "
              f"Δ={row['mean_diff']:+.3f} 95%CI=[{row['ci_low']:+.3f},{row['ci_high']:+.3f}] "
              f"d={row['cohens_d']:+.2f} p={row['p_exact']:.4f} p_holm={row['p_holm']:.4f} {sig}")


if __name__ == "__main__":
    main()
