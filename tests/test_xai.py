"""Regression tests for the from-scratch XAI utilities."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai import (integrated_gradients, kernel_shap, normalize_attributions,
                 topk_overlap, avg_relative, spearman_rank,
                 deletion_insertion_curves, faithfulness_metrics,
                 completeness_errors, group_attribution_shares,
                 group_deletion_insertion, select_stratified_indices,
                 target_probabilities)


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 6))
    w = np.array([0.5, -1.0, 2.0, 0.0, -0.25, 1.5])
    score_fn = lambda Z: np.asarray(Z) @ w[:, None]
    targets = np.zeros(len(X), dtype=np.int64)

    ig = integrated_gradients(score_fn, X, targets, steps=4, eps=1e-4)
    exact = X * w[None, :]
    assert np.allclose(ig, exact, atol=1e-6), (ig, exact)

    shap = kernel_shap(score_fn, X, targets, n_coalitions=512, seed=1)
    assert np.allclose(shap.sum(axis=1),
                       (X @ w) - 0.0, atol=1e-6), shap.sum(axis=1)
    assert np.allclose(shap, exact, atol=0.25), (shap, exact)

    A = normalize_attributions(ig)
    B = normalize_attributions(shap)
    assert np.allclose(np.abs(A).sum(axis=1), 1.0)
    assert topk_overlap(A, A) == 1.0
    assert 0.0 <= avg_relative(A, B) < 2.0
    assert -1.0 <= spearman_rank(A, B) <= 1.0

    curves = deletion_insertion_curves(score_fn, X, targets, ig,
                                       fractions=np.linspace(0, 1, 5))
    assert curves["deletion"].shape == (5,)
    assert curves["insertion"].shape == (5,)
    assert curves["deletion"][0] >= curves["deletion"][-1]
    assert curves["insertion"][0] <= curves["insertion"][-1]
    fm = faithfulness_metrics(score_fn, X, targets, ig)
    assert set(fm) >= {"deletion_auc", "insertion_auc",
                       "comprehensiveness", "sufficiency"}

    comp = completeness_errors(score_fn, X, targets, ig)
    assert comp["mean_rel_error"] < 1e-6, comp
    nl_fn = lambda Z: np.sin(Z).sum(axis=1, keepdims=True)
    nl_ig = integrated_gradients(nl_fn, X, targets, steps=512, eps=1e-4)
    comp_nl = completeness_errors(nl_fn, X, targets, nl_ig)
    assert comp_nl["mean_rel_error"] < 0.02, comp_nl
    assert comp_nl["frac_within_5pct"] == 1.0, comp_nl

    groups = {"g0": [0, 1], "g1": [2, 3, 4, 5]}
    shares, signed = group_attribution_shares(ig, groups)
    assert shares.shape == (5, 2)
    assert np.allclose(shares.sum(axis=1), 1.0)
    expected_signed = ig.sum(axis=1) / np.abs(ig).sum(axis=1)
    assert np.allclose(signed.sum(axis=1), expected_signed)
    gd = group_deletion_insertion(score_fn, X, targets, groups)
    assert len(gd) == 2
    assert {r["group"] for r in gd} == {"g0", "g1"}
    assert {r["group"]: r["n_features"] for r in gd} == {"g0": 2, "g1": 4}
    zero_baseline_prob = float(target_probabilities(
        score_fn, np.zeros_like(X), targets).mean())
    for r in gd:
        assert np.isclose(r["baseline_prob"], zero_baseline_prob)
        assert np.isclose(r["insertion_gain"],
                          r["inserted_prob"] - r["baseline_prob"])
        assert np.isclose(r["insertion_gap_vs_full"],
                          r["inserted_prob"] - r["base_prob"])

    baseline = np.full_like(X, 0.25)
    gd_b = group_deletion_insertion(score_fn, X, targets, groups,
                                    baseline=baseline)
    for r in gd_b:
        assert np.isclose(r["baseline_prob"],
                          float(target_probabilities(
                              score_fn, baseline, targets).mean()))
        assert np.isclose(r["insertion_gain"],
                          r["inserted_prob"] - r["baseline_prob"])
        assert np.isclose(r["insertion_gap_vs_full"],
                          r["inserted_prob"] - r["base_prob"])

    y = np.repeat(np.arange(4), 10)
    sel = select_stratified_indices(y, max_samples=16, seed=3)
    assert len(sel) == 16
    assert len(np.unique(y[sel])) == 4

    print("test_xai: OK")


if __name__ == "__main__":
    main()
