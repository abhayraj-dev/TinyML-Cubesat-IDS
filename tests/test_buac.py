"""Unit checks for BUAC's gate primitives (no training data required)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import run_buac as rb


def test_circular_distance():
    d = rb._circular_median_nearest(np.array([0.1]), np.array([6.2]))
    assert abs(d - (2.0 * np.pi - 6.1)) < 1e-6


def test_auroc_and_threshold():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(rb._auroc(y, score) - 1.0) < 1e-9
    tau, sa, ua = rb._tune_threshold(y, score, max_unsafe=0.0)
    assert sa == 1.0 and ua == 0.0 and tau > 0.2 and tau <= 0.8


def test_logistic_learns_order():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    mu, sd = rb._standardize_fit(X)
    w = rb._fit_logistic(rb._standardize_apply(X, mu, sd), y)
    s = rb._logistic_score(rb._standardize_apply(X, mu, sd), w)
    assert s[0] < s[1] < s[2] < s[3]
    assert rb._auroc(y, s) > 0.99


def test_safe_label():
    R = np.array([
        [1.0, np.nan, np.nan],
        [0.9, 0.8, np.nan],
        [0.2, 0.3, 0.4],
    ])
    post, forget, safe = rb._safe_label(R, 1)
    assert abs(post - 0.85) < 1e-9 and abs(forget - 0.1) < 1e-9 and safe == 1
    post, forget, safe = rb._safe_label(R, 2)
    assert post < rb.SAFE_ACC and forget > rb.SAFE_FORGET and safe == 0


if __name__ == "__main__":
    test_circular_distance()
    test_auroc_and_threshold()
    test_logistic_learns_order()
    test_safe_label()
    print("test_buac: OK")
