"""Regression test for the exact sign-flip permutation p-value floor."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from run_stats import paired_permutation_test


def main():
    x = np.linspace(0.9, 0.99, 10)
    y = np.linspace(0.1, 0.19, 10)
    res = paired_permutation_test(x, y)
    assert res["p"] > 0.0, "exact p must never be zero"
    assert abs(res["p"] - 2.0 / 1024.0) < 1e-12, res
    print("test_stats_pvalue: OK", res["p"])


if __name__ == "__main__":
    main()
