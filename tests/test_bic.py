"""Focused regression test for the BiC stage-composition bug."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import BiasCorrector


class SpyCorrector(BiasCorrector):
    def __init__(self):
        super().__init__()
        self.apply_calls = 0

    def apply(self, logits):
        self.apply_calls += 1
        return super().apply(logits)


def main():
    c = SpyCorrector()
    c.stages = [{"n_old": 2, "n_new": 1, "alpha": 2.0, "beta": 0.0}]
    logits = np.array([[5.0, 1.0, 0.1, 0.0],
                       [0.1, 5.0, 1.0, 0.0],
                       [1.0, 0.1, 5.0, 0.0]])
    y = np.array([0, 1, 2])
    c.fit_stage(logits, y, n_old=3, n_new=1,
                alphas=[1.0, 2.0], betas=[0.0])
    assert c.apply_calls >= 1, "fit_stage must compose earlier stages before fitting"
    assert len(c.stages) == 2
    print("test_bic: OK")


if __name__ == "__main__":
    main()
