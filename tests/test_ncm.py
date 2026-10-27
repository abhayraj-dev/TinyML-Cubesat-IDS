"""Regression tests for the NCM evaluation head."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ncm import NCMClassifier
from cmmu import ReplayBuffer


class IdentityTrunk:
    trunk_dim = 4
    NAME = "identity"

    def __init__(self):
        self.bias = np.zeros(4, dtype=np.float32)

    def forward_trunk(self, X):
        return np.asarray(X, dtype=np.float32) + self.bias, None


def main():
    X = np.array([
        [3.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 2.0, 1.0, 0.0],
    ], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    model = IdentityTrunk()

    ncm = NCMClassifier(n_classes=2, temperature=1.0)
    ncm.fit_from_arrays(model, X, y)
    assert ncm.fitted
    assert ncm.class_means.shape == (2, 4)
    # Means must be unit norm after L2 normalisation.
    norms = np.linalg.norm(ncm.class_means, axis=1)
    assert np.allclose(norms, 1.0), norms

    probes = np.array([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]],
                      dtype=np.float32)
    pred = ncm.predict(model, probes)
    assert pred.tolist() == [0, 1], pred

    buffer = ReplayBuffer(capacity=8, n_features=4)
    buffer.add_batch(X, y, 0)
    ncm2 = NCMClassifier(n_classes=2, temperature=10.0)
    ncm2.fit_from_buffer(model, buffer)
    assert ncm2.predict(model, probes).tolist() == [0, 1]
    assert ncm2.memory_bytes() > 0

    # A class with no exemplars must not crash and must receive a zero mean.
    ncm3 = NCMClassifier(n_classes=3)
    ncm3.fit_from_arrays(model, X, y, n_classes=3)
    assert np.allclose(ncm3.class_means[2], 0.0)
    assert ncm3.class_counts.tolist() == [2, 2, 0]

    # fit_from_buffer must re-embed with the *current* model, not reuse stale
    # means from the previous weight state.
    before = ncm2.class_means.copy()
    model.bias = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    ncm2.fit_from_buffer(model, buffer)
    assert not np.allclose(before, ncm2.class_means), "stale buffer means"

    # Memory accounting: FP16 class means + int32 counts.
    expected = 2 * 4 * 2 + 2 * 4
    assert ncm2.memory_bytes(bytes_per_val=2) == expected

    print("test_ncm: OK")


if __name__ == "__main__":
    main()
