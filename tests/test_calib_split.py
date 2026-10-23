"""Calibration/selection halves are disjoint, stratified, and deterministic."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import Task


def main():
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    t1 = Task(task_id=0, name="a", class_names=["x", "y"],
              X_train=X[:2], y_train=y[:2], X_test=X[:2], y_test=y[:2],
              X_val=X, y_val=y)
    t2 = Task(task_id=3, name="b", class_names=["x", "y"],
              X_train=X[:2], y_train=y[:2], X_test=X[:2], y_test=y[:2],
              X_val=X, y_val=y)

    assert t1.has_calib and t1.has_select
    assert len(t1.X_calib) == len(t1.X_select) == 10
    for c in (0, 1):
        n_c = int((t1.y_calib == c).sum())
        n_s = int((t1.y_select == c).sum())
        assert (n_c, n_s) == (5, 5), (c, n_c, n_s)

    assert np.array_equal(t1.X_calib, t2.X_calib)
    assert np.array_equal(t1.X_select, t2.X_select)

    cal = set(t1.X_calib[:, 0].tolist())
    sel = set(t1.X_select[:, 0].tolist())
    assert not (cal & sel), "calibration and selection rows overlap"
    print("test_calib_split: OK")


if __name__ == "__main__":
    main()
