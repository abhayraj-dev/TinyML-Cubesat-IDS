"""Ensure task-order permutations do not change splits or the launch scaler."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data_utils as du


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = du.load_cucd_id(
        os.path.join(root, "data/CuCD-ID/noised_dataset.csv"),
        os.path.join(root, "data/CuCD-ID/label_schema.json"))
    a = du.build_cucd_id_tasks_order(df, [0, 1, 2, 3], seed=0, budget_mode="fixed_position")
    b = du.build_cucd_id_tasks_order(df, [3, 2, 1, 0], seed=0, budget_mode="fixed_position")
    assert np.allclose(a.scaler_mean, b.scaler_mean)
    assert np.allclose(a.scaler_std, b.scaler_std)
    tests_a, tests_b = {}, {}
    for ts, out in ((a, tests_a), (b, tests_b)):
        for task in ts.tasks:
            for c in range(task.n_new_classes):
                out[task.class_names[c]] = task.X_test[task.y_test == c]
    assert set(tests_a) == set(tests_b)
    for name in tests_a:
        assert np.array_equal(tests_a[name], tests_b[name]), name

    def by_class(ts, attr_x, attr_y):
        out = {}
        for task in ts.tasks:
            X = getattr(task, attr_x)
            y = getattr(task, attr_y)
            if X is None:
                continue
            for c in range(task.n_new_classes):
                out[task.class_names[c]] = X[y == c]
        return out

    for attr_x, attr_y in (("X_calib", "y_calib"), ("X_select", "y_select")):
        ca, cb = by_class(a, attr_x, attr_y), by_class(b, attr_x, attr_y)
        assert set(ca) == set(cb)
        for name in ca:
            assert np.array_equal(ca[name], cb[name]), (attr_x, name)

    print("test_task_order: OK")


if __name__ == "__main__":
    main()
