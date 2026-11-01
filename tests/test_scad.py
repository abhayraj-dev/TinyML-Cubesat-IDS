"""SCAD loader and scenario-disjoint task construction smoke checks."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data_utils as du


def main():
    data = du.load_scad(
        residual=True, residual_mode="mean", max_scenarios_per_class=12)
    assert len(data["feature_names"]) == 80
    assert set(data["classes"]) == {"normal", "RWA", "RWB", "RWC"}

    for cls, items in data["classes"].items():
        expected = 12 if cls == "normal" else 60
        assert len(items) == expected, (cls, len(items))
        assert all(X.shape[1] == 80 for _, X in items), cls
        assert all(len(X) > 0 for _, X in items), cls
        assert all(np.isfinite(X).all() for _, X in items), cls

    assert all(len(X) == 302 for _, X in data["classes"]["normal"])
    assert all(len(X) == 300 for _, X in data["classes"]["RWA"])

    ts = du.build_scad_tasks(data, seed=0)
    assert ts.dataset_name == "SCAD"
    assert ts.n_classes_total == 4
    assert ts.split_mode == "scenario_disjoint"
    assert [t.class_names for t in ts.tasks] == [
        ["normal", "RWA"], ["RWB"], ["RWC"]]
    assert all(t.X_train.shape[1] == 80 for t in ts.tasks)
    assert all(t.X_train.shape[0] == t.y_train.shape[0] for t in ts.tasks)
    assert all(t.X_test.shape[0] == t.y_test.shape[0] for t in ts.tasks)

    report = ts.scad_split_report
    for task_name, per_class in report.items():
        for cls, split in per_class.items():
            assert split["train_scenarios"] > 0, (task_name, cls)
            assert split["val_scenarios"] > 0, (task_name, cls)
            assert split["test_scenarios"] > 0, (task_name, cls)

    ts2 = du.build_scad_tasks(data, seed=0)
    for a, b in zip(ts.tasks, ts2.tasks):
        assert np.array_equal(a.X_train, b.X_train)
        assert np.array_equal(a.X_test, b.X_test)

    print("SCAD loader/task checks passed")


if __name__ == "__main__":
    main()
