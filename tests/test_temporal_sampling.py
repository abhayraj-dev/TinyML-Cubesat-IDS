"""Temporal train-sampling modes preserve the future test block and purge."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data_utils as du


def _synthetic_df(n_per_class=1000):
    labels = [
        "normal", "command_flooding", "data_injection",
        "defence_impairment", "storage_exhaustion",
    ]
    rows = []
    for i, label in enumerate(labels):
        t = np.linspace(0.1 + 0.2 * i, 1.0 + 0.2 * i, n_per_class)
        for j, tj in enumerate(t):
            rows.append(dict(
                TimeRadians=float(tj),
                MsgId=float(i * n_per_class + j),
                x=float(tj + 0.001 * (i + 1)),
                Label=i,
                label_name=label,
            ))
    return pd.DataFrame(rows)


def _split_times(ts):
    time_idx = ts.feature_names.index("TimeRadians")
    mean = ts.scaler_mean[time_idx]
    std = ts.scaler_std[time_idx]
    out = {}
    for task in ts.tasks:
        for c, name in enumerate(task.class_names):
            Xtr = task.X_train[task.y_train == c]
            Xte = task.X_test[task.y_test == c]
            Xva = (task.X_val[task.y_val == c]
                   if task.X_val is not None else np.zeros((0, 1)))
            out[name] = (
                Xtr[:, time_idx] * std + mean,
                Xva[:, time_idx] * std + mean if len(Xva) else Xva[:, 0],
                Xte[:, time_idx] * std + mean,
            )
    return out


def main():
    pool = np.arange(100)
    rng = np.random.default_rng(0)
    head = du._select_train_idx(pool, 10, "earliest", rng)
    assert np.array_equal(head, pool[:10])

    a = du._select_train_idx(pool, 10, "random_past", np.random.default_rng(0))
    b = du._select_train_idx(pool, 10, "random_past", np.random.default_rng(0))
    assert np.array_equal(a, b)
    assert len(a) == len(set(a.tolist())) == 10
    assert set(a.tolist()).issubset(set(pool.tolist()))
    assert not np.array_equal(np.sort(a), np.sort(head))

    try:
        du._select_train_idx(pool, 10, "unknown", rng)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown train_sample_mode should raise")

    df = _synthetic_df()
    kwargs = dict(seed=0, initial_n_per_class=50, uplink_n_per_class=20,
                  val_frac=0.2, purge_time_rad=0.01)
    earliest = du.build_cucd_id_tasks_purged_temporal(
        df, train_sample_mode="earliest", **kwargs)
    random_past = du.build_cucd_id_tasks_purged_temporal(
        df, train_sample_mode="random_past", **kwargs)
    random_past_2 = du.build_cucd_id_tasks_purged_temporal(
        df, train_sample_mode="random_past", **kwargs)

    assert earliest.train_sample_mode == "earliest"
    assert random_past.train_sample_mode == "random_past"
    for task, task_2 in zip(random_past.tasks, random_past_2.tasks):
        assert np.array_equal(task.X_train, task_2.X_train)

    early_times = _split_times(earliest)
    random_times = _split_times(random_past)
    assert set(early_times) == set(random_times)
    for name in early_times:
        early_train, _, early_test = early_times[name]
        random_train, _, random_test = random_times[name]
        assert early_train.max() < early_test.min()
        assert random_train.max() < random_test.min()
        assert random_train.max() > early_train.max()

    narrow = _synthetic_df()
    for i, label in enumerate(narrow.label_name.unique()):
        m = narrow.label_name == label
        narrow.loc[m, "TimeRadians"] = np.linspace(
            0.1 + 0.2 * i, 0.24 + 0.2 * i, int(m.sum()))
    fair_kwargs = dict(seed=0, initial_n_per_class=50, uplink_n_per_class=20,
                       val_frac=0.1, test_frac=0.2, purge_time_rad=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        block = du.build_cucd_id_tasks_purged_temporal(
            narrow, train_sample_mode="random_past", val_mode="block",
            **fair_kwargs)
    fair = du.build_cucd_id_tasks_purged_temporal(
        narrow, train_sample_mode="random_past", val_mode="random_past",
        **fair_kwargs)
    assert fair.val_sample_mode == "random_past"
    assert any(not t.has_val for t in block.tasks)
    assert all(t.has_val for t in fair.tasks)
    fair_times = _split_times(fair)
    for name, (train_t, val_t, test_t) in fair_times.items():
        assert len(train_t) and len(val_t) and len(test_t), name
        assert train_t.max() < test_t.min(), name
        assert val_t.max() < test_t.min(), name
        assert len(np.intersect1d(train_t, val_t)) == 0, name

    print("temporal sampling tests passed")


if __name__ == "__main__":
    main()
