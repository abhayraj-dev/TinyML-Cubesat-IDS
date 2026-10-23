"""
Dataset loading and class-incremental task split construction.

Two datasets are used:
  * CuCD-ID (primary)   -- `noised_dataset.csv`, CubeSat telemetry, 5 classes.
  * STIN    (secondary) -- `SAT20.csv` + `TER20.csv` concatenated, network-flow
                            features, 9 attack classes, used as a larger-scale
                            stress test and as the hyperparameter-tuning set.

Class-incremental protocol: every task introduces a set of classes that have
never appeared in any earlier task (classes are never repeated across
tasks). No task ID is available at inference time -- the model must
discriminate among ALL classes learned so far through a single shared
(growing) classifier. This is the harder, more realistic sibling of the
task-incremental protocol (where each task would keep its own private
output head and the task ID would be given at test time).

Each Task's y_train/y_test are stored as *local* indices, 0..n_new_classes-1,
relative only to the classes that task itself introduces (since a task's
classes are by construction disjoint from every other task's, this is
unambiguous). The mapping from local index to the model's actual global
head-column index is resolved dynamically, at training time, in
`src/train.py`, based on the order tasks are actually processed in -- this
is what allows the same TaskSequence to be safely replayed in a different
task order (see experiments/run_task_order.py's task-order-sensitivity study)
without corrupting label alignment.

Design rationale for the sample budgets: the operational concept behind
this project assumes a satellite is trained on a modest pre-launch corpus
for its first task, and thereafter receives only a *small, uplink-budget-
constrained* labelled CSV (order of a few hundred rows, i.e. kilobytes)
each time new attack types must be learned. We therefore draw a large
"initial" training pool for task 0 and a small "uplink" training pool for
every subsequent task, while evaluation (test) sets are held out
independently and are NOT subject to the uplink-size constraint (they play
the role of ground-truth telemetry archives kept on the ground for research
evaluation only -- they are never used for training).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import hashlib

import numpy as np
import pandas as pd


def permute_task_features(task: "Task", perm) -> None:
    """Apply the same feature permutation to every array attached to a task."""
    perm = np.asarray(perm)
    for name in ("X_train", "X_test", "X_val", "X_calib", "X_select"):
        arr = getattr(task, name, None)
        if arr is not None:
            setattr(task, name, arr[:, perm])


def _stable_seed(class_names) -> int:
    """Stable 32-bit seed from class identity (not task position).

    Task-order permutations reuse the same class-level split, so the seed must
    not depend on task_id/position.  Python's hash() is salted per process, so
    use SHA-256 instead.
    """
    key = "|".join(str(c) for c in class_names).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _split_calib_select(X_val: np.ndarray, y_val: np.ndarray, class_names,
                        calib_frac: float = 0.5):
    """Deterministically split validation rows into disjoint calibration and
    selection halves, stratified by class.

    The split must be identical for every method/architecture and for every
    task-order permutation; it must not consume the caller's RNG.  The seed is
    derived from the task's class identity, not its position.
    """
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    if X_val.size == 0:
        return None, None, None, None
    rng = np.random.default_rng(0xC0FFEE ^ _stable_seed(class_names))
    calib_idx, select_idx = [], []
    for c in np.unique(y_val):
        idx = np.flatnonzero(y_val == c)
        perm = rng.permutation(len(idx))
        idx = idx[perm]
        n = len(idx)
        if n == 1:
            # One row cannot be split; use it for calibration (the only
            # consumer that can fail loudly if empty) and leave selection empty.
            n_cal = 1
        else:
            n_cal = int(round(n * float(calib_frac)))
            n_cal = min(max(1, n_cal), n - 1)
        calib_idx.extend(idx[:n_cal].tolist())
        select_idx.extend(idx[n_cal:].tolist())
    calib_idx = np.asarray(calib_idx, dtype=np.int64)
    select_idx = np.asarray(select_idx, dtype=np.int64)
    X_calib = X_val[calib_idx] if len(calib_idx) else None
    y_calib = y_val[calib_idx] if len(calib_idx) else None
    X_select = X_val[select_idx] if len(select_idx) else None
    y_select = y_val[select_idx] if len(select_idx) else None
    return X_calib, y_calib, X_select, y_select


# --------------------------------------------------------------------------- #
# Task container
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    task_id: int
    name: str
    class_names: List[str]           # the classes newly introduced BY this task
    X_train: np.ndarray
    y_train: np.ndarray              # LOCAL ids, 0..n_new_classes-1 (offset resolved at train time)
    X_test: np.ndarray
    y_test: np.ndarray               # LOCAL ids, same convention as y_train
    # Phase 0: held-out validation slice. Consumed onboard by BiC calibration
    # and val-balanced Fisher estimation, and therefore counted inside the
    # reported uplink budget alongside the adaptation CSV (Option A: adaptation
    # + calibration). Optional so old Task(...) call sites keep working.
    X_val: np.ndarray = None
    y_val: np.ndarray = None         # LOCAL ids, same convention
    # Confirmatory split of X_val into two disjoint halves.  Calibration is
    # used to fit BiC / EWC Fisher; selection is used only for hyperparameter
    # scoring.  Builders may populate them explicitly; otherwise
    # __post_init__ performs a deterministic stratified split.
    X_calib: np.ndarray = None
    y_calib: np.ndarray = None
    X_select: np.ndarray = None
    y_select: np.ndarray = None
    calib_frac: float = 0.5

    def __post_init__(self):
        if self.X_val is None or len(self.X_val) == 0:
            return
        if self.X_calib is not None and self.X_select is not None:
            return
        self.X_calib, self.y_calib, self.X_select, self.y_select = \
            _split_calib_select(self.X_val, self.y_val, self.class_names,
                                self.calib_frac)

    @property
    def has_val(self) -> bool:
        return self.X_val is not None and len(self.X_val) > 0

    @property
    def has_calib(self) -> bool:
        return self.X_calib is not None and len(self.X_calib) > 0

    @property
    def has_select(self) -> bool:
        return self.X_select is not None and len(self.X_select) > 0

    @property
    def n_new_classes(self) -> int:
        return len(self.class_names)

    @property
    def uplink_bytes_fp32(self) -> int:
        """Raw float32 size of the labelled CSV that would need to be uplinked."""
        return int(self.X_train.nbytes + self.y_train.nbytes)

    @property
    def uplink_bytes_int8(self) -> int:
        """Quantised (int8 feature + int8 label) uplink payload size."""
        return int(self.X_train.shape[0] * self.X_train.shape[1] * 1 + self.X_train.shape[0] * 1)

    @property
    def uplink_calib_bytes_fp32(self) -> int:
        """Option-A calibration payload: held-out validation samples consumed
        onboard by BiC calibration and val-balanced Fisher estimation. These
        must be uplinked alongside the adaptation CSV for the onboard update
        to reproduce the reported results."""
        if self.X_calib is None or len(self.X_calib) == 0:
            return 0
        return int(self.X_calib.nbytes + self.y_calib.nbytes)

    @property
    def uplink_calib_bytes_int8(self) -> int:
        """Packed INT8 calibration payload (calibration features + labels)."""
        if self.X_calib is None or len(self.X_calib) == 0:
            return 0
        return int(self.X_calib.shape[0] * self.X_calib.shape[1] * 1
                   + self.X_calib.shape[0] * 1)


@dataclass
class TaskSequence:
    dataset_name: str
    tasks: List[Task]
    feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    class_names_global: List[str] = None  # descriptive only: canonical introduction order
    # Phase 0 provenance: "random" (legacy row-shuffle, leaks sliding-window
    # correlation) vs "temporal" (time-ordered blocks, leakage-safe).
    split_mode: str = "random"
    val_frac: float = 0.0

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_classes_total(self) -> int:
        return sum(t.n_new_classes for t in self.tasks)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_cucd_id(csv_path: str, schema_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    with open(schema_path) as f:
        schema = json.load(f)
    mapping = {int(k): v for k, v in schema["mapping"].items()}
    df["label_name"] = df["Label"].astype(int).map(mapping)
    return df


def load_stin(sat_path: str, ter_path: str) -> pd.DataFrame:
    a = pd.read_csv(sat_path)
    b = pd.read_csv(ter_path)
    a.columns = [c.strip() for c in a.columns]
    b.columns = [c.strip() for c in b.columns]
    assert list(a.columns) == list(b.columns), "SAT20/TER20 column mismatch"
    df = pd.concat([a, b], axis=0, ignore_index=True)
    df["label_name"] = df["Label"].astype(str)
    return df


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _stratified_train_test_split(idx: np.ndarray, test_frac: float, rng: np.random.Generator):
    idx = idx.copy()
    rng.shuffle(idx)
    n_test = max(1, int(round(len(idx) * test_frac)))
    return idx[n_test:], idx[:n_test]  # train_idx, test_idx


def _stratified_train_val_test_split(idx: np.ndarray, test_frac: float,
                                     val_frac: float, rng: np.random.Generator):
    """Phase 0: carve test FIRST (locked), then val from the remaining train
    pool. val_frac is a fraction of the *per-class pool* (not of train_pool),
    so test stays identical whether val_frac is 0 or not up to rounding.
    Returns (train_pool_idx, val_pool_idx, test_idx)."""
    idx = idx.copy()
    rng.shuffle(idx)
    n_test = max(1, int(round(len(idx) * test_frac)))
    test_idx = idx[:n_test]
    rest = idx[n_test:]
    if val_frac <= 0:
        return rest, np.zeros(0, dtype=idx.dtype), test_idx
    n_val = max(1, int(round(len(idx) * val_frac)))
    n_val = min(n_val, len(rest))
    val_idx = rest[:n_val]
    train_pool = rest[n_val:]
    return train_pool, val_idx, test_idx


def _temporal_block_split(idx_sorted: np.ndarray, test_frac: float, val_frac: float = 0.0):
    """Phase 0: leakage-safe contiguous split on a time-ordered index array.
    Layout is [train_pool | val | test] so test is always the *future* block
    (mimics deployment on unseen telemetry). No shuffling here."""
    n = len(idx_sorted)
    n_test = max(1, int(round(n * test_frac)))
    n_val = 0 if val_frac <= 0 else min(n - n_test - 1, max(1, int(round(n * val_frac))))
    test_idx = idx_sorted[n - n_test:]
    val_idx = idx_sorted[n - n_test - n_val:n - n_test] if n_val else np.zeros(0, dtype=idx_sorted.dtype)
    train_pool = idx_sorted[:n - n_test - n_val]
    return train_pool, val_idx, test_idx


def _chronological_order(sub: pd.DataFrame, time_col: str = "TimeRadians"):
    """Return `sub` ordered by unwrapped circular time.

    TimeRadians is periodic on [0, 2π); a plain sort puts samples after an
    orbital wrap at the beginning of the timeline. `np.unwrap` restores
    chronological order across wraps while preserving the original rows.
    """
    t = sub[time_col].to_numpy(dtype=np.float64)
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]

    # Find the largest gap on the circle. If it is an internal gap, rotate
    # there and unwrap across the orbital wrap. If the largest gap is the
    # wrap itself, ascending phase is already chronological.
    gaps = np.diff(t_sorted)
    wrap_gap = t_sorted[0] + 2.0 * np.pi - t_sorted[-1]
    if len(gaps) and wrap_gap <= gaps.max():
        cut = int(np.argmax(gaps)) + 1
        order = np.concatenate([order[cut:], order[:cut]])
        t_sorted = t[order]
    t_unwrapped = np.unwrap(t_sorted)
    return sub.iloc[order], t_unwrapped


def _temporal_purged_split(
    df: pd.DataFrame,
    mask: np.ndarray,
    test_frac: float,
    val_frac: float,
    purge_time_rad: float,
    time_col: str = "TimeRadians",
) -> tuple:
    """Leakage-safe contiguous split with time-based purge gaps on TimeRadians.
    Layout: [train_pool | purge(purge_time_rad) | val | purge(purge_time_rad) | test]
    The purge gaps are empty time regions to prevent sliding-window feature leakage.
    purge_time_rad: radians on TimeRadians (0.02 rad ≈ 22s equivalent,
    exceeds 20s sliding-window horizon + 20-22s clearing pauses).
    Sorting uses unwrapped circular time so orbital wraps do not scramble the
    chronological order. Returns (train_idx, val_idx, test_idx) as original
    dataframe indices."""
    sub, time_sorted = _chronological_order(df.loc[mask], time_col=time_col)
    idx_sorted = sub.index.to_numpy()

    n = len(idx_sorted)
    n_test = max(1, int(round(n * test_frac)))
    n_val = 0 if val_frac <= 0 else min(n - n_test - 1, max(1, int(round(n * val_frac))))

    # Test is the future block
    test_idx = idx_sorted[n - n_test:]
    test_start_time = time_sorted[n - n_test]

    # Val block ends purge_time_rad before test starts
    if n_val > 0:
        val_end_time = test_start_time - purge_time_rad
        val_time_slice = time_sorted[n - n_test - n_val:n - n_test]
        val_idx_slice = idx_sorted[n - n_test - n_val:n - n_test]
        val_mask = val_time_slice <= val_end_time
        val_idx = val_idx_slice[val_mask]
    else:
        val_idx = np.zeros(0, dtype=idx_sorted.dtype)

    # Train pool ends purge_time_rad before val starts (or test if no val)
    if len(val_idx) > 0:
        val_time_slice = time_sorted[n - n_test - n_val:n - n_test]
        val_mask = val_time_slice <= (test_start_time - purge_time_rad)
        val_start_time = val_time_slice[val_mask][0] if val_mask.any() else val_time_slice[0]
        train_end_time = val_start_time - purge_time_rad
    else:
        train_end_time = test_start_time - purge_time_rad

    train_time_slice = time_sorted[:n - n_test - n_val]
    train_idx_slice = idx_sorted[:n - n_test - n_val]
    train_mask = train_time_slice <= train_end_time
    train_pool = train_idx_slice[train_mask]

    return train_pool, val_idx, test_idx


def _draw(idx: np.ndarray, n: int, rng: np.random.Generator):
    if len(idx) <= n:
        return idx
    return rng.choice(idx, size=n, replace=False)


def _fit_scaler(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def _apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)


def _assert_split_integrity(tasks):
    """Structural integrity of a class-incremental task sequence.

    Checks class-once introduction, X/y shape alignment, and non-empty
    train/test per task. NOTE: this does NOT prove row-level
    train/val/test disjointness (scaled arrays lose index provenance);
    disjointness is guaranteed by construction in the split helpers
    (test locked first, val carved from the remainder, purge gaps applied
    on the time column before budget draws).
    """
    all_class_names = []
    for task in tasks:
        all_class_names.extend(task.class_names)
    # Verify each class appears exactly once across all tasks
    assert len(all_class_names) == len(set(all_class_names)), \
        f"Duplicate classes in task sequence: {all_class_names}"
    # Verify task structure integrity
    for i, task in enumerate(tasks):
        assert task.n_new_classes == len(task.class_names), \
            f"Task {i}: n_new_classes mismatch"
        assert task.X_train.shape[0] > 0, \
            f"Task {i}: empty X_train (purge or budget emptied it)"
        assert task.X_test.shape[0] > 0, \
            f"Task {i}: empty X_test"
        assert task.X_train.shape[0] == task.y_train.shape[0], \
            f"Task {i}: X_train/y_train shape mismatch"
        assert task.X_test.shape[0] == task.y_test.shape[0], \
            f"Task {i}: X_test/y_test shape mismatch"
        if task.has_val:
            assert task.X_val.shape[0] == task.y_val.shape[0], \
                f"Task {i}: X_val/y_val shape mismatch"
        if task.has_calib:
            assert task.X_calib.shape[0] == task.y_calib.shape[0], \
                f"Task {i}: X_calib/y_calib shape mismatch"
        if task.has_select:
            assert task.X_select.shape[0] == task.y_select.shape[0], \
                f"Task {i}: X_select/y_select shape mismatch"


# --------------------------------------------------------------------------- #
# CuCD-ID task construction  (class-incremental: each task = brand-new classes)
# --------------------------------------------------------------------------- #
CUCD_TASK_DEFS = [
    ("T0_launch", ["normal", "command_flooding"]),
    ("T1_data_injection", ["data_injection"]),
    ("T2_defence_impairment", ["defence_impairment"]),
    ("T3_storage_exhaustion", ["storage_exhaustion"]),
]


def build_cucd_id_tasks(
    df: pd.DataFrame,
    seed: int = 0,
    test_frac: float = 0.30,
    initial_n_per_class: int = 700,
    uplink_n_per_class: int = 150,
    val_frac: float = 0.20,
    val_cap_per_class: int = 300,
    split_mode: str = "random",
) -> TaskSequence:
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in df.columns if c not in ("Label", "label_name")]

    tasks = []
    class_names_global = []
    scaler_mean = scaler_std = None

    for t, (tname, class_list) in enumerate(CUCD_TASK_DEFS):
        n_per_class = initial_n_per_class if t == 0 else uplink_n_per_class
        Xs_train, ys_train, Xs_test, ys_test = [], [], [], []
        Xs_val, ys_val = [], []
        for local_c, lbl in enumerate(class_list):
            class_names_global.append(lbl)
            idx = df.index[df["label_name"] == lbl].to_numpy().copy()
            train_pool, val_pool, test_idx = _stratified_train_val_test_split(
                idx, test_frac, val_frac, rng)
            train_idx = _draw(train_pool, n_per_class, rng)
            val_idx = _draw(val_pool, min(len(val_pool), val_cap_per_class), rng)

            Xs_train.append(df.loc[train_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_train.append(np.full(len(train_idx), local_c, dtype=np.int64))
            Xs_test.append(df.loc[test_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_test.append(np.full(len(test_idx), local_c, dtype=np.int64))
            if len(val_idx):
                Xs_val.append(df.loc[val_idx, feature_cols].to_numpy(dtype=np.float32))
                ys_val.append(np.full(len(val_idx), local_c, dtype=np.int64))

        X_train = np.concatenate(Xs_train, axis=0)
        y_train = np.concatenate(ys_train, axis=0)
        X_test = np.concatenate(Xs_test, axis=0)
        y_test = np.concatenate(ys_test, axis=0)
        X_val = np.concatenate(Xs_val, axis=0) if Xs_val else None
        y_val = np.concatenate(ys_val, axis=0) if ys_val else None

        if t == 0:
            scaler_mean, scaler_std = _fit_scaler(X_train)

        X_train = _apply_scaler(X_train, scaler_mean, scaler_std)
        X_test = _apply_scaler(X_test, scaler_mean, scaler_std)
        if X_val is not None:
            X_val = _apply_scaler(X_val, scaler_mean, scaler_std)

        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        tasks.append(Task(
            task_id=t, name=tname, class_names=class_list,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            X_val=X_val, y_val=y_val,
        ))

    _assert_split_integrity(tasks)

    return TaskSequence("CuCD-ID", tasks, feature_cols, scaler_mean, scaler_std,
                        class_names_global, split_mode=split_mode, val_frac=val_frac)


def build_cucd_id_tasks_order(
    df: pd.DataFrame,
    order: list,
    seed: int = 0,
    test_frac: float = 0.30,
    initial_n_per_class: int = 700,
    uplink_n_per_class: int = 150,
    val_frac: float = 0.20,
    val_cap_per_class: int = 300,
    budget_mode: str = "fixed_position",
) -> TaskSequence:
    """Disentangled task-order builder for CuCD-ID.

    order: permutation of [0,1,2,3] giving which canonical task's classes are
    introduced at each position. budget_mode:
      - "fixed_position": position 0 always gets initial_n_per_class, later
        positions get uplink_n_per_class.
      - "moves_with_task": each task keeps its original budget wherever it is
        processed.

    To isolate processing order, every order reuses the *same* per-class
    train/val/test indices and the *same* pre-launch scaler. The scaler is fit
    once on the canonical launch corpus (normal + command_flooding), not on
    whichever task happens to occupy position 0. Each class has a common
    700-row train draw; later positions take a prefix of that same draw, so a
    class's 150-row uplink subset is nested inside its 700-row pool across
    orders. This removes the previous order-dependent split/scaler confound.
    """
    order = [int(x) for x in order]
    assert sorted(order) == [0, 1, 2, 3], order
    assert budget_mode in ("fixed_position", "moves_with_task")
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in df.columns if c not in ("Label", "label_name")]
    base_n = max(int(initial_n_per_class), int(uplink_n_per_class))

    # One canonical draw per class, independent of processing order.
    base = {}
    for _, class_list in CUCD_TASK_DEFS:
        for lbl in class_list:
            idx = df.index[df["label_name"] == lbl].to_numpy().copy()
            train_pool, val_pool, test_idx = _stratified_train_val_test_split(
                idx, test_frac, val_frac, rng)
            base[lbl] = {
                "train": _draw(train_pool, base_n, rng),
                "val": _draw(val_pool, min(len(val_pool), val_cap_per_class), rng),
                "test": test_idx,
            }

    # Fixed pre-launch normalizer: canonical task-0 training rows only.
    launch_class_list = CUCD_TASK_DEFS[0][1]
    launch_X = np.concatenate([
        df.loc[base[lbl]["train"][:initial_n_per_class], feature_cols].to_numpy(dtype=np.float32)
        for lbl in launch_class_list
    ], axis=0)
    scaler_mean, scaler_std = _fit_scaler(launch_X)

    tasks, class_names_global = [], []
    for pos, t_orig in enumerate(order):
        tname, class_list = CUCD_TASK_DEFS[t_orig]
        if budget_mode == "fixed_position":
            n_per_class = initial_n_per_class if pos == 0 else uplink_n_per_class
        else:
            n_per_class = initial_n_per_class if t_orig == 0 else uplink_n_per_class
        Xs_train, ys_train, Xs_test, ys_test, Xs_val, ys_val = [], [], [], [], [], []
        for local_c, lbl in enumerate(class_list):
            class_names_global.append(lbl)
            train_idx = base[lbl]["train"][:n_per_class]
            val_idx = base[lbl]["val"]
            test_idx = base[lbl]["test"]
            Xs_train.append(df.loc[train_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_train.append(np.full(len(train_idx), local_c, dtype=np.int64))
            Xs_test.append(df.loc[test_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_test.append(np.full(len(test_idx), local_c, dtype=np.int64))
            if len(val_idx):
                Xs_val.append(df.loc[val_idx, feature_cols].to_numpy(dtype=np.float32))
                ys_val.append(np.full(len(val_idx), local_c, dtype=np.int64))
        X_train = np.concatenate(Xs_train, axis=0)
        y_train = np.concatenate(ys_train, axis=0)
        X_test = np.concatenate(Xs_test, axis=0)
        y_test = np.concatenate(ys_test, axis=0)
        X_val = np.concatenate(Xs_val, axis=0) if Xs_val else None
        y_val = np.concatenate(ys_val, axis=0) if ys_val else None
        X_train = _apply_scaler(X_train, scaler_mean, scaler_std)
        X_test = _apply_scaler(X_test, scaler_mean, scaler_std)
        if X_val is not None:
            X_val = _apply_scaler(X_val, scaler_mean, scaler_std)
        tasks.append(Task(task_id=pos, name=f"pos{pos}_{tname}", class_names=class_list,
                          X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          X_val=X_val, y_val=y_val))
    _assert_split_integrity(tasks)
    order_tag = "-".join(str(x) for x in order)
    return TaskSequence("CuCD-ID", tasks, feature_cols, scaler_mean, scaler_std,
                        class_names_global, split_mode=f"random-order{order_tag}-{budget_mode}",
                        val_frac=val_frac)


# --------------------------------------------------------------------------- #
# STIN task construction (already class-disjoint per task by design)
# --------------------------------------------------------------------------- #
STIN_TASK_DEFS = [
    ("T0_flood_base", ["Syn_DDoS", "UDP_DDoS", "LDAP_DDoS"]),
    ("T1_flood_ext", ["MSSQL_DDoS", "Botnet", "Portmap_DDoS"]),
    ("T2_app_layer", ["Web Attack", "Backdoor", "NetBIOS_DDoS"]),
]


def build_stin_tasks(
    df: pd.DataFrame,
    seed: int = 0,
    test_frac: float = 0.20,
    initial_n_per_class: int = 700,
    uplink_n_per_class: int = 150,
    max_test_per_class: int = 1200,
    val_frac: float = 0.20,
    val_cap_per_class: int = 300,
    split_mode: str = "random",
) -> TaskSequence:
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in df.columns if c not in ("id", "Label", "label_name")]

    tasks = []
    class_names_global = []
    scaler_mean = scaler_std = None

    for t, (tname, class_labels) in enumerate(STIN_TASK_DEFS):
        Xs_train, ys_train, Xs_test, ys_test = [], [], [], []
        Xs_val, ys_val = [], []
        n_per_class = initial_n_per_class if t == 0 else uplink_n_per_class
        for local_c, lbl in enumerate(class_labels):
            class_names_global.append(lbl)
            idx = df.index[df["label_name"] == lbl].to_numpy()
            train_pool, val_pool, test_idx = _stratified_train_val_test_split(
                idx, test_frac, val_frac, rng)
            test_idx = _draw(test_idx, max_test_per_class, rng)
            train_idx = _draw(train_pool, n_per_class, rng)
            val_idx = _draw(val_pool, min(len(val_pool), val_cap_per_class), rng)

            Xs_train.append(df.loc[train_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_train.append(np.full(len(train_idx), local_c, dtype=np.int64))
            Xs_test.append(df.loc[test_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_test.append(np.full(len(test_idx), local_c, dtype=np.int64))
            if len(val_idx):
                Xs_val.append(df.loc[val_idx, feature_cols].to_numpy(dtype=np.float32))
                ys_val.append(np.full(len(val_idx), local_c, dtype=np.int64))

        X_train = np.concatenate(Xs_train, axis=0)
        y_train = np.concatenate(ys_train, axis=0)
        X_test = np.concatenate(Xs_test, axis=0)
        y_test = np.concatenate(ys_test, axis=0)
        X_val = np.concatenate(Xs_val, axis=0) if Xs_val else None
        y_val = np.concatenate(ys_val, axis=0) if ys_val else None

        if t == 0:
            scaler_mean, scaler_std = _fit_scaler(X_train)

        X_train = _apply_scaler(X_train, scaler_mean, scaler_std)
        X_test = _apply_scaler(X_test, scaler_mean, scaler_std)
        if X_val is not None:
            X_val = _apply_scaler(X_val, scaler_mean, scaler_std)

        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        tasks.append(Task(
            task_id=t, name=tname, class_names=class_labels,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            X_val=X_val, y_val=y_val,
        ))

    _assert_split_integrity(tasks)

    return TaskSequence("STIN", tasks, feature_cols, scaler_mean, scaler_std,
                        class_names_global, split_mode=split_mode, val_frac=val_frac)


# --------------------------------------------------------------------------- #
# Phase 0: temporal (leakage-safe) builders.
# Why: CuCD-ID features contain SlidingWindow*/InWindow aggregates over
# neighbouring rows, so a random row-shuffle leaks correlated windows across
# train/val/test and inflates absolute ACC. These builders sort each class by
# time (CuCD-ID: TimeRadians; STIN: file/row order as proxy) and cut
# contiguous [train | val | test-future] blocks. Budgets/scaler/test-frac
# semantics are otherwise identical to the random builders, so a
# random-vs-temporal comparison isolates leakage.
# --------------------------------------------------------------------------- #
def _ordered_idx_cucd(df: pd.DataFrame, mask) -> np.ndarray:
    sub = df.loc[mask]
    if "TimeRadians" in df.columns:
        sub, _ = _chronological_order(sub, time_col="TimeRadians")
    return sub.index.to_numpy()


def build_cucd_id_tasks_temporal(
    df: pd.DataFrame,
    seed: int = 0,
    test_frac: float = 0.30,
    initial_n_per_class: int = 700,
    uplink_n_per_class: int = 150,
    val_frac: float = 0.20,
    val_cap_per_class: int = 300,
) -> TaskSequence:
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in df.columns if c not in ("Label", "label_name")]
    tasks, class_names_global = [], []
    scaler_mean = scaler_std = None
    for t, (tname, class_list) in enumerate(CUCD_TASK_DEFS):
        n_per_class = initial_n_per_class if t == 0 else uplink_n_per_class
        Xs_train, ys_train, Xs_test, ys_test, Xs_val, ys_val = [], [], [], [], [], []
        for local_c, lbl in enumerate(class_list):
            class_names_global.append(lbl)
            idx_sorted = _ordered_idx_cucd(df, df["label_name"] == lbl)
            train_pool, val_pool, test_idx = _temporal_block_split(idx_sorted, test_frac, val_frac)
            # Train budget: earliest block (draw deterministically: head of block,
            # so temporal order is preserved in expectation; shuffle only within batch later).
            train_idx = train_pool[:n_per_class] if len(train_pool) > n_per_class else train_pool
            val_idx = val_pool[:val_cap_per_class]
            Xs_train.append(df.loc[train_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_train.append(np.full(len(train_idx), local_c, dtype=np.int64))
            Xs_test.append(df.loc[test_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_test.append(np.full(len(test_idx), local_c, dtype=np.int64))
            if len(val_idx):
                Xs_val.append(df.loc[val_idx, feature_cols].to_numpy(dtype=np.float32))
                ys_val.append(np.full(len(val_idx), local_c, dtype=np.int64))
        X_train = np.concatenate(Xs_train, axis=0)
        y_train = np.concatenate(ys_train, axis=0)
        X_test = np.concatenate(Xs_test, axis=0)
        y_test = np.concatenate(ys_test, axis=0)
        X_val = np.concatenate(Xs_val, axis=0) if Xs_val else None
        y_val = np.concatenate(ys_val, axis=0) if ys_val else None
        if t == 0:
            scaler_mean, scaler_std = _fit_scaler(X_train)
        X_train = _apply_scaler(X_train, scaler_mean, scaler_std)
        X_test = _apply_scaler(X_test, scaler_mean, scaler_std)
        if X_val is not None:
            X_val = _apply_scaler(X_val, scaler_mean, scaler_std)
        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        tasks.append(Task(task_id=t, name=tname, class_names=class_list,
                          X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          X_val=X_val, y_val=y_val))
    _assert_split_integrity(tasks)
    return TaskSequence("CuCD-ID", tasks, feature_cols, scaler_mean, scaler_std,
                        class_names_global, split_mode="temporal", val_frac=val_frac)


def build_stin_tasks_temporal(
    df: pd.DataFrame,
    seed: int = 0,
    test_frac: float = 0.20,
    initial_n_per_class: int = 700,
    uplink_n_per_class: int = 150,
    max_test_per_class: int = 1200,
    val_frac: float = 0.20,
    val_cap_per_class: int = 300,
) -> TaskSequence:
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in df.columns if c not in ("id", "Label", "label_name")]
    order_col = "id" if "id" in df.columns else None
    tasks, class_names_global = [], []
    scaler_mean = scaler_std = None
    for t, (tname, class_labels) in enumerate(STIN_TASK_DEFS):
        n_per_class = initial_n_per_class if t == 0 else uplink_n_per_class
        Xs_train, ys_train, Xs_test, ys_test, Xs_val, ys_val = [], [], [], [], [], []
        for local_c, lbl in enumerate(class_labels):
            class_names_global.append(lbl)
            sub = df.loc[df["label_name"] == lbl]
            if order_col:
                sub = sub.sort_values(order_col, kind="mergesort")
            idx_sorted = sub.index.to_numpy()
            train_pool, val_pool, test_idx = _temporal_block_split(idx_sorted, test_frac, val_frac)
            if len(test_idx) > max_test_per_class:
                test_idx = test_idx[-max_test_per_class:]
            train_idx = train_pool[:n_per_class] if len(train_pool) > n_per_class else train_pool
            val_idx = val_pool[:val_cap_per_class]
            Xs_train.append(df.loc[train_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_train.append(np.full(len(train_idx), local_c, dtype=np.int64))
            Xs_test.append(df.loc[test_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_test.append(np.full(len(test_idx), local_c, dtype=np.int64))
            if len(val_idx):
                Xs_val.append(df.loc[val_idx, feature_cols].to_numpy(dtype=np.float32))
                ys_val.append(np.full(len(val_idx), local_c, dtype=np.int64))
        X_train = np.concatenate(Xs_train, axis=0)
        y_train = np.concatenate(ys_train, axis=0)
        X_test = np.concatenate(Xs_test, axis=0)
        y_test = np.concatenate(ys_test, axis=0)
        X_val = np.concatenate(Xs_val, axis=0) if Xs_val else None
        y_val = np.concatenate(ys_val, axis=0) if ys_val else None
        if t == 0:
            scaler_mean, scaler_std = _fit_scaler(X_train)
        X_train = _apply_scaler(X_train, scaler_mean, scaler_std)
        X_test = _apply_scaler(X_test, scaler_mean, scaler_std)
        if X_val is not None:
            X_val = _apply_scaler(X_val, scaler_mean, scaler_std)
        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        tasks.append(Task(task_id=t, name=tname, class_names=class_labels,
                          X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          X_val=X_val, y_val=y_val))
    _assert_split_integrity(tasks)
    return TaskSequence("STIN", tasks, feature_cols, scaler_mean, scaler_std,
                        class_names_global, split_mode="temporal", val_frac=val_frac)


def build_cucd_id_tasks_purged_temporal(
    df: pd.DataFrame,
    seed: int = 0,
    test_frac: float = 0.30,
    initial_n_per_class: int = 700,
    uplink_n_per_class: int = 150,
    val_frac: float = 0.20,
    val_cap_per_class: int = 300,
    purge_time_rad: float = 0.02,
    feature_drop=None,
) -> TaskSequence:
    """Leakage-safe temporal split with purge gaps on TimeRadians.
    Layout per class: [train_pool | purge | val | purge | test]
    The purge gaps use TimeRadians units (0.02 rad ≈ 22s equivalent)
    to exceed the 20-second sliding-window feature horizon
    and the documented 20-22s window-clearing pauses in CuCD-ID."""
    import warnings
    rng = np.random.default_rng(seed)
    drop = set(feature_drop or [])
    feature_cols = [c for c in df.columns
                    if c not in ("Label", "label_name") and c not in drop]
    tasks, class_names_global = [], []
    scaler_mean = scaler_std = None
    for t, (tname, class_list) in enumerate(CUCD_TASK_DEFS):
        n_per_class = initial_n_per_class if t == 0 else uplink_n_per_class
        Xs_train, ys_train, Xs_test, ys_test, Xs_val, ys_val = [], [], [], [], [], []
        for local_c, lbl in enumerate(class_list):
            class_names_global.append(lbl)
            mask = df["label_name"] == lbl
            train_pool, val_pool, test_idx = _temporal_purged_split(
                df, mask, test_frac, val_frac, purge_time_rad, time_col="TimeRadians")
            # Train budget: earliest block (deterministic head)
            train_idx = train_pool[:n_per_class] if len(train_pool) > n_per_class else train_pool
            if len(train_idx) == 0:
                raise ValueError(
                    f"Purged temporal split emptied train for class '{lbl}' "
                    f"(task {tname}, seed={seed}, purge_time_rad={purge_time_rad}). "
                    "Reduce the purge or check the class TimeRadians span.")
            if len(test_idx) == 0:
                raise ValueError(
                    f"Purged temporal split emptied test for class '{lbl}' "
                    f"(task {tname}, seed={seed}).")
            val_idx = val_pool[:val_cap_per_class]
            if val_frac > 0 and len(val_idx) == 0:
                warnings.warn(
                    f"Purged temporal split emptied val for class '{lbl}' "
                    f"(task {tname}, seed={seed}, purge_time_rad={purge_time_rad}); "
                    "BiC/val-Fisher lose this class. Narrow TimeRadians span suspected.")
            Xs_train.append(df.loc[train_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_train.append(np.full(len(train_idx), local_c, dtype=np.int64))
            Xs_test.append(df.loc[test_idx, feature_cols].to_numpy(dtype=np.float32))
            ys_test.append(np.full(len(test_idx), local_c, dtype=np.int64))
            if len(val_idx):
                Xs_val.append(df.loc[val_idx, feature_cols].to_numpy(dtype=np.float32))
                ys_val.append(np.full(len(val_idx), local_c, dtype=np.int64))
        X_train = np.concatenate(Xs_train, axis=0)
        y_train = np.concatenate(ys_train, axis=0)
        X_test = np.concatenate(Xs_test, axis=0)
        y_test = np.concatenate(ys_test, axis=0)
        X_val = np.concatenate(Xs_val, axis=0) if Xs_val else None
        y_val = np.concatenate(ys_val, axis=0) if ys_val else None
        if t == 0:
            scaler_mean, scaler_std = _fit_scaler(X_train)
        X_train = _apply_scaler(X_train, scaler_mean, scaler_std)
        X_test = _apply_scaler(X_test, scaler_mean, scaler_std)
        if X_val is not None:
            X_val = _apply_scaler(X_val, scaler_mean, scaler_std)
        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        tasks.append(Task(task_id=t, name=tname, class_names=class_list,
                          X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          X_val=X_val, y_val=y_val))
    _assert_split_integrity(tasks)
    return TaskSequence("CuCD-ID", tasks, feature_cols, scaler_mean, scaler_std,
                        class_names_global, split_mode="temporal_purged", val_frac=val_frac)
