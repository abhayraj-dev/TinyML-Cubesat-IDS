"""
CMMU -- Continual Memory Management Unit.

Two independent sub-mechanisms, combinable:

  A) ReplayBuffer   -- fixed-capacity reservoir-sampling exemplar buffer.
                        Interleaves a handful of old-class samples into every
                        new-task minibatch so gradient steps are not purely
                        one-sided toward the newest classes.

  B) OnlineEWC       -- "Online" Elastic Weight Consolidation (Schwarz et al.,
                        2018). Keeps a *single* running diagonal Fisher-
                        information estimate and a single anchor parameter
                        snapshot (O(1) memory in the number of tasks, which
                        is essential under the assumed 60 KB persistent-state
                        budget -- the original per-task EWC would grow memory
                        linearly with the number of tasks learned).

                        Under the class-incremental protocol used here, the
                        model has a single *shared* classification head that
                        is reused (and grown) across every task, so unlike a
                        task-incremental multi-head setup, the head itself is
                        just as susceptible to forgetting as the trunk (the
                        well-known "task-recency" / logit-competition effect:
                        softmax cross-entropy on a new class's samples pushes
                        down the logits of *every* other class, including old
                        ones). OnlineEWC therefore protects ALL of the model's
                        parameters, trunk and head alike. Because the head
                        grows in width over time, Fisher/anchor arrays for a
                        growing parameter are padded to the new shape -- the
                        newly-added output columns start with zero Fisher
                        (unconstrained) until the task that introduced them
                        is itself consolidated.

Both mechanisms report their own memory footprint in bytes so the total CMMU
budget can be checked against the hardware constraint.
"""
from __future__ import annotations
import numpy as np

from losses import softmax_cross_entropy


def _pad_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Zero-pad arr up to target_shape (only grows, never shrinks); existing
    values are preserved in their original positions."""
    if arr.shape == target_shape:
        return arr
    padded = np.zeros(target_shape, dtype=arr.dtype)
    slices = tuple(slice(0, s) for s in arr.shape)
    padded[slices] = arr
    return padded


class ReplayBuffer:
    def __init__(self, capacity: int, n_features: int, seed: int = 0):
        self.capacity = capacity
        self.n_features = n_features
        self.X = np.zeros((capacity, n_features), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.int64)
        self.task_id = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.n_seen = 0
        self._rng = np.random.default_rng(seed)

    def add_batch(self, X, y, task_id):
        """y must already be in the model's global class-id space."""
        for i in range(X.shape[0]):
            self.n_seen += 1
            if self.size < self.capacity:
                self.X[self.size] = X[i]
                self.y[self.size] = y[i]
                self.task_id[self.size] = task_id
                self.size += 1
            else:
                j = self._rng.integers(0, self.n_seen)
                if j < self.capacity:
                    self.X[j] = X[i]
                    self.y[j] = y[i]
                    self.task_id[j] = task_id

    def sample(self, batch_size: int):
        if self.size == 0:
            return None
        n = min(batch_size, self.size)
        idx = self._rng.choice(self.size, size=n, replace=False)
        return self.X[idx], self.y[idx], self.task_id[idx]

    def sample_balanced(self, batch_size: int):
        """Phase 0: class-balanced replay. Draws ~equally from each buffered
        class (round-robin) instead of proportionally to buffer occupancy, so
        the large T0 pool cannot starve uplink classes in the minibatch.
        Falls back to uniform sampling when a class has no exemplars."""
        if self.size == 0:
            return None
        classes = np.unique(self.y[:self.size])
        if len(classes) <= 1:
            return self.sample(batch_size)
        per_cls = max(1, batch_size // len(classes))
        picks = []
        for c in classes:
            pool = np.flatnonzero(self.y[:self.size] == c)
            n = min(per_cls, len(pool))
            picks.append(self._rng.choice(pool, size=n, replace=False))
        idx = np.concatenate(picks)
        if len(idx) < batch_size:  # top up uniformly
            extra = self._rng.choice(self.size, size=batch_size - len(idx), replace=False)
            idx = np.concatenate([idx, extra])
        else:
            idx = idx[:batch_size]
        return self.X[idx], self.y[idx], self.task_id[idx]

    @staticmethod
    def select_herding(X: np.ndarray, n_select: int, rng=None) -> np.ndarray:
        """Phase 0: iCaRL-style herding — greedily pick exemplars whose mean
        best matches the class mean (in raw feature space here; Phase 1 can
        switch to trunk-embedding space). Returns selected row indices into X.
        Static so Phase-1 iCaRL-lite can reuse it without a buffer instance."""
        n = len(X)
        if n_select >= n or n <= 0:
            return np.arange(n)
        mu = X.mean(axis=0)
        chosen, cur_sum = [], np.zeros_like(mu, dtype=np.float64)
        remaining = set(range(n))
        for k in range(1, n_select + 1):
            best, best_err = -1, None
            for i in remaining:
                m = (cur_sum + X[i]) / k
                err = float(((mu - m) ** 2).sum())
                if best_err is None or err < best_err:
                    best_err, best = err, i
            chosen.append(best)
            cur_sum = cur_sum + X[best]
            remaining.discard(best)
        return np.array(chosen, dtype=np.int64)

    def memory_bytes(self, quantized: bool = True) -> int:
        """Deployed CMMU buffer footprint. Quantized=True assumes int8
        feature storage (features are already roughly unit-scaled by the
        fixed launch-time normalizer) + int8 labels/task-ids."""
        if quantized:
            return int(self.capacity * self.n_features * 1 + self.capacity * 1 + self.capacity * 1)
        return int(self.X.nbytes + self.y.nbytes + self.task_id.nbytes)


class OnlineEWC:
    def __init__(self, lam: float = 1000.0, gamma: float = 1.0, fisher_n_samples: int = 256, seed: int = 0):
        # NOTE: gamma is a validation-tuned weighting on retained Fisher
        # (F_new = gamma*F_old + F_current), not necessarily a decay factor:
        # frozen EWC-only configs use MLP 0.5 / GRU 1.0 / CNN 0.5.
        self.lam = lam
        self.gamma = gamma
        self.fisher_n_samples = fisher_n_samples
        self.fisher: dict[str, np.ndarray] = {}
        self.anchor: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)
        self._consolidated = False

    def penalty_grad(self, params: dict) -> dict:
        """Gradient of (lam/2) * sum F_i (theta_i - theta*_i)^2 w.r.t. each
        param. Handles params whose shape has grown (new head columns)
        since the last consolidation: only the previously-existing
        sub-region is penalised, new columns are left unconstrained."""
        if not self._consolidated:
            return {}
        grads = {}
        for name, F in self.fisher.items():
            if name not in params:
                continue
            p = params[name]
            if p.shape == F.shape:
                grads[name] = self.lam * F * (p - self.anchor[name])
            else:
                g = np.zeros_like(p)
                slices = tuple(slice(0, s) for s in F.shape)
                g[slices] = self.lam * F * (p[slices] - self.anchor[name])
                grads[name] = g
        return grads

    def penalty_loss(self, params: dict) -> float:
        if not self._consolidated:
            return 0.0
        total = 0.0
        for name, F in self.fisher.items():
            if name not in params:
                continue
            p = params[name]
            slices = tuple(slice(0, s) for s in F.shape)
            total += 0.5 * self.lam * np.sum(F * (p[slices] - self.anchor[name]) ** 2)
        return float(total)

    def consolidate(self, model, X_task: np.ndarray, y_task: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Estimate diagonal Fisher information on this task's data (empirical
        Fisher via squared gradient of the log-likelihood of the true label,
        per-sample, averaged) and fold it into the running estimate, then
        snapshot current params as the new anchor. ALL current model
        parameters (trunk + the shared, currently-full-width head) are
        protected -- see module docstring.

        When X_val/y_val are supplied, the Fisher sample is drawn from a
        class-balanced mix of (X_task, X_val). `train_continual` passes the
        cumulative calibration pool (old + new tasks), so old-task curvature
        is represented rather than only the current task. Callers may omit
        them (legacy behaviour: Fisher on X_task only)."""
        protected_names = list(model.params.keys())
        # Phase 0 val-balanced Fisher: mix new-train with val (old+new) so the
        # sample is not dominated by the current task. Balance by class.
        if X_val is not None and y_val is not None and len(X_val):
            X_all = np.concatenate([X_task, X_val], axis=0)
            y_all = np.concatenate([y_task, y_val], axis=0)
            classes = np.unique(y_all)
            per = max(1, self.fisher_n_samples // max(1, len(classes)))
            picks = []
            for c in classes:
                pool = np.flatnonzero(y_all == c)
                n_take = min(per, len(pool))
                picks.append(self._rng.choice(pool, size=n_take, replace=False))
            idx = np.concatenate(picks)[:self.fisher_n_samples]
            self._rng.shuffle(idx)
            Xs, ys = X_all[idx], y_all[idx]
        else:
            n = X_task.shape[0]
            idx = self._rng.choice(n, size=min(self.fisher_n_samples, n), replace=False)
            Xs, ys = X_task[idx], y_task[idx]

        fisher_accum = {name: np.zeros_like(model.params[name]) for name in protected_names}
        for i in range(Xs.shape[0]):
            xb = Xs[i:i + 1]
            yb = ys[i:i + 1]
            logits, cache = model.forward(xb)
            _, dlogits, _ = softmax_cross_entropy(logits, yb)
            grads = model.backward(cache, dlogits)
            for name in protected_names:
                if name in grads:
                    fisher_accum[name] += grads[name] ** 2
        for name in protected_names:
            fisher_accum[name] /= max(1, Xs.shape[0])

        if not self._consolidated:
            self.fisher = {name: fisher_accum[name].copy() for name in protected_names}
        else:
            for name in protected_names:
                prev = self.fisher.get(name)
                if prev is None:
                    prev = np.zeros_like(fisher_accum[name])
                else:
                    prev = _pad_to_shape(prev, fisher_accum[name].shape)
                self.fisher[name] = self.gamma * prev + fisher_accum[name]

        self.anchor = {name: model.params[name].copy() for name in protected_names}
        self._consolidated = True

    def memory_bytes(self, bytes_per_val: int = 2) -> int:
        """Fisher + anchor snapshot, stored at reduced precision (float16 by
        default) since exact precision is not required for the quadratic
        penalty -- halves the CMMU memory cost vs. float32."""
        total = 0
        for name, F in self.fisher.items():
            total += F.size * bytes_per_val * 2  # fisher + anchor
        return int(total)
