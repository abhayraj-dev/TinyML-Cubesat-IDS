"""
Class-incremental continual-learning training loop.

Unlike a task-incremental setup, there is no task ID at inference time: all
classes learned so far are discriminated through ONE shared, dynamically
growing softmax head (`model.expand_head`). Each Task's labels are stored
locally (0..n_new_classes-1, see `src/data_utils.py`); this loop resolves
them to the model's actual global head-column indices *dynamically*, based
on the order tasks are processed in (`class_offset` below) -- so the very
same TaskSequence can safely be replayed in a different task order (used by
the task-order-sensitivity ablation) without corrupting label alignment.

Methods:
  finetune -- naive sequential fine-tuning, no CMMU. Lower-bound baseline,
              expected to catastrophically forget earlier classes -- and,
              under the class-incremental protocol, doubly so, since new-
              class training actively pushes down old classes' logits via
              softmax competition even before considering weight drift.
  replay   -- CMMU sub-mechanism A only (experience replay buffer).
  ewc      -- CMMU sub-mechanism B only (online EWC regularisation, now
              covering the shared head as well as the trunk).
  full     -- both CMMU sub-mechanisms combined (the proposed system).
  joint    -- LEGACY cumulative warm-start: trains on the union of all
              classes' training data seen so far but continues from current
              weights at every task boundary. Kept for comparability with
              old CSVs; do NOT cite as a strict upper bound.
  joint_scratch -- (Phase 0) offline joint-training reference (NOT a strict
               upper bound: warm-start joint can exceed it): at each task boundary
              the model AND optimizer are re-initialised from scratch and
              trained on the union of training data seen so far. Strictly
              stronger reference than `joint`.
  lwf      -- (Phase 1) Learning-without-Forgetting: memory-free distillation
              from a frozen teacher snapshot (Hinton T, lambda); no replay.
  bic      -- (Phase 1) replay (herding, balanced) + distillation + BiC
              bias-correction stage fit on held-out val (alpha,beta on new
              logits, Wu et al. 2019 lite: correction at inference, trunk
              frozen during fit).
  icarl    -- (Phase 1 lite) herding buffer + balanced replay + distillation,
              softmax eval (embedding-space herding / NCM omitted; documented
              limitation vs full iCaRL).
"""
from __future__ import annotations
import time
import numpy as np

from optim import Adam
from cmmu import ReplayBuffer, OnlineEWC
from losses import softmax_cross_entropy
from utils import set_seed


DEFAULT_HPARAMS = dict(
    lr=3e-3,
    epochs=25,
    batch_size=32,
    ewc_lambda=1000.0,
    ewc_gamma=1.0,  # fallback only: frozen per-arch configs always override
                    # (MLP 0.5 / GRU 2.0 / CNN 0.5); gamma is treated as a
                    # validation-tuned weighting parameter, not a decay factor
    fisher_n_samples=200,
    buffer_capacity=128,
    replay_frac=0.5,
    # Phase 1 distillation / BiC-iCaRL knobs (defaults keep old runs identical
    # when the new methods are not selected).
    distill_lambda=1.0,   # weight on KD term for lwf/bic/icarl (CE gets 1-distill_lambda when <1)
    distill_T=2.0,        # Hinton temperature for soft targets
)


def _softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = (logits - logits.max(axis=1, keepdims=True)) / max(T, 1e-8)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def _distill_dlogits(student_logits: np.ndarray, teacher_logits_old: np.ndarray,
                     T: float, lam: float) -> np.ndarray:
    """Gradient of lam * T^2 * KL(teacher_old/T || student_old/T) averaged over
    the batch, padded with zeros for new-class columns. Follows Hinton et al.
    (the T^2 keeps the KD step magnitude comparable across temperatures)."""
    n_old = teacher_logits_old.shape[1]
    if n_old == 0:
        return np.zeros_like(student_logits)
    p_t = _softmax_T(teacher_logits_old, T)
    p_s = _softmax_T(student_logits[:, :n_old], T)
    N = student_logits.shape[0]
    g_old = lam * (T ** 2) * (p_s - p_t) / (T * max(N, 1))
    out = np.zeros_like(student_logits)
    out[:, :n_old] = g_old
    return out


def _snapshot_teacher(model_cls, model_kwargs, model, seed: int):
    """Frozen copy of the current model (trunk + old head width). The teacher
    is never expanded again; student grows past it."""
    teacher = model_cls(**model_kwargs, seed=seed)
    for k, v in model.params.items():
        teacher.params[k] = v.copy()
    return teacher


def _rebuild_herding_buffer(buffer, per_class_data: dict, capacity: int):
    """Phase 1 iCaRL-style buffer rebuild: equal exemplars per class, herding
    selection in input space (documented limitation: embedding-space herding
    would need trunk access per arch; input-space is the honest lite version).
    per_class_data: {global_class_id: X (N,F)}. Clears buffer first."""
    from cmmu import ReplayBuffer
    buffer.size = 0
    buffer.n_seen = 0
    if not per_class_data:
        return
    classes = sorted(per_class_data.keys())
    per = max(1, capacity // max(1, len(classes)))
    for c in classes:
        X = np.asarray(per_class_data[c])
        if len(X) == 0:
            continue
        sel = ReplayBuffer.select_herding(X, min(per, len(X)))
        for i in sel:
            if buffer.size >= buffer.capacity:
                return
            buffer.X[buffer.size] = X[i]
            buffer.y[buffer.size] = c
            buffer.task_id[buffer.size] = -1  # herding rebuild loses task id; eval doesn't need it
            buffer.size += 1


def _train_step_batch(model, xb, yb, opt, ewc, teacher=None, teacher_logits=None,
                        distill_lam: float = 0.0, distill_T: float = 2.0):
    """One Adam step on a (possibly replay-augmented) minibatch, all routed
    through the single shared head -- no per-task grouping needed.
    When teacher_logits (old-head soft targets) are given, adds the KD
    gradient on old-class columns (LwF/BiC/iCaRL family)."""
    logits, cache = model.forward(xb)
    loss, dlogits, _ = softmax_cross_entropy(logits, yb)
    if teacher_logits is not None and distill_lam > 0:
        # Additive KD (keeps full CE on new classes; lam scales old-class
        # retention). NOTE: earlier convex combo (1-lam)*CE killed plasticity
        # at lam=1 -- fixed after 3-epoch smoke showed lwf ACC~=chance.
        dlogits = dlogits + _distill_dlogits(logits, teacher_logits, distill_T, distill_lam)
    grads = model.backward(cache, dlogits)

    if ewc is not None:
        pen_g = ewc.penalty_grad(model.params)
        for k, v in pen_g.items():
            grads[k] = grads.get(k, 0.0) + v

    opt.step(model.params, grads)
    return loss


class BiasCorrector:
    """Phase 0 BiC-style inference hook (Wu et al., 2019, light version).

    Models the well-known new-class head bias as
        logits_new_corrected = alpha * logits_new + beta
    with old-class logits untouched. One (alpha, beta) pair is fit per
    incremental stage on held-out val/buffer data with trunk+head frozen
    (grid search on accuracy); full BiC two-stage training lands in Phase 1.
    """

    def __init__(self):
        self.stages = []  # list of dicts {n_old, n_new, alpha, beta}

    @property
    def fitted(self) -> bool:
        return len(self.stages) > 0

    def apply(self, logits: np.ndarray) -> np.ndarray:
        out = logits.copy()
        for s in self.stages:
            n_old, n_new, alpha, beta = s["n_old"], s["n_new"], s["alpha"], s["beta"]
            if logits.shape[1] >= n_old + n_new:
                out[:, n_old:n_old + n_new] = alpha * out[:, n_old:n_old + n_new] + beta
        return out

    def fit_stage(self, logits_val: np.ndarray, y_val: np.ndarray,
                  n_old: int, n_new: int, alphas=None, betas=None) -> dict:
        if alphas is None:
            alphas = np.concatenate([[1.0], np.linspace(0.1, 2.0, 12)])
        if betas is None:
            betas = np.array([0.0])
        best = {"n_old": n_old, "n_new": n_new, "alpha": 1.0, "beta": 0.0, "acc": -1.0}
        for a in alphas:
            for b in betas:
                corr = logits_val.copy()
                corr[:, n_old:n_old + n_new] = float(a) * corr[:, n_old:n_old + n_new] + float(b)
                acc = float((corr.argmax(axis=1) == y_val).mean()) if len(y_val) else -1.0
                if acc > best["acc"]:
                    best = {"n_old": n_old, "n_new": n_new, "alpha": float(a), "beta": float(b), "acc": acc}
        self.stages.append({k: best[k] for k in ("n_old", "n_new", "alpha", "beta")})
        return best


def train_continual(model_cls, model_kwargs, task_seq, method: str, hparams: dict, seed: int = 0,
                    bias_correction: bool = False, replay_balance: str = "uniform"):
    assert method in ("finetune", "replay", "ewc", "full", "joint", "joint_scratch",
                      "lwf", "bic", "icarl")
    assert replay_balance in ("uniform", "class_balanced")
    hp = {**DEFAULT_HPARAMS, **hparams}
    set_seed(seed)
    rng = np.random.default_rng(seed)

    model = model_cls(**model_kwargs, seed=seed)
    opt = Adam(lr=hp["lr"])

    NEED_BUFFER = method in ("replay", "full", "bic", "icarl")
    NEED_DISTILL = method in ("lwf", "bic", "icarl")
    NEED_BIC_FIT = method in ("bic",) or bias_correction
    buffer = ReplayBuffer(hp["buffer_capacity"], task_seq.n_features, seed=seed) \
        if NEED_BUFFER else None
    ewc = OnlineEWC(lam=hp["ewc_lambda"], gamma=hp["ewc_gamma"],
                     fisher_n_samples=hp["fisher_n_samples"], seed=seed) \
        if method in ("ewc", "full") else None
    corrector = BiasCorrector() if NEED_BIC_FIT else None
    # BiC/iCaRL papers mandate balanced rehearsal; force it regardless of flag.
    use_balanced = (replay_balance == "class_balanced") or method in ("bic", "icarl")
    distill_lam = float(hp.get("distill_lambda", 1.0))
    distill_T = float(hp.get("distill_T", 2.0))

    T = len(task_seq.tasks)
    R = np.full((T, T), np.nan)
    V = np.full((T, T), np.nan)  # Validation accuracy matrix
    per_task_train_seconds = []
    joint_pool = []          # for method="joint"/"joint_scratch": list of (X, y_global)
    seen_tasks = []          # list of (X_test, y_test_global) for tasks processed so far
    seen_val_tasks = []      # list of (X_val, y_val_global) for validation evaluation
    seen_val = []            # Phase 0: (X_val, y_val_global) for BiasCorrector fitting
    class_offset = 0         # cumulative #classes introduced so far, in PROCESSING order
    n_classes_before = []    # n_old at each stage (for bias correction)
    last_teacher = None      # frozen teacher from the most recent incremental stage (for peak-RAM accounting)
    train_steps_total = 0    # minibatch optimizer steps actually performed
    train_samples_total = 0  # student samples actually processed (incl. replay augmentation)
    teacher_samples_total = 0  # teacher-forward samples actually processed (distillation methods)

    for t, task in enumerate(task_seq.tasks):
        n_old = class_offset
        n_classes_before.append(n_old)
        # Teacher snapshot BEFORE growing the head (frozen old trunk+head).
        teacher = None
        if NEED_DISTILL and n_old > 0:
            teacher = _snapshot_teacher(model_cls, model_kwargs, model, seed)
            last_teacher = teacher
        if method == "joint_scratch":
            # Offline reference: fresh weights + fresh optimizer at every boundary.
            model = model_cls(**model_kwargs, seed=seed)
            opt = Adam(lr=hp["lr"])
            for tt in range(t + 1):
                if model.n_classes_total < sum(task_seq.tasks[j].n_new_classes for j in range(tt + 1)):
                    model.expand_head(task_seq.tasks[tt].n_new_classes)
        else:
            model.expand_head(task.n_new_classes)
        y_train_global = task.y_train + class_offset
        y_test_global = task.y_test + class_offset
        y_val_global = task.y_val + class_offset if task.y_val is not None else None
        class_offset += task.n_new_classes

        if method in ("joint", "joint_scratch"):
            joint_pool.append((task.X_train, y_train_global))
            pool = joint_pool
        else:
            pool = [(task.X_train, y_train_global)]

        t0 = time.perf_counter()
        for epoch in range(hp["epochs"]):
            Xc = np.concatenate([x for x, y in pool], axis=0)
            yc = np.concatenate([y for x, y in pool], axis=0)
            perm = rng.permutation(len(Xc))
            Xc, yc = Xc[perm], yc[perm]

            for start in range(0, len(Xc), hp["batch_size"]):
                xb = Xc[start:start + hp["batch_size"]]
                yb = yc[start:start + hp["batch_size"]]

                if buffer is not None and buffer.size > 0:
                    n_replay = max(1, int(len(xb) * hp["replay_frac"]))
                    if use_balanced and hasattr(buffer, "sample_balanced"):
                        sampled = buffer.sample_balanced(n_replay)
                    else:
                        sampled = buffer.sample(n_replay)
                    if sampled is not None:
                        rx, ry, _ = sampled
                        xb = np.concatenate([xb, rx], axis=0)
                        yb = np.concatenate([yb, ry], axis=0)

                t_logits = None
                if teacher is not None:
                    t_logits, _ = teacher.forward(xb)
                    teacher_samples_total += int(xb.shape[0])
                train_steps_total += 1
                train_samples_total += int(yb.shape[0])
                _train_step_batch(model, xb, yb, opt, ewc,
                                  teacher_logits=t_logits,
                                  distill_lam=distill_lam if teacher is not None else 0.0,
                                  distill_T=distill_T)
        per_task_train_seconds.append(time.perf_counter() - t0)

        # ---- evaluate on all tasks seen so far, using the FULL current
        #      shared head (no task ID / no restriction to a subset of
        #      classes -- this is the crux of the class-incremental setting) ----
        seen_tasks.append((task.X_test, y_test_global))
        if task.X_val is not None and y_val_global is not None:
            seen_val.append((task.X_val, y_val_global))
            seen_val_tasks.append((task.X_val, y_val_global))
        # Phase 1 BiC: fit one bias stage on held-out val (frozen model) before
        # eval; full (alpha,beta) grid for method="bic", alpha-only for the
        # generic bias_correction hook.
        if corrector is not None and seen_val and n_old > 0:
            Xv = np.concatenate([x for x, y in seen_val], axis=0)
            yv = np.concatenate([y for x, y in seen_val], axis=0)
            logits_v, _ = model.forward(Xv)
            if method == "bic":
                corrector.fit_stage(logits_v, yv, n_old, task.n_new_classes,
                                    betas=np.linspace(-1.0, 1.0, 9))
            else:
                corrector.fit_stage(logits_v, yv, n_old, task.n_new_classes)
        for j, (Xj, yj) in enumerate(seen_tasks):
            logits, _ = model.forward(Xj)
            if corrector is not None and corrector.fitted:
                logits = corrector.apply(logits)
            preds = logits.argmax(axis=1)
            R[t, j] = float((preds == yj).mean())
        # Validation accuracy matrix (for hyperparameter selection, no test leakage)
        for j, (Xj, yj) in enumerate(seen_val_tasks):
            logits, _ = model.forward(Xj)
            if corrector is not None and corrector.fitted:
                logits = corrector.apply(logits)
            preds = logits.argmax(axis=1)
            V[t, j] = float((preds == yj).mean())

        # ---- update CMMU state ----
        if buffer is not None:
            if method in ("bic", "icarl"):
                # Bounded herding rebuild: candidates are resident buffer
                # exemplars + incoming task batch only (no unbounded history).
                per_class_data = {}
                if buffer.size > 0:
                    buf_X = np.asarray(buffer.X[:buffer.size])
                    buf_y = np.asarray(buffer.y[:buffer.size])
                    for g_old in np.unique(buf_y).tolist():
                        per_class_data[int(g_old)] = buf_X[buf_y == g_old].copy()
                base = class_offset - task.n_new_classes
                for c_local in range(task.n_new_classes):
                    g = base + c_local
                    mask = (y_train_global == g)
                    if np.any(mask):
                        per_class_data[g] = task.X_train[mask].copy()
                _rebuild_herding_buffer(buffer, per_class_data, int(hp["buffer_capacity"]))
            else:
                buffer.add_batch(task.X_train, y_train_global, t)
        if ewc is not None:
            # Phase 0: prefer val-balanced Fisher when val exists (old-task
            # curvature is otherwise understated by new-train-only samples).
            if task.X_val is not None and y_val_global is not None and len(task.X_val):
                try:
                    ewc.consolidate(model, task.X_train, y_train_global,
                                    X_val=task.X_val, y_val=y_val_global)
                except TypeError:
                    ewc.consolidate(model, task.X_train, y_train_global)
            else:
                ewc.consolidate(model, task.X_train, y_train_global)

    return {
        "R": R,
        "V": V,
        "model": model,
        "buffer": buffer,
        "ewc": ewc,
        "corrector": corrector,
        "teacher": last_teacher,
        "train_steps_total": train_steps_total,
        "train_samples_total": train_samples_total,
        "teacher_samples_total": teacher_samples_total,
        "per_task_train_seconds": per_task_train_seconds,
        "hparams": hp,
        "replay_balance": replay_balance,
        "bias_correction": bias_correction,
    }
