"""Standard continual-learning metrics computed from an accuracy matrix R,
where R[i, j] = test accuracy on task j immediately after finishing training
on task i (only defined for j <= i). NaN elsewhere.

  ACC  -- average accuracy across all tasks after learning the final task.
  BWT  -- backward transfer (Lopez-Paz & Ranzato, 2017). Negative = forgetting.
  FM   -- forgetting measure (Chaudhry et al., 2018): average of the largest
          drop from a task's peak accuracy to its final accuracy.

Also provides a per-class confusion matrix / precision / recall / F1
breakdown (`confusion_matrix`, `per_class_prf1`) for the final trained
model. ACC/BWT/FM are task-level aggregates; the per-class breakdown makes
the same "old classes forgotten" phenomenon visible at the level of
individual attack types -- e.g. showing an old class's F1 collapse to ~0
under naive fine-tuning even though the task-level ACC number alone only
implies it indirectly.
"""
import numpy as np

from inference import concatenate_test, eval_logits


def compute_cl_metrics(R: np.ndarray):
    T = R.shape[0]
    final_row = R[T - 1]
    acc = float(np.nanmean(final_row))

    if T > 1:
        bwt_terms = [R[T - 1, j] - R[j, j] for j in range(T - 1)]
        bwt = float(np.mean(bwt_terms))

        fm_terms = []
        for j in range(T - 1):
            peak = np.nanmax(R[j:T - 1, j]) if T - 1 > j else R[j, j]
            fm_terms.append(peak - R[T - 1, j])
        fm = float(np.mean(fm_terms))
    else:
        bwt = 0.0
        fm = 0.0

    return {"ACC": acc, "BWT": bwt, "FM": fm}


def compute_macro_f1_balanced_acc(model, task_seq, corrector=None, ncm=None):
    """Compute Macro-F1 and Balanced Accuracy on the final model's test predictions.
    Evaluates on the concatenated test sets of all tasks seen so far.
    Uses the same deployed predictor as the rest of the evaluation: NCM when
    supplied, otherwise BiC-corrected or raw softmax logits."""
    X_all, y_all = concatenate_test(task_seq)
    logits = eval_logits(model, X_all, corrector=corrector, ncm=ncm)
    y_pred = logits.argmax(axis=1)
    n_classes = len(task_seq.class_names_global)

    # Per-class metrics
    cm = confusion_matrix(y_all, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                    out=np.zeros_like(tp), where=(precision + recall) > 0)

    macro_f1 = float(np.mean(f1))

    # Balanced Accuracy = mean recall per class
    bal_acc = float(np.mean(recall))

    return {"MacroF1": macro_f1, "BalancedAccuracy": bal_acc}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """cm[i, j] = number of true-class-i samples predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Per-class precision / recall / F1 / support, computed from a
    confusion matrix (no sklearn dependency -- consistent with the rest of
    this from-scratch pipeline). Classes with zero predicted or zero true
    samples get precision/recall/F1 = 0 rather than NaN, so per-class bar
    charts render cleanly (a class the model never predicts, or that never
    appears in this test slice, genuinely has zero recall/precision)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                    out=np.zeros_like(tp), where=(precision + recall) > 0)
    support = cm.sum(axis=1)
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": support, "confusion_matrix": cm}


# --------------------------------------------------------------------------- #
# Phase 1: binary detection metrics (normal-vs-attack) + from-scratch stats.
# CuCD-ID has a `normal` class, so a multiclass CIL head also implies a
# detector: score = 1 - p(normal). STIN has no benign class and must NOT use
# these (callers should guard on dataset name). No sklearn/scipy: all ranks,
# curves and p-values below are hand-implemented in NumPy.
# --------------------------------------------------------------------------- #

def detection_scores_from_logits(logits: np.ndarray, normal_idx: int) -> np.ndarray:
    """P(attack) = 1 - softmax(logits)[normal]. Higher => more attack-like."""
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return (1.0 - probs[:, normal_idx]).astype(np.float64)


def _binary_auroc(y_bin: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U / rank formulation. Returns 0.5 on degenerate input."""
    y_bin = np.asarray(y_bin).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_bin.sum())
    n_neg = len(y_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = float(ranks[y_bin == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _binary_aupr(y_bin: np.ndarray, scores: np.ndarray) -> float:
    """Average-precision (step-function PR-AUC). Baseline = prevalence."""
    y_bin = np.asarray(y_bin).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_bin.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ys = y_bin[order]
    tp_cum = np.cumsum(ys)
    precision = tp_cum / (np.arange(len(ys)) + 1.0)
    recall = tp_cum / n_pos
    # step integration over recall gains
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        if r > prev_r:
            ap += float(p) * float(r - prev_r)
            prev_r = float(r)
    return float(np.clip(ap, 0.0, 1.0))


def _fpr_at_tpr(y_bin: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    y_bin = np.asarray(y_bin).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y_bin.sum()), int((1 - y_bin).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ys = y_bin[order]
    tp_cum = np.cumsum(ys)
    fp_cum = np.cumsum(1 - ys)
    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg
    idx = np.flatnonzero(tpr >= target_tpr)
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def compute_detection_metrics(y_true_global: np.ndarray, logits: np.ndarray,
                              normal_global_idx: int) -> dict:
    """Binary normal-vs-attack metrics from a multiclass head. Returns
    AUROC / AUPR / FPR@95%TPR plus prevalence (for AUPR context)."""
    y_bin = (np.asarray(y_true_global) != normal_global_idx).astype(int)
    scores = detection_scores_from_logits(np.asarray(logits), int(normal_global_idx))
    return {
        "DET_AUROC": _binary_auroc(y_bin, scores),
        "DET_AUPR": _binary_aupr(y_bin, scores),
        "DET_FPR95": _fpr_at_tpr(y_bin, scores, 0.95),
        "DET_PREV": float(y_bin.mean()) if len(y_bin) else float("nan"),
    }


def _normal_cdf(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> dict:
    """Exact-ish Wilcoxon signed-rank for small n (normal approx, no scipy).
    Returns W+, z, two-sided p. Zero differences are dropped; ties averaged."""
    import math
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "W": 0.0, "z": 0.0, "p_two_sided": 1.0}
    absd = np.abs(d)
    order = np.argsort(absd, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_pos = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4.0
    # tie correction
    _, counts = np.unique(absd, return_counts=True)
    tie = float(np.sum(counts ** 3 - counts) / 48.0)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie
    if var <= 0:
        return {"n": n, "W": w_pos, "z": 0.0, "p_two_sided": 1.0}
    z = (w_pos - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {"n": n, "W": w_pos, "z": float(z), "p_two_sided": float(np.clip(p, 0.0, 1.0))}


def model_persistent_footprint(model, buffer=None, ewc=None, ncm=None):
    """Persistent deployed storage footprint (what gets stored in Flash/EEPROM).
    INT8 model weights + INT8 replay buffer + FP16 EWC + FP16 NCM class means."""
    model_bytes = model.param_count() * 1  # INT8
    buffer_bytes = buffer.memory_bytes(quantized=True) if buffer is not None else 0
    ewc_bytes = ewc.memory_bytes(bytes_per_val=2) if ewc is not None else 0  # FP16
    ncm_bytes = ncm.memory_bytes(bytes_per_val=2) if ncm is not None else 0  # FP16
    total = model_bytes + buffer_bytes + ewc_bytes + ncm_bytes
    return {
        "persistent_model_bytes_int8": model_bytes,
        "persistent_buffer_bytes_int8": buffer_bytes,
        "persistent_ewc_bytes_fp16": ewc_bytes,
        "persistent_ncm_bytes_fp16": ncm_bytes,
        "persistent_cmmu_bytes": buffer_bytes + ewc_bytes + ncm_bytes,
        "persistent_total_bytes": total,
        "persistent_total_KB": total / 1024.0,
    }


def model_peak_training_ram(model, buffer=None, ewc=None, teacher=None,
                             batch_size: int = 32, n_features: int = 22,
                             n_replay: int = 0, retained_state_bytes: int = 0,
                             ncm=None):
    """Estimated peak RAM during training (SRAM usage on MCU).

    FP32 weights + gradients + Adam states + *batch* activations + current
    batch + replay batch + teacher + temp + any retained update-time state
    (e.g. cumulative validation arrays used by EWC/BiC).
    """
    n_params = model.param_count()

    # FP32 model weights (4 bytes each)
    weights_bytes = n_params * 4

    # Gradients (same size as weights, FP32)
    gradients_bytes = n_params * 4

    # Adam optimizer states: m and v for each parameter (2 * 4 bytes each)
    adam_bytes = n_params * 4 * 2  # m + v

    # Activations for the actual minibatch, FP32. The old batch=1 estimate
    # understated GRU/CNN peak RAM by roughly the batch factor.
    act_floats = _estimate_activation_floats(model, n_features)
    activations_bytes = act_floats * 4 * int(batch_size)

    # Training batch (input + labels)
    batch_bytes = batch_size * n_features * 4 + batch_size * 8

    # Replay batch (if applicable)
    replay_batch_bytes = n_replay * n_features * 4 + n_replay * 8 if n_replay > 0 else 0

    # Teacher model (for distillation methods): weights + its minibatch
    # activations. The teacher sees the replay-augmented batch.
    if teacher is not None:
        teacher_bytes = teacher.param_count() * 4
        teacher_bytes += (_estimate_activation_floats(teacher, n_features)
                          * 4 * int(batch_size + n_replay))
    else:
        teacher_bytes = 0

    # Temporary buffers (gradients, intermediate computations) ~2x weights
    temp_bytes = n_params * 4 * 2

    peak_bytes = (weights_bytes + gradients_bytes + adam_bytes +
                  activations_bytes + batch_bytes + replay_batch_bytes +
                  teacher_bytes + temp_bytes)

    # Also include CMMU structures in FP32 during training
    buffer_fp32 = buffer.memory_bytes(quantized=False) if buffer is not None else 0
    ewc_fp32 = ewc.memory_bytes(bytes_per_val=4) if ewc is not None else 0
    ncm_fp32 = ncm.memory_bytes(bytes_per_val=4) if ncm is not None else 0

    peak_bytes += buffer_fp32 + ewc_fp32 + ncm_fp32 + int(retained_state_bytes)

    return {
        "peak_weights_bytes_fp32": weights_bytes,
        "peak_gradients_bytes_fp32": gradients_bytes,
        "peak_adam_bytes_fp32": adam_bytes,
        "peak_activations_bytes_fp32": activations_bytes,
        "peak_batch_bytes_fp32": batch_bytes,
        "peak_replay_batch_bytes_fp32": replay_batch_bytes,
        "peak_teacher_bytes_fp32": teacher_bytes,
        "peak_temp_bytes_fp32": temp_bytes,
        "peak_buffer_bytes_fp32": buffer_fp32,
        "peak_ewc_bytes_fp32": ewc_fp32,
        "peak_ncm_bytes_fp32": ncm_fp32,
        "peak_retained_state_bytes": int(retained_state_bytes),
        "peak_total_bytes": peak_bytes,
        "peak_total_KB": peak_bytes / 1024.0,
    }


def _estimate_activation_floats(model, n_features: int) -> int:
    """Estimate FP32 activation count for batch=1 forward pass."""
    name = model.NAME
    if name == "MLP":
        # W1: (F, h1) -> h1, W2: (h1, h2) -> h2, head: (h2, C) -> C
        h1 = model.params["W1"].shape[1]
        h2 = model.params["W2"].shape[1]
        C = model.params["head.W"].shape[1] if "head.W" in model.params else 0
        return h1 + h2 + C
    if name == "GRU":
        H = model.hidden
        S = model.seq_len
        # Per step: 3 gates * (H + H) + H hidden state
        per_step = 3 * 2 * H + H
        C = model.params["head.W"].shape[1] if "head.W" in model.params else 0
        return per_step * S + C
    if name == "CNN1D":
        c1 = model.c1
        c2 = model.c2
        L = model.L
        # Conv1: L * c1, Conv2: L * c2, head: C
        C = model.params["head.W"].shape[1] if "head.W" in model.params else 0
        return L * c1 + L * c2 + C
    raise ValueError(name)


def estimate_macs(model, n_features: int) -> int:
    """Rough multiply-accumulate count for one forward pass (batch=1).
    Includes the trunk plus the CURRENT width of the shared classification
    head (head compute grows as more classes are learned -- unlike the
    task-incremental multi-head design, only one head is ever active, but
    its width is the total number of classes seen so far)."""
    name = model.NAME
    head_macs = model.params["head.W"].size if "head.W" in model.params else 0
    if name == "MLP":
        w1 = model.params["W1"]; w2 = model.params["W2"]
        return int(w1.size + w2.size + head_macs)
    if name == "GRU":
        # 3 gates, each: x@W (F*H) + h@U (H*H), summed over seq_len timesteps
        H = model.hidden
        F = model.feat_per_step
        per_step = 3 * (F * H + H * H)
        return int(per_step * model.seq_len + head_macs)
    if name == "CNN1D":
        c1, c2, k = model.c1, model.c2, model.k
        L = model.L
        conv1 = L * c1 * 1 * k
        conv2 = L * c2 * c1 * k
        return int(conv1 + conv2 + head_macs)
    raise ValueError(name)


def estimate_macs_per_sample(model, n_features: int) -> int:
    """MACs for one forward pass (batch=1)."""
    return estimate_macs(model, n_features)


def estimate_macs_per_update(model, n_features: int, n_train_samples: int,
                              epochs: int, batch_size: int = 32) -> int:
    """Estimated/conservative MACs for one training update.
    Uses the final-head-width per-sample proxy for all tasks (earlier tasks
    had fewer output classes, so this overestimates them); excludes replay
    augmentation, teacher forwards, and val/BiC passes unless counted by
    the caller via performed sample counters."""
    macs_per_fwd = estimate_macs(model, n_features)
    # Forward + backward ≈ 3x forward (gradient computation)
    macs_per_sample = macs_per_fwd * 3
    steps_per_epoch = max(1, n_train_samples // batch_size)
    return macs_per_sample * steps_per_epoch * epochs
