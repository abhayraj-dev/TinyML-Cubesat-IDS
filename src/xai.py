"""Post-hoc input attributions and faithfulness metrics for CIL-IDS models.

This module deliberately stays NumPy-only (no shap/captum/scipy dependency).
It implements:

  * Integrated Gradients (IG) -- finite-difference gradients along the path
    from a caller-supplied baseline.  If no baseline is supplied, the default
    is the zero vector in standardised feature space; the XAI experiment
    passes a real training medoid instead.
  * KernelSHAP -- the model-agnostic Shapley-value approximation, solved with
    a constrained weighted least-squares system per sample.  The default run
    uses 256 sampled coalitions; the efficiency constraint is enforced exactly
    through the KKT system, so completeness is exact by construction and is a
    sanity check, not evidence of attribution accuracy.

It also implements the two agreement metrics used by Cerasuolo et al. (2025)
(Top5Ratio, Avg.Rel.) plus two stronger checks that the base paper does not
report:

  * deletion / insertion curves and their AUCs (faithfulness), and
  * Spearman rank agreement between SHAP and IG.

All functions take a ``score_fn`` mapping a 2-D batch ``(N, F)`` to class
logits/scores ``(N, C)``.  This makes the same code work for the softmax head,
the BiC-corrected head, and the NCM cosine head.
"""
from __future__ import annotations

import math

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _as_baseline(baseline, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if baseline is None:
        # Standardised features: zero is the launch-corpus training mean.
        return np.zeros_like(X)
    baseline = np.asarray(baseline, dtype=np.float64)
    if baseline.ndim == 1:
        return np.tile(baseline[None, :], (len(X), 1))
    if baseline.shape != X.shape:
        raise ValueError(f"baseline shape {baseline.shape} != X shape {X.shape}")
    return baseline


def integrated_gradients(score_fn, X: np.ndarray, targets: np.ndarray,
                         baseline=None, steps: int = 16,
                         eps: float = 1e-3, grad_fn=None,
                         integration: str = "trapezoid") -> np.ndarray:
    """Integrated Gradients along the straight-line baseline path.

    If ``grad_fn(X, targets) -> dscore/dX`` is supplied, it is used (the
    hand-written models expose exact input gradients).  Otherwise a central
    finite-difference fallback keeps the implementation model-agnostic.
    ``integration="trapezoid"`` is the default because the original left-
    Riemann sum (including both endpoints with equal weight) has a first-order
    bias that violates completeness; the trapezoidal rule is exact for the
    linear-in-alpha case and materially reduces completeness error for the
    nonlinear models.
    """
    X = np.asarray(X, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    N, F = X.shape
    if len(targets) != N:
        raise ValueError("targets must have one entry per row")
    B = _as_baseline(baseline, X)
    steps = max(1, int(steps))
    if integration not in ("trapezoid", "riemann"):
        raise ValueError("integration must be 'trapezoid' or 'riemann'")
    alphas = np.linspace(0.0, 1.0, steps + 1)
    if integration == "trapezoid":
        weights = np.ones(steps + 1, dtype=np.float64)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights /= float(steps)
    else:
        weights = np.full(steps + 1, 1.0 / float(steps + 1))
    attrs = np.zeros((N, F), dtype=np.float64)

    if grad_fn is not None:
        for a, w in zip(alphas, weights):
            attrs += float(w) * np.asarray(
                grad_fn(B + float(a) * (X - B), targets))
        attrs *= (X - B)
        return attrs

    row_idx = np.arange(N * 2 * F)
    feat_idx = np.tile(np.repeat(np.arange(F), 2), N)
    signs = np.tile(np.array([1.0, -1.0]), N * F)
    for a, w in zip(alphas, weights):
        Xa = B + float(a) * (X - B)
        rows = np.repeat(Xa, 2 * F, axis=0)
        rows[row_idx, feat_idx] += signs * float(eps)
        logits = np.asarray(score_fn(rows))
        tgt = logits[row_idx, np.repeat(targets, 2 * F)].reshape(N, F, 2)
        grad = (tgt[:, :, 0] - tgt[:, :, 1]) / (2.0 * float(eps))
        attrs += float(w) * grad
    attrs *= (X - B)
    return attrs


def _coalition_weights(F: int, masks: np.ndarray) -> np.ndarray:
    """KernelSHAP weights for coalition sizes 1..F-1; endpoints get weight 1."""
    w = np.ones(len(masks), dtype=np.float64)
    for i, m in enumerate(masks):
        s = int(m.sum())
        if 0 < s < F:
            w[i] = (F - 1) / (s * (F - s) * math.comb(F, s))
    w = w / max(float(w.max()), 1e-12)
    return np.maximum(w, 1e-12)


def kernel_shap(score_fn, X: np.ndarray, targets: np.ndarray,
                baseline=None, n_coalitions: int = 256, seed: int = 0,
                ridge: float = 1e-4) -> np.ndarray:
    """KernelSHAP with a constrained weighted least-squares solve.

    The constraint sum(phi) = f(x) - f(baseline) is enforced through a KKT
    system, so the attributions satisfy efficiency exactly even when the
    sampled coalition set is small.  Coalitions always include the empty and
    full sets.
    """
    X = np.asarray(X, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    N, F = X.shape
    if len(targets) != N:
        raise ValueError("targets must have one entry per row")
    B = _as_baseline(baseline, X)
    if F == 1:
        base = np.asarray(score_fn(B))[np.arange(N), targets]
        full = np.asarray(score_fn(X))[np.arange(N), targets]
        return (full - base).reshape(N, 1)

    rng = np.random.default_rng(seed)
    masks = [np.zeros(F, dtype=bool), np.ones(F, dtype=bool)]
    for _ in range(max(0, int(n_coalitions))):
        s = int(rng.integers(1, F))
        idx = rng.choice(F, size=s, replace=False)
        m = np.zeros(F, dtype=bool)
        m[idx] = True
        masks.append(m)
    masks = np.unique(np.stack(masks, axis=0), axis=0).astype(np.float64)
    empty_idx = int(np.flatnonzero(masks.sum(axis=1) == 0)[0])
    full_idx = int(np.flatnonzero(masks.sum(axis=1) == F)[0])
    w = _coalition_weights(F, masks)
    sw = np.sqrt(w)
    Zm = masks * sw[:, None]
    A = Zm.T @ Zm + float(ridge) * np.eye(F)
    K = np.zeros((F + 1, F + 1), dtype=np.float64)
    K[:F, :F] = A
    K[:F, F] = 1.0
    K[F, :F] = 1.0

    attrs = np.zeros((N, F), dtype=np.float64)
    row_eye = np.arange(F)
    for i in range(N):
        Xmask = B[i] + (X[i] - B[i]) * masks
        scores = np.asarray(score_fn(Xmask))[:, int(targets[i])]
        y = scores - scores[empty_idx]
        rhs = np.empty(F + 1, dtype=np.float64)
        rhs[:F] = Zm.T @ (y * sw)
        rhs[F] = scores[full_idx] - scores[empty_idx]
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        attrs[i] = sol[row_eye]
    return attrs


def normalize_attributions(attr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L1-normalise each row; preserves sign and makes Avg.Rel. scale-free."""
    attr = np.asarray(attr, dtype=np.float64)
    denom = np.abs(attr).sum(axis=1, keepdims=True)
    return attr / np.maximum(denom, eps)


def topk_overlap(A: np.ndarray, B: np.ndarray, k: int = 5) -> float:
    """Top-k agreement (Top5Ratio in the base paper), averaged over rows."""
    A = np.asarray(A)
    B = np.asarray(B)
    k = int(min(k, A.shape[1]))
    if len(A) == 0:
        return 0.0
    top_a = np.argsort(-np.abs(A), axis=1)[:, :k]
    top_b = np.argsort(-np.abs(B), axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k
                          for a, b in zip(top_a, top_b)]))


def avg_relative(A: np.ndarray, B: np.ndarray) -> float:
    """Mean absolute difference between normalised explanations (Avg.Rel.)."""
    A = np.asarray(A)
    B = np.asarray(B)
    if len(A) == 0:
        return 0.0
    return float(np.mean(np.abs(A - B)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rank(A: np.ndarray, B: np.ndarray) -> float:
    """Mean per-row Spearman correlation between two attribution matrices."""
    A = np.asarray(A)
    B = np.asarray(B)
    if len(A) == 0:
        return 0.0
    vals = []
    for a, b in zip(A, B):
        ra, rb = _rankdata(a), _rankdata(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
        vals.append(float((ra * rb).sum() / denom) if denom > 0 else 0.0)
    return float(np.mean(vals))


def target_probabilities(score_fn, X: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logits = np.asarray(score_fn(X))
    probs = softmax(logits)
    return probs[np.arange(len(X)), np.asarray(targets, dtype=np.int64)]


def deletion_insertion_curves(score_fn, X: np.ndarray, targets: np.ndarray,
                              attr: np.ndarray, baseline=None,
                              fractions=None):
    """Return deletion/insertion probability curves for one attribution matrix.

    Features are ranked by |attribution|.  Deletion replaces the top-k features
    with the baseline; insertion starts from the baseline and restores the
    top-k features.  Averaged across samples.
    """
    X = np.asarray(X, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    B = _as_baseline(baseline, X)
    N, F = X.shape
    if fractions is None:
        fractions = np.linspace(0.0, 1.0, 11)
    fractions = np.asarray(fractions, dtype=np.float64)
    order = np.argsort(-np.abs(attr), axis=1)
    base_p = target_probabilities(score_fn, X, targets)
    del_curve, ins_curve = [], []
    for frac in fractions:
        k = int(round(float(frac) * F))
        Xdel = X.copy()
        Xins = B.copy()
        if k > 0:
            for i in range(N):
                idx = order[i, :k]
                Xdel[i, idx] = B[i, idx]
                Xins[i, idx] = X[i, idx]
        del_curve.append(float(target_probabilities(score_fn, Xdel, targets).mean()))
        ins_curve.append(float(target_probabilities(score_fn, Xins, targets).mean()))
    return {
        "fractions": fractions,
        "base_prob": float(base_p.mean()),
        "deletion": np.asarray(del_curve),
        "insertion": np.asarray(ins_curve),
    }


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float(y.mean()) if len(y) else 0.0
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def faithfulness_metrics(score_fn, X: np.ndarray, targets: np.ndarray,
                         attr: np.ndarray, baseline=None,
                         fractions=None) -> dict:
    curves = deletion_insertion_curves(
        score_fn, X, targets, attr, baseline=baseline, fractions=fractions)
    frac = curves["fractions"]
    base = curves["base_prob"]
    dele = curves["deletion"]
    ins = curves["insertion"]
    # Comprehensiveness = probability drop after removing the top 20%.
    # Sufficiency = probability retained after inserting only the top 20%.
    k20 = int(np.argmin(np.abs(frac - 0.2)))
    return {
        "deletion_auc": _auc(frac, dele),
        "insertion_auc": _auc(frac, ins),
        "comprehensiveness": float(base - dele[k20]),
        "sufficiency": float(ins[k20]),
        "base_prob": float(base),
    }


def completeness_errors(score_fn, X: np.ndarray, targets: np.ndarray,
                        attr: np.ndarray, baseline=None,
                        rtol_floor: float = 1e-8) -> dict:
    """Measure IG/SHAP efficiency: sum(attr) vs f(x)-f(baseline).

    Returns absolute and relative completeness errors; the relative error uses
    a small floor so near-zero score differences do not explode.
    """
    X = np.asarray(X, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    attr = np.asarray(attr, dtype=np.float64)
    B = _as_baseline(baseline, X)
    fx = np.asarray(score_fn(X))[np.arange(len(X)), targets]
    fb = np.asarray(score_fn(B))[np.arange(len(X)), targets]
    diff = float(0.0) + (fx - fb)
    err = attr.sum(axis=1) - diff
    abs_err = np.abs(err)
    rel_err = abs_err / np.maximum(np.abs(diff), float(rtol_floor))
    return {
        "mean_abs_error": float(abs_err.mean()) if len(err) else 0.0,
        "max_abs_error": float(abs_err.max()) if len(err) else 0.0,
        "mean_rel_error": float(rel_err.mean()) if len(err) else 0.0,
        "frac_within_1pct": float(np.mean(rel_err <= 0.01)) if len(err) else 0.0,
        "frac_within_5pct": float(np.mean(rel_err <= 0.05)) if len(err) else 0.0,
    }


def group_attribution_shares(attr: np.ndarray, groups: dict[str, list[int]]):
    """Aggregate signed attributions into group shares.

    Returns ``(abs_share, signed_share)`` arrays of shape ``(N, G)``.  The
    absolute share is the fraction of total |attribution| in each group; the
    signed share is the group's net signed contribution divided by the same
    total absolute attribution.
    """
    attr = np.asarray(attr, dtype=np.float64)
    total_abs = np.abs(attr).sum(axis=1, keepdims=True)
    total_abs = np.maximum(total_abs, 1e-12)
    names = list(groups)
    abs_share = np.column_stack([
        np.abs(attr[:, groups[name]]).sum(axis=1) for name in names
    ]) / total_abs
    signed_share = np.column_stack([
        attr[:, groups[name]].sum(axis=1) for name in names
    ]) / total_abs
    return abs_share, signed_share


def group_deletion_insertion(score_fn, X: np.ndarray, targets: np.ndarray,
                             groups: dict[str, list[int]], baseline=None):
    """Group-level deletion/insertion against the baseline.

    Deletion removes an entire group (sets its columns to the baseline);
    insertion restores an entire group starting from the baseline.

    ``deletion_drop`` is ``base_prob - deleted_prob``, where ``base_prob`` is
    the full-input probability.  ``insertion_gain`` is measured against the
    all-baseline probability, ``inserted_prob - baseline_prob``, so it is a
    genuine gain.  ``insertion_gap_vs_full`` retains the older
    ``inserted_prob - base_prob`` contrast; it is negative whenever the full
    input is more confident than a partial group restoration, so it is
    reported only as a supplementary diagnostic.
    """
    X = np.asarray(X, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    B = _as_baseline(baseline, X)
    base_prob = float(target_probabilities(score_fn, X, targets).mean())
    baseline_prob = float(target_probabilities(score_fn, B, targets).mean())
    rows = []
    for name, idx in groups.items():
        idx = np.asarray(idx, dtype=np.int64)
        Xdel = X.copy()
        Xdel[:, idx] = B[:, idx]
        Xins = B.copy()
        Xins[:, idx] = X[:, idx]
        del_prob = float(target_probabilities(score_fn, Xdel, targets).mean())
        ins_prob = float(target_probabilities(score_fn, Xins, targets).mean())
        rows.append(dict(
            group=name, n_features=len(idx),
            base_prob=base_prob,
            baseline_prob=baseline_prob,
            deleted_prob=del_prob,
            inserted_prob=ins_prob,
            deletion_drop=base_prob - del_prob,
            insertion_gain=ins_prob - baseline_prob,
            insertion_gap_vs_full=ins_prob - base_prob,
        ))
    return rows


def select_stratified_indices(y: np.ndarray, max_samples: int = 48,
                              seed: int = 0) -> np.ndarray:
    """Deterministic class-stratified sample index selection.

    The same indices are used for every CIL method on a given dataset/arch/seed
    so that cross-method explanation comparisons are not confounded by
    different test samples.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    max_samples = int(max_samples)
    if max_samples <= 0:
        return np.zeros(0, dtype=np.int64)
    per = max(1, max_samples // max(1, len(classes)))
    chosen = []
    for c in classes:
        idx = np.flatnonzero(y == c).copy()
        rng.shuffle(idx)
        chosen.extend(idx[:per].tolist())
    if len(chosen) < max_samples:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(chosen, dtype=np.int64))
        rng.shuffle(remaining)
        chosen.extend(remaining[:max_samples - len(chosen)].tolist())
    return np.asarray(sorted(chosen[:max_samples]), dtype=np.int64)
