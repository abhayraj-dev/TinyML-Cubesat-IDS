"""Shared evaluation-time inference helpers.

The same final predictor must be used by accuracy, detection, per-class and XAI
code, otherwise the NCM / BiC / softmax variants can silently disagree.
"""
from __future__ import annotations

import numpy as np


def eval_logits(model, X: np.ndarray, corrector=None, ncm=None) -> np.ndarray:
    """Return class logits for the deployed predictor.

    Priority: NCM (if supplied) > BiC correction (if fitted) > raw softmax head.
    NCM and BiC are mutually exclusive in practice (iCaRL does not use BiC).
    """
    if ncm is not None:
        return ncm.logits(model, X)
    logits, _ = model.forward(X)
    if corrector is not None and getattr(corrector, "fitted", False):
        logits = corrector.apply(logits)
    return logits


def concatenate_test(task_seq):
    """Concatenate every task's held-out test split with global class ids."""
    Xs, ys, offset = [], [], 0
    for task in task_seq.tasks:
        Xs.append(task.X_test)
        ys.append(task.y_test + offset)
        offset += task.n_new_classes
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
