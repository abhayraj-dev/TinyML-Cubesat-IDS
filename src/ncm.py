"""Nearest-Class-Mean (NCM) inference for the iCaRL family.

The softmax head used by the CIL loop is trained with cross-entropy and is
therefore sensitive to the well-known new-class / magnitude bias.  iCaRL's
original classifier is instead a nearest-class-mean rule over L2-normalised
penultimate features.  This module implements that rule as a *post-hoc*
evaluation head: it does not add parameters to the trained network and it does
not change the training objective.

Two details matter for a faithful, non-leaky implementation:

1.  Class means are built only from the rehearsal buffer (the exemplars that
    would actually be stored on the satellite).  Test rows are never used to
    fit the classifier.
2.  Features are L2-normalised per sample, then averaged per class, and the
    resulting mean is L2-normalised again before cosine matching.  This is the
    standard NCM construction and avoids a class mean whose norm depends on
    the number of stored exemplars.

The cosine scores are multiplied by a temperature only to produce a
softmax-compatible "logit" matrix for the existing detection and per-class
metric helpers; argmax predictions are unchanged by this monotone scaling.
"""
from __future__ import annotations

import numpy as np


class NCMClassifier:
    def __init__(self, n_classes: int = 0, temperature: float = 10.0):
        self.n_classes = int(n_classes)
        self.temperature = float(temperature)
        self.class_means = None
        self.class_counts = None
        self.fitted = False

    @staticmethod
    def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, eps)

    @staticmethod
    def _embed(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        X = np.asarray(X)
        if len(X) == 0:
            return np.zeros((0, model.trunk_dim), dtype=np.float32)
        chunks = []
        for start in range(0, len(X), int(batch_size)):
            feat, _ = model.forward_trunk(X[start:start + int(batch_size)])
            chunks.append(np.asarray(feat))
        return np.concatenate(chunks, axis=0)

    def fit_from_arrays(self, model, X: np.ndarray, y: np.ndarray,
                        n_classes: int | None = None) -> "NCMClassifier":
        """Fit class means from labelled exemplars in the model's feature space."""
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        if n_classes is None:
            n_classes = int(y.max()) + 1 if len(y) else 0
        self.n_classes = int(n_classes)
        feat = self._embed(model, X) if len(X) else np.zeros((0, model.trunk_dim))
        means = np.zeros((self.n_classes, model.trunk_dim), dtype=np.float64)
        counts = np.zeros(self.n_classes, dtype=np.int64)
        if len(feat):
            feat = self._l2_normalize(feat)
            for c in range(self.n_classes):
                mask = (y == c)
                if not np.any(mask):
                    continue
                mu = feat[mask].mean(axis=0)
                means[c] = mu / max(float(np.linalg.norm(mu)), 1e-12)
                counts[c] = int(mask.sum())
        self.class_means = means.astype(np.float32)
        self.class_counts = counts
        self.fitted = True
        return self

    def fit_from_buffer(self, model, buffer,
                        n_classes: int | None = None) -> "NCMClassifier":
        if buffer is None or buffer.size == 0:
            raise ValueError("NCM requires a non-empty rehearsal buffer")
        return self.fit_from_arrays(
            model, buffer.X[:buffer.size], buffer.y[:buffer.size],
            n_classes=n_classes)

    def logits(self, model, X: np.ndarray) -> np.ndarray:
        if not self.fitted or self.class_means is None:
            raise RuntimeError("NCMClassifier must be fitted before inference")
        X = np.asarray(X)
        if len(X) == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        feat = self._l2_normalize(self._embed(model, X))
        return (feat @ self.class_means.T * self.temperature).astype(np.float32)

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        return self.logits(model, X).argmax(axis=1)

    def input_grad(self, model, X: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Gradient of the selected NCM score w.r.t. the input (for IG)."""
        if not self.fitted or self.class_means is None:
            raise RuntimeError("NCMClassifier must be fitted before gradients")
        X = np.asarray(X)
        targets = np.asarray(targets, dtype=np.int64)
        feat, cache = model.forward_trunk(X)
        feat = np.asarray(feat, dtype=np.float64)
        norms = np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
        u = feat / norms
        means = self.class_means.astype(np.float64)
        cos = u @ means.T
        c = cos[np.arange(len(X)), targets]
        dfeat = self.temperature * (means[targets] - c[:, None] * u) / norms
        return model.backward_trunk_input(cache, dfeat)

    def memory_bytes(self, bytes_per_val: int = 2) -> int:
        """Deployed storage: class means (FP16) + optional class counts."""
        if self.class_means is None:
            return 0
        means = int(self.class_means.size * int(bytes_per_val))
        counts = int(self.class_counts.size * 4) if self.class_counts is not None else 0
        return means + counts
