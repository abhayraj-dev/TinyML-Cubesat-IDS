import numpy as np


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    """logits: (N,C) ; y: (N,) int labels. Returns (loss_scalar_mean, dlogits, probs)."""
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    logp = -np.log(np.clip(probs[np.arange(N), y], 1e-12, None))
    loss = logp.mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N
    return loss, dlogits, probs
