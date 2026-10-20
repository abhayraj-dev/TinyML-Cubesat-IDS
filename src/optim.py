"""Adam optimizer operating on a dict[str, np.ndarray] of parameters.
Handles growing parameters (the shared classification head widens as new
classes are learned) by zero-padding existing momentum/variance state to
the new shape -- new columns start with fresh (zero) moments while
existing columns keep their accumulated momentum."""
import numpy as np


def _pad_state_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    padded = np.zeros(target_shape, dtype=arr.dtype)
    slices = tuple(slice(0, s) for s in arr.shape)
    padded[slices] = arr
    return padded


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = {}

    def step(self, params: dict, grads: dict):
        for k, g in grads.items():
            if g is None:
                continue
            p = params[k]
            if self.wd > 0:
                g = g + self.wd * p
            if k not in self.m:
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)
                self.t[k] = 0
            elif self.m[k].shape != p.shape:
                # parameter grew (new head columns) -- preserve existing
                # momentum for old entries, zero-init the new ones
                self.m[k] = _pad_state_to_shape(self.m[k], p.shape)
                self.v[k] = _pad_state_to_shape(self.v[k], p.shape)
            self.t[k] += 1
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t[k])
            vhat = self.v[k] / (1 - self.b2 ** self.t[k])
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
