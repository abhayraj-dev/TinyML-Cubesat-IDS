"""
From-scratch (pure NumPy) implementations of three TinyML-sized backbones,
each topped with a single, dynamically-growing shared classification head
(class-incremental protocol: no task ID is available at train or test time;
every sample is classified against the full set of classes learned so far
through one shared softmax layer). When a new task introduces classes that
have never been seen before, new output columns are appended to the shared
head (existing columns/weights are preserved untouched).

No autograd framework is used -- every forward/backward pass is derived and
implemented by hand, mirroring the kind of fixed-point / fixed-shape
inference code that would eventually be hand-ported to C for an MCU
(CMSIS-NN style). This also keeps the whole pipeline dependency-free (no
PyTorch/TensorFlow available or needed).

All three models expose the same interface:
    forward(X)                 -> logits, cache      (logits over ALL classes seen so far)
    backward(cache, dlogits)   -> grads (dict[str, np.ndarray])
    expand_head(n_new_classes) -> grows the shared output layer
    trunk_dim
    param_count()
"""
from __future__ import annotations
import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class _BaseModel:
    """Shared, dynamically-growing single classification head."""

    def __init__(self, trunk_dim: int, seed: int = 0, head_scale: float = 0.1):
        self.trunk_dim = trunk_dim
        self.params: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)
        self._head_scale = head_scale

    # ---- shared growing head ---------------------------------------------
    @property
    def n_classes_total(self) -> int:
        return self.params["head.W"].shape[1] if "head.W" in self.params else 0

    def expand_head(self, n_new_classes: int):
        """Append n_new_classes new output columns to the shared head,
        preserving all existing columns/weights untouched."""
        if n_new_classes <= 0:
            return
        new_W = (self._rng.standard_normal((self.trunk_dim, n_new_classes)) * self._head_scale).astype(np.float32)
        new_b = np.zeros(n_new_classes, dtype=np.float32)
        if "head.W" not in self.params:
            self.params["head.W"] = new_W
            self.params["head.b"] = new_b
        else:
            self.params["head.W"] = np.concatenate([self.params["head.W"], new_W], axis=1)
            self.params["head.b"] = np.concatenate([self.params["head.b"], new_b], axis=0)

    def forward(self, X: np.ndarray):
        trunk_out, cache_trunk = self.forward_trunk(X)
        logits = trunk_out @ self.params["head.W"] + self.params["head.b"]
        cache = (cache_trunk, trunk_out)
        return logits, cache

    def backward(self, cache, dlogits: np.ndarray):
        cache_trunk, trunk_out = cache
        grads = {}
        grads["head.W"] = trunk_out.T @ dlogits
        grads["head.b"] = dlogits.sum(axis=0)
        dtrunk = dlogits @ self.params["head.W"].T
        grads.update(self.backward_trunk(cache_trunk, dtrunk))
        return grads

    def backward_input(self, cache, dlogits: np.ndarray):
        """Gradient of the selected logits w.r.t. the model input.

        Used by Integrated Gradients.  It follows the same chain rule as
        ``backward`` but stops at the input instead of accumulating parameter
        gradients.
        """
        cache_trunk, _ = cache
        dtrunk = dlogits @ self.params["head.W"].T
        return self.backward_trunk_input(cache_trunk, dtrunk)

    def param_count(self) -> int:
        return sum(v.size for v in self.params.values())

    # to be implemented by subclasses
    def forward_trunk(self, X):
        raise NotImplementedError

    def backward_trunk(self, cache, dtrunk):
        raise NotImplementedError

    def backward_trunk_input(self, cache, dtrunk):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. MLP
# --------------------------------------------------------------------------- #
class CubeSatMLP(_BaseModel):
    NAME = "MLP"

    def __init__(self, input_dim: int, h1: int = 32, h2: int = 16, seed: int = 0):
        super().__init__(trunk_dim=h2, seed=seed)
        rng = self._rng
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / h1)
        self.params["W1"] = (rng.standard_normal((input_dim, h1)) * s1).astype(np.float32)
        self.params["b1"] = np.zeros(h1, dtype=np.float32)
        self.params["W2"] = (rng.standard_normal((h1, h2)) * s2).astype(np.float32)
        self.params["b2"] = np.zeros(h2, dtype=np.float32)

    def forward_trunk(self, X):
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = relu(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = relu(z2)
        return a2, (X, z1, a1, z2, a2)

    def backward_trunk(self, cache, da2):
        X, z1, a1, z2, a2 = cache
        dz2 = da2 * relu_grad(z2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def backward_trunk_input(self, cache, da2):
        X, z1, a1, z2, a2 = cache
        dz2 = da2 * relu_grad(z2)
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * relu_grad(z1)
        return dz1 @ self.params["W1"].T


# --------------------------------------------------------------------------- #
# 2. GRU
# --------------------------------------------------------------------------- #
class CubeSatGRU(_BaseModel):
    NAME = "GRU"

    def __init__(self, input_dim: int, seq_len: int, hidden: int = 10, seed: int = 0):
        assert input_dim % seq_len == 0, "input_dim must be divisible by seq_len"
        self.seq_len = seq_len
        self.feat_per_step = input_dim // seq_len
        self.hidden = hidden
        super().__init__(trunk_dim=hidden, seed=seed)
        rng = self._rng
        F, H = self.feat_per_step, hidden
        sx = np.sqrt(1.0 / F)
        sh = np.sqrt(1.0 / H)
        for gate in ("z", "r", "n"):
            self.params[f"W{gate}"] = (rng.standard_normal((F, H)) * sx).astype(np.float32)
            self.params[f"U{gate}"] = (rng.standard_normal((H, H)) * sh).astype(np.float32)
            self.params[f"b{gate}"] = np.zeros(H, dtype=np.float32)

    def forward_trunk(self, X):
        N = X.shape[0]
        Xseq = X.reshape(N, self.seq_len, self.feat_per_step)
        H = self.hidden
        h = np.zeros((N, H), dtype=np.float32)
        steps = []
        for t in range(self.seq_len):
            xt = Xseq[:, t, :]
            h_prev = h
            zt_pre = xt @ self.params["Wz"] + h_prev @ self.params["Uz"] + self.params["bz"]
            rt_pre = xt @ self.params["Wr"] + h_prev @ self.params["Ur"] + self.params["br"]
            z = sigmoid(zt_pre)
            r = sigmoid(rt_pre)
            u = h_prev @ self.params["Un"]
            nt_pre = xt @ self.params["Wn"] + r * u + self.params["bn"]
            n = np.tanh(nt_pre)
            h = (1 - z) * n + z * h_prev
            steps.append((xt, h_prev, z, r, n, u, h))
        return h, steps

    def backward_trunk(self, steps, dh_last):
        H = self.hidden
        grads = {k: np.zeros_like(v) for k, v in self.params.items() if k in
                 ("Wz", "Uz", "bz", "Wr", "Ur", "br", "Wn", "Un", "bn")}
        dh = dh_last
        for t in reversed(range(self.seq_len)):
            xt, h_prev, z, r, n, u, h = steps[t]

            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dh_prev = dh * z  # path 1: h = z * h_prev (+...)

            dnt_pre = dn * (1 - n ** 2)
            grads["bn"] += dnt_pre.sum(axis=0)
            grads["Wn"] += xt.T @ dnt_pre
            dru = dnt_pre  # nt_pre = x@Wn + r*u + bn
            dr = dru * u
            du = dru * r
            grads["Un"] += h_prev.T @ du
            dh_prev += du @ self.params["Un"].T

            drt_pre = dr * r * (1 - r)
            grads["br"] += drt_pre.sum(axis=0)
            grads["Wr"] += xt.T @ drt_pre
            dh_prev += drt_pre @ self.params["Ur"].T
            grads["Ur"] += h_prev.T @ drt_pre

            dzt_pre = dz * z * (1 - z)
            grads["bz"] += dzt_pre.sum(axis=0)
            grads["Wz"] += xt.T @ dzt_pre
            dh_prev += dzt_pre @ self.params["Uz"].T
            grads["Uz"] += h_prev.T @ dzt_pre

            dh = dh_prev  # becomes gradient at previous timestep's h
        return grads

    def backward_trunk_input(self, steps, dh_last):
        N = dh_last.shape[0]
        dXseq = np.zeros((N, self.seq_len, self.feat_per_step), dtype=dh_last.dtype)
        dh = dh_last
        for t in reversed(range(self.seq_len)):
            xt, h_prev, z, r, n, u, h = steps[t]

            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dh_prev = dh * z

            dnt_pre = dn * (1 - n ** 2)
            dxt = dnt_pre @ self.params["Wn"].T
            dru = dnt_pre
            dr = dru * u
            du = dru * r
            dh_prev += du @ self.params["Un"].T

            drt_pre = dr * r * (1 - r)
            dxt += drt_pre @ self.params["Wr"].T
            dh_prev += drt_pre @ self.params["Ur"].T

            dzt_pre = dz * z * (1 - z)
            dxt += dzt_pre @ self.params["Wz"].T
            dh_prev += dzt_pre @ self.params["Uz"].T

            dXseq[:, t, :] = dxt
            dh = dh_prev
        return dXseq.reshape(N, self.seq_len * self.feat_per_step)


# --------------------------------------------------------------------------- #
# 3. 1-D CNN
# --------------------------------------------------------------------------- #
class CubeSat1DCNN(_BaseModel):
    NAME = "CNN1D"

    def __init__(self, input_dim: int, c1: int = 6, c2: int = 10, kernel: int = 3, seed: int = 0):
        self.L = input_dim
        self.c1, self.c2, self.k = c1, c2, kernel
        self.pad = kernel // 2
        super().__init__(trunk_dim=c2, seed=seed)
        rng = self._rng
        s1 = np.sqrt(2.0 / (1 * kernel))
        s2 = np.sqrt(2.0 / (c1 * kernel))
        self.params["W1"] = (rng.standard_normal((c1, 1, kernel)) * s1).astype(np.float32)
        self.params["b1"] = np.zeros(c1, dtype=np.float32)
        self.params["W2"] = (rng.standard_normal((c2, c1, kernel)) * s2).astype(np.float32)
        self.params["b2"] = np.zeros(c2, dtype=np.float32)

    @staticmethod
    def _conv1d_forward(X, W, b, pad):
        # X: (N, Cin, L) ; W: (Cout, Cin, K) ; returns (N, Cout, L)
        N, Cin, L = X.shape
        Cout, _, K = W.shape
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad)))
        Wflat = W.reshape(Cout, -1)  # (Cout, Cin*K)
        out = np.empty((N, Cout, L), dtype=X.dtype)
        windows = np.empty((L, N, Cin * K), dtype=X.dtype)
        for i in range(L):
            win = Xp[:, :, i:i + K].reshape(N, -1)
            windows[i] = win
            out[:, :, i] = win @ Wflat.T + b
        return out, (Xp, windows, W.shape, pad)

    def forward_trunk(self, X):
        N = X.shape[0]
        X1 = X.reshape(N, 1, self.L)
        c1_out, cache1 = self._conv1d_forward(X1, self.params["W1"], self.params["b1"], self.pad)
        a1 = relu(c1_out)
        c2_out, cache2 = self._conv1d_forward(a1, self.params["W2"], self.params["b2"], self.pad)
        a2 = relu(c2_out)
        trunk = a2.mean(axis=2)  # global average pool -> (N, c2)
        cache = (cache1, c1_out, a1, cache2, c2_out, a2, N)
        return trunk, cache

    def backward_trunk(self, cache, dtrunk):
        cache1, c1_out, a1, cache2, c2_out, a2, N = cache
        L = self.L
        # undo global avg pool
        da2 = np.repeat((dtrunk / L)[:, :, None], L, axis=2)
        dc2 = da2 * relu_grad(c2_out)

        # conv2 backward (manual, correct version using stored weight)
        Xp2, windows2, Wshape2, pad2 = cache2
        Cout2, Cin2, K2 = Wshape2
        W2flat = self.params["W2"].reshape(Cout2, -1)
        dW2 = np.zeros_like(W2flat)
        db2 = np.zeros(Cout2, dtype=dc2.dtype)
        dXp2 = np.zeros_like(Xp2)
        for i in range(L):
            dout_i = dc2[:, :, i]
            win = windows2[i]
            dW2 += dout_i.T @ win
            db2 += dout_i.sum(axis=0)
            dwin = dout_i @ W2flat
            dXp2[:, :, i:i + K2] += dwin.reshape(-1, Cin2, K2)
        da1_from_conv2 = dXp2[:, :, pad2:pad2 + a1.shape[2]]
        dc1 = da1_from_conv2 * relu_grad(c1_out)

        # conv1 backward
        Xp1, windows1, Wshape1, pad1 = cache1
        Cout1, Cin1, K1 = Wshape1
        W1flat = self.params["W1"].reshape(Cout1, -1)
        dW1 = np.zeros_like(W1flat)
        db1 = np.zeros(Cout1, dtype=dc1.dtype)
        for i in range(L):
            dout_i = dc1[:, :, i]
            win = windows1[i]
            dW1 += dout_i.T @ win
            db1 += dout_i.sum(axis=0)
            # dX not needed further (input layer)

        return {
            "W1": dW1.reshape(Wshape1), "b1": db1,
            "W2": dW2.reshape(Wshape2), "b2": db2,
        }

    def backward_trunk_input(self, cache, dtrunk):
        cache1, c1_out, a1, cache2, c2_out, a2, N = cache
        L = self.L

        da2 = np.repeat((dtrunk / L)[:, :, None], L, axis=2)
        dc2 = da2 * relu_grad(c2_out)

        Xp2, windows2, Wshape2, pad2 = cache2
        Cout2, Cin2, K2 = Wshape2
        W2flat = self.params["W2"].reshape(Cout2, -1)
        dXp2 = np.zeros_like(Xp2)
        for i in range(L):
            dout_i = dc2[:, :, i]
            dwin = dout_i @ W2flat
            dXp2[:, :, i:i + K2] += dwin.reshape(-1, Cin2, K2)
        da1 = dXp2[:, :, pad2:pad2 + L]
        dc1 = da1 * relu_grad(c1_out)

        Xp1, windows1, Wshape1, pad1 = cache1
        Cout1, Cin1, K1 = Wshape1
        W1flat = self.params["W1"].reshape(Cout1, -1)
        dXp1 = np.zeros_like(Xp1)
        for i in range(L):
            dout_i = dc1[:, :, i]
            dwin = dout_i @ W1flat
            dXp1[:, :, i:i + K1] += dwin.reshape(-1, Cin1, K1)
        dX1 = dXp1[:, :, pad1:pad1 + L]
        return dX1.reshape(N, L)


MODEL_REGISTRY = {
    "MLP": CubeSatMLP,
    "GRU": CubeSatGRU,
    "CNN1D": CubeSat1DCNN,
}
