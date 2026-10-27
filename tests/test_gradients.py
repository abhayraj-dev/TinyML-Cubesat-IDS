"""Numerical gradient checks for CubeSatMLP / CubeSatGRU / CubeSat1DCNN.

Run: python3 tests/test_gradients.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from models import CubeSatMLP, CubeSatGRU, CubeSat1DCNN
from losses import softmax_cross_entropy
from train import _distill_dlogits
from cmmu import OnlineEWC


def _to_float64(model):
    for k in list(model.params.keys()):
        model.params[k] = model.params[k].astype(np.float64)


def numeric_grad_check(model, X, y, eps=1e-5, n_check=8, tol=1e-2):
    _to_float64(model)
    X = X.astype(np.float64)

    logits, cache = model.forward(X)
    loss, dlogits, _ = softmax_cross_entropy(logits, y)
    grads = model.backward(cache, dlogits)

    rng = np.random.default_rng(0)
    worst = 0.0
    n_fail = 0
    n_total = 0
    for name, garr in grads.items():
        p = model.params[name]
        flat_idx = rng.choice(p.size, size=min(n_check, p.size), replace=False)
        for fi in flat_idx:
            orig = p.flat[fi]
            p.flat[fi] = orig + eps
            logits_p, _ = model.forward(X)
            loss_p, _, _ = softmax_cross_entropy(logits_p, y)
            p.flat[fi] = orig - eps
            logits_m, _ = model.forward(X)
            loss_m, _, _ = softmax_cross_entropy(logits_m, y)
            p.flat[fi] = orig
            numeric = (loss_p - loss_m) / (2 * eps)
            analytic = garr.flat[fi]
            # combined absolute+relative criterion (standard practice; avoids
            # blow-up near zero-crossings / ReLU kinks that pure relative
            # error is oversensitive to)
            err = abs(numeric - analytic) / max(1.0, abs(numeric) + abs(analytic))
            n_total += 1
            worst = max(worst, err)
            if err >= tol:
                n_fail += 1
                print(f"  [FAIL] {name}[{fi}] numeric={numeric:.6f} analytic={analytic:.6f} err={err:.5f}")
    print(f"  -> worst combined error: {worst:.6f} over {n_total} checked entries ({n_fail} failing)")
    assert worst < tol, "Gradient check failed"


def input_grad_check(model, X, targets, eps=1e-5, n_check=8, tol=1e-2):
    """Finite-difference check of d(sum target logits)/dX."""
    _to_float64(model)
    X = X.astype(np.float64)
    logits, cache = model.forward(X)
    dlogits = np.zeros_like(logits)
    dlogits[np.arange(len(X)), targets] = 1.0
    analytic = model.backward_input(cache, dlogits)

    rng = np.random.default_rng(1)
    worst = 0.0
    for fi in rng.choice(X.size, size=min(n_check, X.size), replace=False):
        idx = np.unravel_index(fi, X.shape)
        orig = X[idx]
        X[idx] = orig + eps
        lp = model.forward(X)[0][np.arange(len(X)), targets].sum()
        X[idx] = orig - eps
        lm = model.forward(X)[0][np.arange(len(X)), targets].sum()
        X[idx] = orig
        numeric = (lp - lm) / (2 * eps)
        err = abs(numeric - analytic[idx]) / max(1.0, abs(numeric) + abs(analytic[idx]))
        worst = max(worst, err)
    print(f"  -> input-grad worst error: {worst:.6f}")
    assert worst < tol, "Input-gradient check failed"


def check_distill_gradient(seed=3):
    """Finite-difference check of the KD gradient (old-class columns only)."""
    rng = np.random.default_rng(seed)
    student = rng.standard_normal((6, 5)).astype(np.float64)
    teacher = rng.standard_normal((6, 3)).astype(np.float64)
    T, lam = 2.0, 0.7

    def kd_loss(z):
        p_t = np.exp(teacher / T); p_t /= p_t.sum(1, keepdims=True)
        zz = z[:, :3] / T
        p_s = np.exp(zz - zz.max(1, keepdims=True))
        p_s /= p_s.sum(1, keepdims=True)
        return lam * T * T * np.sum(p_t * (np.log(np.clip(p_t, 1e-12, None)) -
                                          np.log(np.clip(p_s, 1e-12, None)))) / len(z)

    analytic = _distill_dlogits(student, teacher, T, lam)
    eps = 1e-6
    worst = 0.0
    for idx in [(0, 0), (1, 1), (2, 2), (0, 4), (5, 3)]:
        z = student.copy()
        z[idx] += eps
        lp = kd_loss(z)
        z[idx] -= 2 * eps
        lm = kd_loss(z)
        num = (lp - lm) / (2 * eps)
        err = abs(num - analytic[idx]) / max(1.0, abs(num) + abs(analytic[idx]))
        worst = max(worst, err)
    assert worst < 1e-5, f"KD gradient error {worst}"
    print(f"  -> KD gradient worst error: {worst:.2e}")


def check_ewc_penalty_gradient(seed=4):
    """Finite-difference check of the EWC quadratic penalty gradient."""
    rng = np.random.default_rng(seed)
    m = CubeSatMLP(input_dim=7, h1=5, h2=4, seed=seed)
    m.expand_head(3)
    ewc = OnlineEWC(lam=3.0, gamma=1.0)
    ewc.fisher = {k: rng.random(v.shape) for k, v in m.params.items()}
    ewc.anchor = {k: v.copy() for k, v in m.params.items()}
    ewc._consolidated = True
    grads = ewc.penalty_grad(m.params)
    eps = 1e-6
    worst = 0.0
    for name in list(m.params)[:3]:
        arr = m.params[name]
        idx = tuple(0 for _ in arr.shape)
        orig = arr[idx]
        arr[idx] = orig + eps
        lp = ewc.penalty_loss(m.params)
        arr[idx] = orig - eps
        lm = ewc.penalty_loss(m.params)
        arr[idx] = orig
        num = (lp - lm) / (2 * eps)
        ana = grads[name][idx]
        err = abs(num - ana) / max(1.0, abs(num) + abs(ana))
        worst = max(worst, err)
    assert worst < 1e-5, f"EWC penalty gradient error {worst}"
    print(f"  -> EWC penalty gradient worst error: {worst:.2e}")


def main():
    rng = np.random.default_rng(42)
    N, F = 9, 12
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = rng.integers(0, 3, size=N)

    print("== CubeSatMLP ==")
    m = CubeSatMLP(input_dim=F, h1=10, h2=7, seed=1)
    m.expand_head(3)
    numeric_grad_check(m, X, y)

    print("== CubeSatGRU ==")
    m = CubeSatGRU(input_dim=F, seq_len=4, hidden=6, seed=1)  # 12 = 4*3
    m.expand_head(3)
    numeric_grad_check(m, X, y)

    print("== CubeSat1DCNN ==")
    m = CubeSat1DCNN(input_dim=F, c1=4, c2=5, kernel=3, seed=1)
    m.expand_head(3)
    numeric_grad_check(m, X, y)

    print("== Input-gradient checks (for Integrated Gradients) ==")
    tg = rng.integers(0, 3, size=N)
    m = CubeSatMLP(input_dim=F, h1=10, h2=7, seed=1); m.expand_head(3)
    input_grad_check(m, X, tg)
    m = CubeSatGRU(input_dim=F, seq_len=4, hidden=6, seed=1); m.expand_head(3)
    input_grad_check(m, X, tg)
    m = CubeSat1DCNN(input_dim=F, c1=4, c2=5, kernel=3, seed=1); m.expand_head(3)
    input_grad_check(m, X, tg)

    print("== Growing head (class-incremental) gradient check ==")
    m = CubeSatMLP(input_dim=F, h1=10, h2=7, seed=1)
    m.expand_head(2)          # task 0: 2 classes
    y2 = rng.integers(0, 2, size=N)
    numeric_grad_check(m, X, y2)
    m.expand_head(2)          # task 1: 2 more classes appended (now 4 total)
    y4 = rng.integers(0, 4, size=N)
    numeric_grad_check(m, X, y4)

    print("\n== KD gradient ==")
    check_distill_gradient()

    print("\n== EWC penalty gradient ==")
    check_ewc_penalty_gradient()

    print("\nALL GRADIENT CHECKS PASSED")


if __name__ == "__main__":
    main()
