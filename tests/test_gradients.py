"""Numerical gradient checks for CubeSatMLP / CubeSatGRU / CubeSat1DCNN.

Run: python3 tests/test_gradients.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from models import CubeSatMLP, CubeSatGRU, CubeSat1DCNN
from losses import softmax_cross_entropy


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

    print("== Growing head (class-incremental) gradient check ==")
    m = CubeSatMLP(input_dim=F, h1=10, h2=7, seed=1)
    m.expand_head(2)          # task 0: 2 classes
    y2 = rng.integers(0, 2, size=N)
    numeric_grad_check(m, X, y2)
    m.expand_head(2)          # task 1: 2 more classes appended (now 4 total)
    y4 = rng.integers(0, 4, size=N)
    numeric_grad_check(m, X, y4)

    print("\nALL GRADIENT CHECKS PASSED")


if __name__ == "__main__":
    main()
