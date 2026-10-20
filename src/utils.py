"""Small shared utilities: seeding and timing."""
import numpy as np
import time
import random


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


class Timer:
    def __init__(self):
        self.t0 = None
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
