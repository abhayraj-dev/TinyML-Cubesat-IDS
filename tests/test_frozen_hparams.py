"""Frozen hyperparameters must cover every architecture/method entry."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from config import (ARCHS, METHODS, load_frozen_hparams_strict,
                    validate_frozen_hparams)


def main():
    frozen = load_frozen_hparams_strict()
    for arch in ARCHS:
        for method in METHODS:
            hp = frozen[arch][method]["hparams"]
            assert hp, (arch, method)
            assert "lr" in hp and "epochs" in hp, (arch, method, hp)

    incomplete = {arch: {} for arch in ARCHS}
    try:
        validate_frozen_hparams(incomplete)
    except ValueError as exc:
        assert "missing method entry" in str(exc)
    else:
        raise AssertionError("incomplete frozen config must fail validation")

    print("test_frozen_hparams: OK")


if __name__ == "__main__":
    main()
