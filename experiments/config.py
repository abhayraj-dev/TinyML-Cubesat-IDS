import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import CubeSatMLP, CubeSatGRU, CubeSat1DCNN
from tuning_grids import (VAL_FRAC, CALIB_FRAC, VAL_CAP_PER_CLASS,
                          CALIB_CAP_PER_CLASS, SELECT_CAP_PER_CLASS)

ARCHS = ["MLP", "GRU", "CNN1D"]
# Phase 1: lwf (distill-only), bic (replay+distill+BiC), icarl (herding+distill),
# joint_scratch (from-scratch offline joint-training reference; not a strict
# upper bound). `joint` kept as legacy
# cumulative warm-start for comparability with old CSVs.
METHODS = ["finetune", "replay", "ewc", "full", "lwf", "bic", "icarl", "joint", "joint_scratch"]
# 15 seeds: with 216 all-pairwise comparisons, the exact sign-flip floor is
# 2/2^15 = 6.1e-5, so a Holm-adjusted comparison can reach p<0.05 (10 seeds
# cannot: 2/2^10 * 216 = 0.42).  The count is fixed before the confirmatory run.
SEEDS = list(range(15))

MODEL_CLASSES = {"MLP": CubeSatMLP, "GRU": CubeSatGRU, "CNN1D": CubeSat1DCNN}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def validate_frozen_hparams(data):
    """Ensure every architecture/method has a non-empty frozen hparams entry."""
    problems = []
    for arch in ARCHS:
        arch_entry = data.get(arch) if isinstance(data, dict) else None
        if not isinstance(arch_entry, dict):
            problems.append(f"{arch}: missing architecture entry")
            continue
        for method in METHODS:
            entry = arch_entry.get(method)
            if not isinstance(entry, dict):
                problems.append(f"{arch}/{method}: missing method entry")
                continue
            hp = entry.get("hparams")
            if not isinstance(hp, dict) or not hp:
                problems.append(f"{arch}/{method}: missing non-empty hparams")
            elif "lr" not in hp or "epochs" not in hp:
                problems.append(f"{arch}/{method}: hparams missing lr/epochs")
    if problems:
        raise ValueError(
            "Incomplete frozen hyperparameter file: " + "; ".join(problems)
            + ". Re-run `python3 experiments/run_validation_tune.py`."
        )
    return data


def load_frozen_hparams_strict():
    """Load per-method/arch frozen hyperparameters, failing loudly if absent.

    Every experiment downstream of validation tuning must call this instead of
    falling back to legacy or default configs: a missing or incomplete frozen
    file means the pipeline ran out of order (or tuning failed), and silently
    training on wrong hyperparameters would produce a valid-looking but
    invalid matrix.
    """
    import json
    path = os.path.join(RESULTS_DIR, "best_hparams_per_method_arch.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Frozen hyperparameters not found at {path}. "
            "Run `python3 experiments/run_validation_tune.py` first "
            "(via `python3 run_all.py`), then re-run this step. "
            "Refusing to fall back to legacy/default configs."
        )
    with open(path) as f:
        return validate_frozen_hparams(json.load(f))

BASE_TRAIN_HPARAMS = dict(epochs=25, batch_size=32, fisher_n_samples=200)

# LEGACY only: per-arch overrides applied on top of BASE_TRAIN_HPARAMS +
# coarse STIN sweep winners. Superseded by frozen per-method x
# per-architecture validation configs
# (results/best_hparams_per_method_arch.json), which ALREADY include
# per-arch CNN tuning (CNN1D replay/icarl lr=0.01). No live pipeline code
# calls hparams_for() anymore; kept so dormant scripts keep importing.
ARCH_TUNED_HPARAMS = {
    "CNN1D": dict(lr=0.01, epochs=50),
}


def hparams_for(arch: str, best_hp: dict) -> dict:
    """Legacy fallback merge: BASE_TRAIN_HPARAMS < sweep winners < per-arch
    overrides. Superseded by frozen per-method/arch configs; kept only so the
    legacy fallback path in run_main.py keeps working."""
    hp = {**BASE_TRAIN_HPARAMS, **best_hp}
    hp.update(ARCH_TUNED_HPARAMS.get(arch, {}))
    return hp


def model_kwargs_for(arch: str, n_features: int) -> dict:
    if arch == "MLP":
        return dict(input_dim=n_features, h1=32, h2=16)
    if arch == "GRU":
        divisors = [d for d in range(2, n_features) if n_features % d == 0]
        seq_len = min(divisors, key=lambda d: abs(d - 10)) if divisors else n_features
        return dict(input_dim=n_features, seq_len=seq_len, hidden=10)
    if arch == "CNN1D":
        return dict(input_dim=n_features, c1=6, c2=10, kernel=3)
    raise ValueError(arch)
