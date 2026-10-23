"""Single source of truth for validation-tuning grids.

Both `run_validation_tune.py` (execution) and `create_manifest.py`
(documentation/counts) import this module. Do not duplicate these values in
prose or scripts: the manifest previously advertised replay capacities
[128, 256, 512] while the executable grid only ran [128, 512].
"""

TUNING_SEEDS = [0, 1, 2]
# Confirmatory protocol: 20% of each per-class pool is carved from the training
# data, then split into two DISJOINT halves:
#   * calibration half -> BiC fitting and val-balanced EWC Fisher
#   * selection half   -> hyperparameter selection (never used to fit a model)
# This keeps test locked and removes the previous calibration/selection reuse.
VAL_FRAC = 0.20
CALIB_FRAC = 0.50
VAL_CAP_PER_CLASS = 300  # 150 calibration + 150 selection per class
CALIB_CAP_PER_CLASS = int(round(VAL_CAP_PER_CLASS * CALIB_FRAC))
SELECT_CAP_PER_CLASS = VAL_CAP_PER_CLASS - CALIB_CAP_PER_CLASS

# `full` uses a reduced multi-axis grid so the added gamma/epochs dimensions
# stay tractable. The 25/50-epoch choice is tuned for every method.
TUNING_GRIDS = {
    "finetune": {
        "lr": [1e-3, 3e-3, 1e-2],
        "epochs": [25, 50],
    },
    "replay": {
        "lr": [1e-3, 3e-3, 1e-2],
        "buffer_capacity": [128, 512],
        "replay_frac": [0.5, 0.9],
        "epochs": [25, 50],
    },
    "ewc": {
        "lr": [1e-3, 3e-3, 1e-2],
        "ewc_lambda": [100, 1000, 5000],
        "ewc_gamma": [0.5, 1.0, 2.0],
        "epochs": [25, 50],
    },
    "full": {
        "lr": [3e-3, 1e-2],
        "buffer_capacity": [256, 512],
        "replay_frac": [0.7],
        "ewc_lambda": [100, 1000],
        "ewc_gamma": [0.5, 1.0, 2.0],
        "epochs": [25, 50],
    },
    "lwf": {
        "lr": [1e-3, 3e-3, 1e-2],
        "distill_lambda": [0.5, 1.0, 2.0],
        "distill_T": [1.5, 3.0],
        "epochs": [25, 50],
    },
    "bic": {
        "lr": [1e-3, 3e-3, 1e-2],
        "buffer_capacity": [128, 512],
        "distill_lambda": [0.5, 1.0, 2.0],
        "epochs": [25, 50],
    },
    "icarl": {
        "lr": [1e-3, 3e-3, 1e-2],
        "buffer_capacity": [128, 512],
        "distill_lambda": [0.5, 1.0, 2.0],
        "epochs": [25, 50],
    },
    "joint": {
        "lr": [1e-3, 3e-3, 1e-2],
        "epochs": [25, 50],
    },
    "joint_scratch": {
        "lr": [1e-3, 3e-3, 1e-2],
        "epochs": [25, 50],
    },
}
