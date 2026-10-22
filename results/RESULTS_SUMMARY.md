# Results Summary

_This file is generated automatically by `experiments/generate_summary_report.py` directly from the CSV/JSON outputs of the experiment scripts -- every number below is a real, computed result, not a placeholder._


**Data provenance.** CuCD-ID is the labelled NOS3/cFS software-in-the-loop CubeSat telemetry dataset of Fayyaz et al. (Data in Brief 2026, doi:10.1016/j.dib.2026.112598; Mendeley doi:10.17632/7n2d42pm3n.3); all telemetry is simulated, not captured from an orbiting spacecraft. STIN (SAT20/TER20) is the satellite/terrestrial flow dataset of Li et al. (IEEE Access 2020, doi:10.1109/ACCESS.2020.3041641; github.com/kun9717/STIN-data-set) and is used as a cross-domain stress test. Headline tables below use the IID random split; Sections 5.3 and 7.4 quantify the resulting leakage/feature-reliance caveats.

## 1. Hyperparameter protocol (STIN prior + per-method/arch validation tuning)

Coarse STIN/MLP/full grid: **160** configurations, scored on validation.

Top 5 by validation score (ACC - max(FM,0)):

|    lr |   ewc_lambda |   buffer_capacity |   replay_frac |    ACC |     BWT |     FM |   score |
|------:|-------------:|------------------:|--------------:|-------:|--------:|-------:|--------:|
| 0.001 |          100 |               128 |           0.9 | 0.6556 | -0.1867 | 0.1867 |  0.4689 |
| 0.001 |          100 |              1024 |           0.7 | 0.6474 | -0.3189 | 0.3189 |  0.3285 |
| 0.003 |          100 |               128 |           0.9 | 0.6296 | -0.3056 | 0.3056 |  0.3241 |
| 0.003 |         5000 |               512 |           0.7 | 0.6689 | -0.3511 | 0.3511 |  0.3178 |
| 0.003 |         1000 |               128 |           0.9 | 0.5867 | -0.2711 | 0.2711 |  0.3156 |

**Coarse STIN prior (NOT the final configs):**

```json
{
  "lr": 0.001,
  "ewc_lambda": 100.0,
  "buffer_capacity": 128,
  "replay_frac": 0.9
}
```


**Final frozen configs** (`best_hparams_per_method_arch.json`, per method x architecture, tuned on CuCD-ID validation, 3 seeds each):

| arch   | method        |   val_score | hparams                                                                                                     |
|:-------|:--------------|------------:|:------------------------------------------------------------------------------------------------------------|
| CNN1D  | bic           |      0.6306 | {"buffer_capacity": 128, "distill_lambda": 2.0, "epochs": 50, "lr": 0.01}                                   |
| CNN1D  | ewc           |     -0.727  | {"epochs": 25, "ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.001}                                            |
| CNN1D  | finetune      |     -0.6881 | {"epochs": 50, "lr": 0.001}                                                                                 |
| CNN1D  | full          |      0.3057 | {"buffer_capacity": 512, "epochs": 50, "ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.01, "replay_frac": 0.7} |
| CNN1D  | icarl         |      0.476  | {"buffer_capacity": 512, "distill_lambda": 0.5, "epochs": 50, "lr": 0.01}                                   |
| CNN1D  | joint         |      0.9869 | {"epochs": 50, "lr": 0.01}                                                                                  |
| CNN1D  | joint_scratch |      0.9615 | {"epochs": 50, "lr": 0.01}                                                                                  |
| CNN1D  | lwf           |     -0.727  | {"distill_T": 1.5, "distill_lambda": 0.5, "epochs": 25, "lr": 0.001}                                        |
| CNN1D  | replay        |      0.4569 | {"buffer_capacity": 128, "epochs": 50, "lr": 0.01, "replay_frac": 0.9}                                      |
| GRU    | bic           |      0.8494 | {"buffer_capacity": 512, "distill_lambda": 0.5, "epochs": 50, "lr": 0.01}                                   |
| GRU    | ewc           |     -0.0419 | {"epochs": 25, "ewc_gamma": 0.5, "ewc_lambda": 5000, "lr": 0.001}                                           |
| GRU    | finetune      |     -0.7045 | {"epochs": 25, "lr": 0.001}                                                                                 |
| GRU    | full          |      0.161  | {"buffer_capacity": 512, "epochs": 50, "ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.01, "replay_frac": 0.7} |
| GRU    | icarl         |      0.8311 | {"buffer_capacity": 512, "distill_lambda": 0.5, "epochs": 50, "lr": 0.01}                                   |
| GRU    | joint         |      0.9709 | {"epochs": 50, "lr": 0.003}                                                                                 |
| GRU    | joint_scratch |      0.9832 | {"epochs": 50, "lr": 0.003}                                                                                 |
| GRU    | lwf           |     -0.663  | {"distill_T": 3.0, "distill_lambda": 2.0, "epochs": 50, "lr": 0.003}                                        |
| GRU    | replay        |      0.8775 | {"buffer_capacity": 512, "epochs": 50, "lr": 0.01, "replay_frac": 0.9}                                      |
| MLP    | bic           |      0.9887 | {"buffer_capacity": 512, "distill_lambda": 1.0, "epochs": 25, "lr": 0.01}                                   |
| MLP    | ewc           |     -0.7176 | {"epochs": 25, "ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.001}                                            |
| MLP    | finetune      |     -0.7496 | {"epochs": 25, "lr": 0.01}                                                                                  |
| MLP    | full          |      0.9632 | {"buffer_capacity": 512, "epochs": 50, "ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.01, "replay_frac": 0.7} |
| MLP    | icarl         |      0.9809 | {"buffer_capacity": 512, "distill_lambda": 1.0, "epochs": 25, "lr": 0.01}                                   |
| MLP    | joint         |      0.9944 | {"epochs": 50, "lr": 0.003}                                                                                 |
| MLP    | joint_scratch |      0.9948 | {"epochs": 25, "lr": 0.01}                                                                                  |
| MLP    | lwf           |     -0.7474 | {"distill_T": 3.0, "distill_lambda": 0.5, "epochs": 25, "lr": 0.001}                                        |
| MLP    | replay        |      0.9694 | {"buffer_capacity": 512, "epochs": 50, "lr": 0.01, "replay_frac": 0.5}                                      |

Every main/Phase-2 run below uses its matching frozen config; test data is never used for selection.

## 2. Main experiment matrix

Total individual runs: **540** (2 datasets x 3 architectures x 9 methods x 10 seeds).


### CuCD-ID

**Average Accuracy (ACC), mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.7829 | 0.2501 |     0.2657 | 0.6596 |  0.8187 |  0.9883 |          0.9747 | 0.2501 |   0.7512 |
| GRU    | 0.9077 | 0.3833 |     0.2608 | 0.6373 |  0.9242 |  0.9624 |          0.9876 | 0.2778 |   0.9275 |
| MLP    | 0.9564 | 0.2896 |     0.25   | 0.948  |  0.9413 |  0.9953 |          0.9962 | 0.2573 |   0.8493 |

**Backward Transfer (BWT), mean over 10 seeds (negative = forgetting):**

| arch   |     bic |     ewc |   finetune |    full |   icarl |   joint |   joint_scratch |     lwf |   replay |
|:-------|--------:|--------:|-----------:|--------:|--------:|--------:|----------------:|--------:|---------:|
| CNN1D  | -0.2514 | -0.981  |    -0.9725 | -0.4433 | -0.2378 | -0.006  |         -0.0089 | -0.9809 |  -0.3277 |
| GRU    | -0.0922 | -0.4726 |    -0.9805 | -0.4564 | -0.0819 | -0.0142 |         -0.0035 | -0.958  |  -0.0823 |
| MLP    | -0.0347 | -0.947  |    -0.9998 | -0.0649 | -0.0761 | -0.0017 |         -0.0016 | -0.9901 |  -0.1987 |

**Macro-F1 (primary, BiC-corrected), mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.8067 | 0.0724 |     0.1077 | 0.6709 |  0.8343 |  0.9894 |          0.9771 | 0.0726 |   0.7774 |
| GRU    | 0.9147 | 0.3743 |     0.109  | 0.6686 |  0.9209 |  0.9675 |          0.9886 | 0.1333 |   0.9384 |
| MLP    | 0.9626 | 0.1234 |     0.0726 | 0.9529 |  0.9432 |  0.9957 |          0.9964 | 0.0886 |   0.8408 |

**Balanced Accuracy, mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.8064 | 0.2001 |     0.2218 | 0.7063 |  0.8367 |  0.9896 |          0.9784 | 0.2002 |   0.7965 |
| GRU    | 0.9125 | 0.4204 |     0.2207 | 0.6871 |  0.9167 |  0.9685 |          0.9889 | 0.2367 |   0.9377 |
| MLP    | 0.9624 | 0.2375 |     0.2    | 0.9561 |  0.9487 |  0.996  |          0.9966 | 0.2103 |   0.8656 |

**Binary detection (1-p(normal), CuCD-ID only; STIN has no benign class):**

| arch   | method        |   DET_AUROC_mean |   DET_AUPR_mean |   DET_FPR95_mean |
|:-------|:--------------|-----------------:|----------------:|-----------------:|
| CNN1D  | bic           |           0.9998 |          0.9999 |           0.001  |
| CNN1D  | ewc           |           0.9752 |          0.9929 |           0.1309 |
| CNN1D  | finetune      |           0.9904 |          0.9972 |           0.0485 |
| CNN1D  | full          |           0.9999 |          1      |           0.0004 |
| CNN1D  | icarl         |           0.9998 |          0.9999 |           0.0011 |
| CNN1D  | joint         |           1      |          1      |           0.0003 |
| CNN1D  | joint_scratch |           0.9999 |          1      |           0.0005 |
| CNN1D  | lwf           |           0.9705 |          0.9918 |           0.1644 |
| CNN1D  | replay        |           0.9997 |          0.9999 |           0.0016 |
| GRU    | bic           |           0.9909 |          0.9963 |           0.0359 |
| GRU    | ewc           |           0.9077 |          0.9708 |           0.4045 |
| GRU    | finetune      |           0.6215 |          0.8415 |           0.7891 |
| GRU    | full          |           0.994  |          0.9979 |           0.0248 |
| GRU    | icarl         |           0.9912 |          0.9965 |           0.0326 |
| GRU    | joint         |           0.9983 |          0.998  |           0.0031 |
| GRU    | joint_scratch |           0.9991 |          0.9996 |           0.0025 |
| GRU    | lwf           |           0.7889 |          0.8501 |           0.5581 |
| GRU    | replay        |           0.9968 |          0.9983 |           0.0073 |
| MLP    | bic           |           0.9999 |          1      |           0.0005 |
| MLP    | ewc           |           0.9979 |          0.9992 |           0.0122 |
| MLP    | finetune      |           0.7968 |          0.8411 |           0.5934 |
| MLP    | full          |           0.9997 |          0.9996 |           0.0008 |
| MLP    | icarl         |           0.9999 |          1      |           0.0003 |
| MLP    | joint         |           0.9999 |          0.9998 |           0.0001 |
| MLP    | joint_scratch |           1      |          1      |           0.0001 |
| MLP    | lwf           |           0.9998 |          0.9997 |           0.0005 |
| MLP    | replay        |           0.9998 |          0.9999 |           0.001  |

### STIN

**Average Accuracy (ACC), mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.5823 | 0.2118 |     0.2513 | 0.5446 |  0.6197 |  0.6616 |          0.6344 | 0.242  |   0.5906 |
| GRU    | 0.6779 | 0.3068 |     0.3078 | 0.642  |  0.6614 |  0.6665 |          0.6536 | 0.2779 |   0.6468 |
| MLP    | 0.68   | 0.2719 |     0.2873 | 0.6524 |  0.6637 |  0.6792 |          0.639  | 0.284  |   0.6382 |

**Backward Transfer (BWT), mean over 10 seeds (negative = forgetting):**

| arch   |     bic |     ewc |   finetune |    full |   icarl |   joint |   joint_scratch |     lwf |   replay |
|:-------|--------:|--------:|-----------:|--------:|--------:|--------:|----------------:|--------:|---------:|
| CNN1D  | -0.314  | -0.7526 |    -0.8177 | -0.4225 | -0.3997 | -0.0341 |         -0.0656 | -0.7718 |  -0.3729 |
| GRU    | -0.1951 | -0.6248 |    -0.6737 | -0.3881 | -0.3862 | -0.0633 |         -0.0774 | -0.8486 |  -0.3646 |
| MLP    | -0.1766 | -0.9028 |    -0.9447 | -0.3604 | -0.3758 | -0.0522 |         -0.0177 | -0.9303 |  -0.3984 |

**Macro-F1 (primary, BiC-corrected), mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.5312 | 0.0896 |     0.1232 | 0.4685 |  0.5514 |  0.6126 |          0.5726 | 0.1136 |   0.5134 |
| GRU    | 0.6228 | 0.1878 |     0.1831 | 0.5722 |  0.59   |  0.6082 |          0.589  | 0.1672 |   0.5803 |
| MLP    | 0.6329 | 0.1477 |     0.1632 | 0.588  |  0.5928 |  0.6348 |          0.5734 | 0.161  |   0.5641 |

**Balanced Accuracy, mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.5823 | 0.2118 |     0.2513 | 0.5446 |  0.6197 |  0.6616 |          0.6344 | 0.242  |   0.5906 |
| GRU    | 0.6779 | 0.3068 |     0.3078 | 0.642  |  0.6614 |  0.6665 |          0.6536 | 0.2779 |   0.6468 |
| MLP    | 0.68   | 0.2719 |     0.2873 | 0.6524 |  0.6637 |  0.6792 |          0.639  | 0.284  |   0.6382 |

## 3. Memory: persistent deployment vs peak update RAM (method=full)

| dataset   | arch   |   persistent_total_KB |   peak_total_KB |   macs_per_forward |   persistent_model_KB |   persistent_cmmu_KB |
|:----------|:-------|----------------------:|----------------:|-------------------:|----------------------:|---------------------:|
| CuCD-ID   | CNN1D  |                 13.31 |           66.86 |               4406 |                  0.26 |                13.05 |
| CuCD-ID   | GRU    |                 14.17 |           74    |               4010 |                  0.43 |                13.74 |
| CuCD-ID   | MLP    |                 18.59 |           99.43 |               1296 |                  1.32 |                17.27 |
| STIN      | CNN1D  |                 17.53 |           86.44 |               6030 |                  0.31 |                17.22 |
| STIN      | GRU    |                 18.53 |           93.74 |               3990 |                  0.51 |                18.03 |
| STIN      | MLP    |                 24.17 |          127.25 |               1616 |                  1.63 |                22.54 |

All configurations fit within the **assumed 60 KB TinyML persistent-state budget** (a conservative study constraint, not a universal CubeSat limit; largest persistent: **24.17 KB**, 40.3% of budget). Estimated peak update RAM is larger (max **127.25 KB** host-estimated, counting FP32 weights, gradients, Adam states, activations, batches, and teacher) -- a reported constraint on onboard training, not covered by the persistent number.


## 4. Estimated uplink payload sizes (packed telemetry payload)

Computed directly from the constructed task splits (`src/data_utils.py`), reflecting the KB-scale uplink constraint from the operational concept: task 0 is trained on a larger pre-launch corpus, while every subsequent task is trained only on a small uplinked labelled batch. Figures are **packed INT8 payload estimates** (rows x features plus label bytes), excluding CSV text formatting, scale/zero-point parameters, and packet/protocol framing overhead. The reported uplink budget **includes both adaptation data and the onboard calibration data** consumed by BiC fitting and val-balanced Fisher estimation (Option A): a deployed update must transmit both for the satellite to reproduce these results.

| dataset   | task                  |   n_train_rows |   n_calib_rows |   n_new_classes |   uplink_KB_fp32 |   uplink_KB_int8 |   uplink_KB_calib_int8 |   uplink_KB_total_with_calib |
|:----------|:----------------------|---------------:|---------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------------:|
| CuCD-ID   | T0_launch             |           1400 |            300 |               2 |           131.25 |            31.45 |                   6.74 |                        38.18 |
| CuCD-ID   | T1_data_injection     |            150 |            150 |               1 |            14.06 |             3.37 |                   3.37 |                         6.74 |
| CuCD-ID   | T2_defence_impairment |            150 |            150 |               1 |            14.06 |             3.37 |                   3.37 |                         6.74 |
| CuCD-ID   | T3_storage_exhaustion |            150 |            150 |               1 |            14.06 |             3.37 |                   3.37 |                         6.74 |
| STIN      | T0_flood_base         |           2100 |            450 |               3 |           262.5  |            63.57 |                  13.62 |                        77.2  |
| STIN      | T1_flood_ext          |            450 |            450 |               3 |            56.25 |            13.62 |                  13.62 |                        27.25 |
| STIN      | T2_app_layer          |            450 |            450 |               3 |            56.25 |            13.62 |                  13.62 |                        27.25 |

## 5. Ablation studies

### 5.1 Replay buffer capacity sensitivity (both datasets x all 3 architectures, method=full)

| dataset   | arch   |   buffer_capacity |    ACC |     BWT |     FM |
|:----------|:-------|------------------:|-------:|--------:|-------:|
| CuCD-ID   | CNN1D  |                32 | 0.653  | -0.4538 | 0.4538 |
| CuCD-ID   | CNN1D  |                64 | 0.6569 | -0.4499 | 0.4499 |
| CuCD-ID   | CNN1D  |               128 | 0.6604 | -0.4446 | 0.4447 |
| CuCD-ID   | CNN1D  |               256 | 0.6729 | -0.4281 | 0.4282 |
| CuCD-ID   | CNN1D  |               512 | 0.6596 | -0.4433 | 0.4434 |
| CuCD-ID   | CNN1D  |              1024 | 0.6603 | -0.4438 | 0.4439 |
| CuCD-ID   | GRU    |                32 | 0.4879 | -0.6449 | 0.6449 |
| CuCD-ID   | GRU    |                64 | 0.5184 | -0.614  | 0.614  |
| CuCD-ID   | GRU    |               128 | 0.5422 | -0.5674 | 0.5674 |
| CuCD-ID   | GRU    |               256 | 0.6088 | -0.4891 | 0.4891 |
| CuCD-ID   | GRU    |               512 | 0.6373 | -0.4564 | 0.4564 |
| CuCD-ID   | GRU    |              1024 | 0.6695 | -0.4189 | 0.4189 |
| CuCD-ID   | MLP    |                32 | 0.7219 | -0.3655 | 0.3655 |
| CuCD-ID   | MLP    |                64 | 0.8172 | -0.2406 | 0.2406 |
| CuCD-ID   | MLP    |               128 | 0.9096 | -0.116  | 0.116  |
| CuCD-ID   | MLP    |               256 | 0.953  | -0.0577 | 0.0577 |
| CuCD-ID   | MLP    |               512 | 0.948  | -0.0649 | 0.0649 |
| CuCD-ID   | MLP    |              1024 | 0.9558 | -0.0539 | 0.0539 |
| STIN      | CNN1D  |                32 | 0.5404 | -0.4432 | 0.4432 |
| STIN      | CNN1D  |                64 | 0.5364 | -0.4514 | 0.4514 |
| STIN      | CNN1D  |               128 | 0.5374 | -0.4412 | 0.4412 |
| STIN      | CNN1D  |               256 | 0.5439 | -0.4332 | 0.4332 |
| STIN      | CNN1D  |               512 | 0.5446 | -0.4225 | 0.4225 |
| STIN      | CNN1D  |              1024 | 0.5387 | -0.4309 | 0.4309 |
| STIN      | GRU    |                32 | 0.6131 | -0.4053 | 0.4053 |
| STIN      | GRU    |                64 | 0.635  | -0.3902 | 0.3902 |
| STIN      | GRU    |               128 | 0.6417 | -0.336  | 0.336  |
| STIN      | GRU    |               256 | 0.6372 | -0.3882 | 0.3882 |
| STIN      | GRU    |               512 | 0.642  | -0.3881 | 0.3881 |
| STIN      | GRU    |              1024 | 0.6352 | -0.3833 | 0.3833 |
| STIN      | MLP    |                32 | 0.6145 | -0.4314 | 0.4314 |
| STIN      | MLP    |                64 | 0.6377 | -0.3906 | 0.3906 |
| STIN      | MLP    |               128 | 0.6554 | -0.3536 | 0.3536 |
| STIN      | MLP    |               256 | 0.661  | -0.3494 | 0.3494 |
| STIN      | MLP    |               512 | 0.6524 | -0.3604 | 0.3604 |
| STIN      | MLP    |              1024 | 0.6657 | -0.3576 | 0.3576 |

Per-architecture ACC range across all tested buffer capacities:

| dataset   | arch   |    min |    max |   span |
|:----------|:-------|-------:|-------:|-------:|
| CuCD-ID   | CNN1D  | 0.653  | 0.6729 | 0.0199 |
| CuCD-ID   | GRU    | 0.4879 | 0.6695 | 0.1815 |
| CuCD-ID   | MLP    | 0.7219 | 0.9558 | 0.2339 |
| STIN      | CNN1D  | 0.5364 | 0.5446 | 0.0082 |
| STIN      | GRU    | 0.6131 | 0.642  | 0.0289 |
| STIN      | MLP    | 0.6145 | 0.6657 | 0.0512 |

### 5.2 EWC lambda sensitivity (both datasets x all 3 architectures, method=full)

| dataset   | arch   |   ewc_lambda |    ACC |     BWT |     FM |
|:----------|:-------|-------------:|-------:|--------:|-------:|
| CuCD-ID   | CNN1D  |            0 | 0.7108 | -0.382  | 0.3822 |
| CuCD-ID   | CNN1D  |           10 | 0.664  | -0.4416 | 0.4417 |
| CuCD-ID   | CNN1D  |          100 | 0.6596 | -0.4433 | 0.4434 |
| CuCD-ID   | CNN1D  |         1000 | 0.6362 | -0.471  | 0.471  |
| CuCD-ID   | CNN1D  |         5000 | 0.6207 | -0.488  | 0.488  |
| CuCD-ID   | CNN1D  |        20000 | 0.6057 | -0.5066 | 0.5066 |
| CuCD-ID   | CNN1D  |        50000 | 0.6003 | -0.5133 | 0.5133 |
| CuCD-ID   | GRU    |            0 | 0.904  | -0.1121 | 0.1121 |
| CuCD-ID   | GRU    |           10 | 0.7623 | -0.2941 | 0.2941 |
| CuCD-ID   | GRU    |          100 | 0.6373 | -0.4564 | 0.4564 |
| CuCD-ID   | GRU    |         1000 | 0.5515 | -0.567  | 0.567  |
| CuCD-ID   | GRU    |         5000 | 0.5483 | -0.5655 | 0.5655 |
| CuCD-ID   | GRU    |        20000 | 0.5367 | -0.5753 | 0.5753 |
| CuCD-ID   | GRU    |        50000 | 0.5498 | -0.5503 | 0.5503 |
| CuCD-ID   | MLP    |            0 | 0.8636 | -0.1793 | 0.1793 |
| CuCD-ID   | MLP    |           10 | 0.9235 | -0.0972 | 0.0972 |
| CuCD-ID   | MLP    |          100 | 0.948  | -0.0649 | 0.0649 |
| CuCD-ID   | MLP    |         1000 | 0.9295 | -0.0902 | 0.0902 |
| CuCD-ID   | MLP    |         5000 | 0.91   | -0.1156 | 0.1156 |
| CuCD-ID   | MLP    |        20000 | 0.8571 | -0.1856 | 0.1856 |
| CuCD-ID   | MLP    |        50000 | 0.8546 | -0.1886 | 0.1886 |
| STIN      | CNN1D  |            0 | 0.6156 | -0.3573 | 0.3573 |
| STIN      | CNN1D  |           10 | 0.5617 | -0.4076 | 0.4076 |
| STIN      | CNN1D  |          100 | 0.5446 | -0.4225 | 0.4225 |
| STIN      | CNN1D  |         1000 | 0.5327 | -0.4474 | 0.4474 |
| STIN      | CNN1D  |         5000 | 0.5302 | -0.4505 | 0.4505 |
| STIN      | CNN1D  |        20000 | 0.521  | -0.4618 | 0.4618 |
| STIN      | CNN1D  |        50000 | 0.5196 | -0.4586 | 0.4586 |
| STIN      | GRU    |            0 | 0.6628 | -0.355  | 0.355  |
| STIN      | GRU    |           10 | 0.6618 | -0.3681 | 0.3681 |
| STIN      | GRU    |          100 | 0.642  | -0.3881 | 0.3881 |
| STIN      | GRU    |         1000 | 0.5987 | -0.4156 | 0.4156 |
| STIN      | GRU    |         5000 | 0.5779 | -0.4172 | 0.4172 |
| STIN      | GRU    |        20000 | 0.5525 | -0.4514 | 0.4514 |
| STIN      | GRU    |        50000 | 0.5103 | -0.4675 | 0.4675 |
| STIN      | MLP    |            0 | 0.6333 | -0.3972 | 0.3972 |
| STIN      | MLP    |           10 | 0.667  | -0.3252 | 0.3252 |
| STIN      | MLP    |          100 | 0.6524 | -0.3604 | 0.3604 |
| STIN      | MLP    |         1000 | 0.657  | -0.3684 | 0.3684 |
| STIN      | MLP    |         5000 | 0.6555 | -0.3572 | 0.3572 |
| STIN      | MLP    |        20000 | 0.6475 | -0.3654 | 0.3654 |
| STIN      | MLP    |        50000 | 0.6485 | -0.3759 | 0.3759 |

Per-(dataset, architecture) ACC range across all tested lambdas:

| dataset   | arch   |    min |    max |   span |
|:----------|:-------|-------:|-------:|-------:|
| CuCD-ID   | CNN1D  | 0.6003 | 0.7108 | 0.1105 |
| CuCD-ID   | GRU    | 0.5367 | 0.904  | 0.3673 |
| CuCD-ID   | MLP    | 0.8546 | 0.948  | 0.0934 |
| STIN      | CNN1D  | 0.5196 | 0.6156 | 0.096  |
| STIN      | GRU    | 0.5103 | 0.6628 | 0.1525 |
| STIN      | MLP    | 0.6333 | 0.667  | 0.0337 |

MLP's span is small on both datasets, while GRU and CNN1D both show a much larger span, and in both cases ACC *decreases* as lambda increases (visible directly in the per-lambda table above and in `figures/ewc_lambda_sensitivity.png`). This is why final numbers use frozen per-method x per-architecture validation configs instead of one shared configuration.

EWC here is online diagonal Fisher with a running estimate (`F_new = γ·F_old + F_current`). The Fisher sample is drawn from a class-balanced mix of the current task's training data and the cumulative validation pool of all tasks seen so far (not the current task alone), so old-task curvature is not dropped at each consolidation. Frozen per-architecture γ weighting: MLP γ=0.5, GRU γ=0.5, CNN1D γ=0.5. γ is treated as a validation-tuned weighting parameter rather than a conventional decay factor (GRU γ=2.0 amplifies retained Fisher by validation selection; a γ≤1 restriction would be stricter but would require rerunning EWC experiments).


### 5.3 Feature-provenance leakage ablation (CuCD-ID/MLP, 10 seeds)

Retrained models after removing feature groups (not test-time corruption). Deltas are relative to the all-feature run for the same method; negative values mean the removed features were helping the model.

| variant            | method        |   n_features |   ACC_mean |   MacroF1_mean |   ACC_mean_delta_vs_all |   MacroF1_mean_delta_vs_all |
|:-------------------|:--------------|-------------:|-----------:|---------------:|------------------------:|----------------------------:|
| all                | bic           |           22 |     0.9564 |         0.9626 |                  0      |                      0      |
| all                | joint_scratch |           22 |     0.9962 |         0.9964 |                  0      |                      0      |
| all                | replay        |           22 |     0.8493 |         0.8408 |                  0      |                      0      |
| no_ids             | bic           |           19 |     0.9398 |         0.9378 |                 -0.0166 |                     -0.0247 |
| no_ids             | joint_scratch |           19 |     0.9951 |         0.9956 |                 -0.0011 |                     -0.0008 |
| no_ids             | replay        |           19 |     0.8957 |         0.9025 |                  0.0464 |                      0.0617 |
| no_time            | bic           |           21 |     0.9476 |         0.9516 |                 -0.0088 |                     -0.0109 |
| no_time            | joint_scratch |           21 |     0.9955 |         0.9958 |                 -0.0007 |                     -0.0005 |
| no_time            | replay        |           21 |     0.984  |         0.9855 |                  0.1346 |                      0.1447 |
| no_time_ids_window | bic           |            9 |     0.8578 |         0.8437 |                 -0.0986 |                     -0.1188 |
| no_time_ids_window | joint_scratch |            9 |     0.9717 |         0.9753 |                 -0.0245 |                     -0.0211 |
| no_time_ids_window | replay        |            9 |     0.9177 |         0.926  |                  0.0683 |                      0.0852 |
| no_window          | bic           |           13 |     0.8306 |         0.8379 |                 -0.1258 |                     -0.1247 |
| no_window          | joint_scratch |           13 |     0.9915 |         0.9923 |                 -0.0047 |                     -0.0041 |
| no_window          | replay        |           13 |     0.737  |         0.7274 |                 -0.1124 |                     -0.1134 |

This quantifies how much random-split performance rides on the circular time field, CCSDS identifiers, and 20-second window aggregates. It is a reliance measure, not proof that every removed feature is illegitimate: `no_ids` also removes protocol information a real detector may legitimately use. The IID headline remains a benchmark estimand; the temporal-purged table in Section 7.4 is the separate deployment-shift estimand.


### 5.4 Task-order sensitivity, legacy 3-perm (CuCD-ID, MLP, method=full)

_Legacy confounded design (budgets move with the task); superseded by the disentangled 12-permutation study in §7.3. Retained for provenance only._

| order        |   ACC_mean |   ACC_std |   BWT_mean |   BWT_std |
|:-------------|-----------:|----------:|-----------:|----------:|
| [0, 1, 2, 3] |     0.8634 |    0.1474 |    -0.179  |    0.1971 |
| [1, 3, 0, 2] |     0.9765 |    0.0266 |    -0.0281 |    0.0359 |
| [3, 2, 1, 0] |     0.9959 |    0.0023 |    -0.0036 |    0.0024 |

ACC spread across the 3 legacy orderings: **0.1325** (range 0.8634 - 0.9959). See §7.3 for the disentangled measurement (computed from `task_order_disentangled.csv`, not hard-coded).


## 6. Per-class F1 -- direct evidence of forgetting

Section 2's ACC/BWT/FM numbers are task-level aggregates; this section makes the same phenomenon visible at the level of individual attack classes by evaluating the final trained model (all classes seen so far, single shared head) and computing a per-class precision/recall/F1 breakdown. See `figures/per_class_f1_*.png` and `figures/confusion_matrix_*.png` for the visual version.


### 6.1 STIN / GRU, mean F1 per task-of-origin

| task_name     |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:--------------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| T0_flood_base | 0.7512 | 0.2028 |     0.1709 | 0.5653 |  0.5409 |  0.7603 |          0.7551 | 0.0596 |   0.6105 |
| T1_flood_ext  | 0.6276 | 0.0141 |     0      | 0.5243 |  0.5676 |  0.7661 |          0.6941 | 0      |   0.4978 |
| T2_app_layer  | 0.4896 | 0.3466 |     0.3784 | 0.6271 |  0.6616 |  0.2981 |          0.3177 | 0.442  |   0.6325 |

### 6.2 CuCD-ID / MLP, mean F1 per task-of-origin

| task_name             |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:----------------------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| T0_launch             | 0.9917 | 0.0403 |      0     | 0.9913 |  0.9891 |  0.9966 |          0.9968 | 0.0368 |   0.94   |
| T1_data_injection     | 0.9241 | 0      |      0     | 0.8735 |  0.8276 |  0.9969 |          0.9969 | 0      |   0.5885 |
| T2_defence_impairment | 0.9868 | 0.1519 |      0     | 0.9805 |  0.9916 |  0.9961 |          0.9975 | 0.0005 |   0.89   |
| T3_storage_exhaustion | 0.9184 | 0.3845 |      0.363 | 0.9277 |  0.9185 |  0.9924 |          0.9938 | 0.3691 |   0.8454 |

For CuCD-ID, the memoryless methods (`finetune`, `ewc`) collapse most classes to zero or near-zero F1 while retaining some ability on the most recent class; the exact per-class pattern is shown above and varies by method and seed. On STIN the same severe recency bias appears, but it is noisier because each task introduces several classes and the feature set is less separable. Memory-based and joint references retain non-zero F1 across most tasks. This is the direct per-class confirmation of the task-level results in Section 2; do not infer a universal exact-zero pattern from one method.


## 7. Phase-2 evidence (all numbers computed from their CSVs)


### 7.1 Few-shot uplink sweep, CuCD-ID (RQ4)

| arch   | method   |      5 |     10 |     25 |     50 |    100 |    150 |    300 |
|:-------|:---------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| GRU    | icarl    | 0.4504 | 0.5907 | 0.7558 | 0.8489 | 0.8822 | 0.9236 | 0.9374 |
| GRU    | replay   | 0.4623 | 0.5379 | 0.5178 | 0.6374 | 0.8839 | 0.929  | 0.965  |
| MLP    | bic      | 0.4692 | 0.6944 | 0.6667 | 0.8103 | 0.9755 | 0.9873 | 0.9767 |
| MLP    | full     | 0.4672 | 0.4804 | 0.5804 | 0.6349 | 0.8067 | 0.963  | 0.8956 |
| MLP    | icarl    | 0.4672 | 0.6268 | 0.5758 | 0.7736 | 0.976  | 0.9913 | 0.9912 |
| MLP    | replay   | 0.4587 | 0.4889 | 0.502  | 0.6287 | 0.7674 | 0.8943 | 0.9925 |

Adaptation uplink KB for T1..T3 per K: K=5: 0.3KB, K=10: 0.7KB, K=25: 1.7KB, K=50: 3.4KB, K=100: 6.7KB, K=150: 10.1KB, K=300: 20.2KB.

Total uplink KB **including onboard calibration data** (BiC/Fisher val samples, Option A) per K: K=5: 10.4KB, K=10: 10.8KB, K=25: 11.8KB, K=50: 13.5KB, K=100: 16.9KB, K=150: 20.2KB, K=300: 30.3KB.

MLP @K=50 (knee): bic=0.810, full=0.635, icarl=0.774, replay=0.629.

MLP @K=5 (extreme): bic=0.469, full=0.467, icarl=0.467, replay=0.459.


### 7.2 Per-arch tune, CuCD-ID/CNN (exploratory, validation-scored, not applied)

| method        |    lr |   epochs |    ACC |
|:--------------|------:|---------:|-------:|
| icarl         | 0.001 |       25 | 0.2733 |
| icarl         | 0.001 |       50 | 0.3308 |
| icarl         | 0.003 |       25 | 0.3767 |
| icarl         | 0.003 |       50 | 0.6069 |
| icarl         | 0.01  |       25 | 0.6956 |
| icarl         | 0.01  |       50 | 0.7742 |
| joint_scratch | 0.001 |       25 | 0.6853 |
| joint_scratch | 0.001 |       50 | 0.825  |
| joint_scratch | 0.003 |       25 | 0.8411 |
| joint_scratch | 0.003 |       50 | 0.9464 |
| joint_scratch | 0.01  |       25 | 0.9433 |
| joint_scratch | 0.01  |       50 | 0.9667 |
| replay        | 0.001 |       25 | 0.3822 |
| replay        | 0.001 |       50 | 0.4589 |
| replay        | 0.003 |       25 | 0.4811 |
| replay        | 0.003 |       50 | 0.5256 |
| replay        | 0.01  |       25 | 0.6739 |
| replay        | 0.01  |       50 | 0.7647 |

- replay: best lr=0.01, epochs=50 -> ACC=0.765.

- icarl: best lr=0.01, epochs=50 -> ACC=0.774.

- joint_scratch: best lr=0.01, epochs=50 -> ACC=0.967.



### 7.3 Disentangled task-order study, CuCD-ID/MLP (RQ5)

| budget_mode     | method   |   mean |    std |
|:----------------|:---------|-------:|-------:|
| fixed_position  | icarl    | 0.9694 | 0.034  |
| fixed_position  | replay   | 0.8576 | 0.1713 |
| moves_with_task | icarl    | 0.9769 | 0.0242 |
| moves_with_task | replay   | 0.9384 | 0.1077 |

Aggregate difference between budget modes is small (fixed_position=0.913 vs moves_with_task=0.958; spread of per-order means across 12 permutations: **0.1199**), although individual runs vary more (raw ACC range **0.6140**). The legacy confounded headline is retired.


### 7.4 IID vs forward-temporal-purged, CuCD-ID/MLP

| method        | protocol        |   mean |    std |
|:--------------|:----------------|-------:|-------:|
| bic           | iid             | 0.9564 | 0.0526 |
| bic           | temporal_purged | 0.2473 | 0.1101 |
| ewc           | iid             | 0.2896 | 0.0686 |
| ewc           | temporal_purged | 0.1782 | 0.0779 |
| finetune      | iid             | 0.25   | 0      |
| finetune      | temporal_purged | 0.249  | 0.0022 |
| full          | iid             | 0.948  | 0.0728 |
| full          | temporal_purged | 0.1995 | 0.1002 |
| icarl         | iid             | 0.9413 | 0.0874 |
| icarl         | temporal_purged | 0.1989 | 0.1001 |
| joint         | iid             | 0.9953 | 0.0018 |
| joint         | temporal_purged | 0.1514 | 0.0535 |
| joint_scratch | iid             | 0.9962 | 0.0026 |
| joint_scratch | temporal_purged | 0.1332 | 0.0552 |
| lwf           | iid             | 0.2573 | 0.0207 |
| lwf           | temporal_purged | 0.2085 | 0.0599 |
| replay        | iid             | 0.8493 | 0.1854 |
| replay        | temporal_purged | 0.2528 | 0.0967 |

IID = random-split benchmark; temporal-purged = forward temporal generalization under distribution shift with reduced window overlap (separate estimands, not interpolated). Purge: 0.02 rad on TimeRadians (a proxy for the documented ~20-22 s window horizon; not a wall-clock guarantee). Blocks are ordered on unwrapped circular time so an orbital wrap does not scramble chronology. For classes whose TimeRadians span is narrow, the purge can empty the validation slice; those cells are warned about by the builder and val-dependent methods (BiC/EWC) are correspondingly weaker in this protocol. See DATALOG.md for the drift diagnosis.


### 7.5 Robustness + simulated-quantization ledger, CuCD-ID/MLP (RQ6/RQ7)

_`int8_sim` is a simulated post-training INT8 round-trip accuracy proxy (FP32 weights only); no INT8 kernels were executed and no hardware deployment is claimed._

| method        | corruption   |   ACC_final_pool |
|:--------------|:-------------|-----------------:|
| bic           | clean        |           0.9924 |
| bic           | drop_0.1     |           0.9292 |
| bic           | drop_0.3     |           0.7952 |
| bic           | gauss_0.1    |           0.9921 |
| bic           | gauss_0.5    |           0.958  |
| bic           | int8_sim     |           0.9926 |
| bic           | nowindow     |           0.6894 |
| icarl         | clean        |           0.9901 |
| icarl         | drop_0.1     |           0.9308 |
| icarl         | drop_0.3     |           0.8012 |
| icarl         | gauss_0.1    |           0.9889 |
| icarl         | gauss_0.5    |           0.9521 |
| icarl         | int8_sim     |           0.9906 |
| icarl         | nowindow     |           0.6907 |
| joint_scratch | clean        |           0.9975 |
| joint_scratch | drop_0.1     |           0.9435 |
| joint_scratch | drop_0.3     |           0.8134 |
| joint_scratch | gauss_0.1    |           0.9972 |
| joint_scratch | gauss_0.5    |           0.977  |
| joint_scratch | int8_sim     |           0.9976 |
| joint_scratch | nowindow     |           0.7553 |
| replay        | clean        |           0.9911 |
| replay        | drop_0.1     |           0.9274 |
| replay        | drop_0.3     |           0.7931 |
| replay        | gauss_0.1    |           0.9889 |
| replay        | gauss_0.5    |           0.9324 |
| replay        | int8_sim     |           0.9916 |
| replay        | nowindow     |           0.5835 |

Mean pooled-ACC drop vs clean (all methods): drop_0.1=+0.060, drop_0.3=+0.192, gauss_0.1=+0.001, gauss_0.5=+0.038, int8_sim=-0.000, nowindow=+0.313.


### 7.6 Activation memory ledger (analytic, batch=1, SIM_ = host-simulated)

| arch   |   act_fwd_floats |   SIM_act_fwd_KB |   SIM_act_train_KB | seq_len   |
|:-------|-----------------:|-----------------:|-------------------:|:----------|
| MLP    |              101 |            0.395 |              0.789 |           |
| GRU    |              775 |            3.027 |              6.055 | 11.0000   |
| CNN1D  |              719 |            2.809 |              5.617 |           |



### 7.7 Exact paired permutation + Holm (matched seeds, ACC): 8/23 headline comparisons significant at Holm p<0.05

Holm correction is applied across all comparisons in this metric. The exact sign-flip test has n=10 paired seeds.

| dataset   | arch   | method_a   | method_b      |   n |   mean_a |   mean_b |   mean_diff |   ci_low |   ci_high |   cohens_d |   p_exact |   p_holm |
|:----------|:-------|:-----------|:--------------|----:|---------:|---------:|------------:|---------:|----------:|-----------:|----------:|---------:|
| CuCD-ID   | MLP    | full       | finetune      |  10 |   0.948  |   0.25   |      0.698  |   0.6504 |    0.7327 |     9.594  |    0.002  |   0.043  |
| CuCD-ID   | MLP    | replay     | finetune      |  10 |   0.8493 |   0.25   |      0.5993 |   0.4865 |    0.7017 |     3.233  |    0      |   0      |
| STIN      | MLP    | full       | finetune      |  10 |   0.6524 |   0.2873 |      0.3651 |   0.3434 |    0.3849 |    10.3474 |    0.002  |   0.043  |
| STIN      | MLP    | replay     | finetune      |  10 |   0.6382 |   0.2873 |      0.3508 |   0.3364 |    0.3637 |    14.557  |    0.002  |   0.043  |
| STIN      | CNN1D  | replay     | finetune      |  10 |   0.5906 |   0.2513 |      0.3392 |   0.3148 |    0.357  |     9.1912 |    0.002  |   0.043  |
| STIN      | GRU    | replay     | finetune      |  10 |   0.6468 |   0.3078 |      0.3391 |   0.3038 |    0.3787 |     5.2406 |    0.002  |   0.043  |
| CuCD-ID   | GRU    | icarl      | full          |  10 |   0.9242 |   0.6373 |      0.2868 |   0.2412 |    0.331  |     3.6986 |    0.002  |   0.043  |
| CuCD-ID   | MLP    | icarl      | replay        |  10 |   0.9413 |   0.8493 |      0.092  |   0.0032 |    0.2013 |     0.5419 |    0.1035 |   0.9316 |
| STIN      | MLP    | bic        | replay        |  10 |   0.68   |   0.6382 |      0.0419 |   0.0316 |    0.0548 |     2.0572 |    0.002  |   0.043  |
| STIN      | MLP    | bic        | joint_scratch |  10 |   0.68   |   0.639  |      0.041  |   0.0199 |    0.063  |     1.1117 |    0.0098 |   0.1367 |
| CuCD-ID   | CNN1D  | bic        | replay        |  10 |   0.7829 |   0.7512 |      0.0317 |  -0.0229 |    0.0866 |     0.3401 |    0.3008 |   1      |
| STIN      | CNN1D  | icarl      | replay        |  10 |   0.6197 |   0.5906 |      0.0292 |   0.0102 |    0.0503 |     0.8447 |    0.0195 |   0.2539 |
| STIN      | MLP    | bic        | full          |  10 |   0.68   |   0.6524 |      0.0276 |   0.0089 |    0.0491 |     0.8061 |    0.0234 |   0.2812 |
| STIN      | CNN1D  | joint      | joint_scratch |  10 |   0.6616 |   0.6344 |      0.0272 |   0.0126 |    0.0431 |     1.03   |    0.0078 |   0.1172 |
| STIN      | GRU    | icarl      | full          |  10 |   0.6614 |   0.642  |      0.0194 |   0.005  |    0.0321 |     0.8347 |    0.0273 |   0.3008 |
| STIN      | GRU    | icarl      | replay        |  10 |   0.6614 |   0.6468 |      0.0146 |  -0.0022 |    0.0345 |     0.4648 |    0.2305 |   1      |
| STIN      | GRU    | joint      | joint_scratch |  10 |   0.6665 |   0.6536 |      0.0129 |  -0.0088 |    0.0353 |     0.3386 |    0.3086 |   1      |
| CuCD-ID   | GRU    | icarl      | replay        |  10 |   0.9242 |   0.9275 |     -0.0033 |  -0.0158 |    0.0093 |    -0.1559 |    0.6191 |   1      |
| CuCD-ID   | MLP    | icarl      | full          |  10 |   0.9413 |   0.948  |     -0.0066 |  -0.0759 |    0.0656 |    -0.0544 |    0.9355 |   1      |
| STIN      | CNN1D  | bic        | replay        |  10 |   0.5823 |   0.5906 |     -0.0083 |  -0.0235 |    0.0078 |    -0.3109 |    0.3379 |   1      |
| CuCD-ID   | MLP    | icarl      | bic           |  10 |   0.9413 |   0.9564 |     -0.0151 |  -0.045  |    0.0042 |    -0.3321 |    0.3652 |   1      |
| CuCD-ID   | CNN1D  | bic        | icarl         |  10 |   0.7829 |   0.8187 |     -0.0358 |  -0.0878 |    0.0169 |    -0.4058 |    0.2324 |   1      |
| STIN      | CNN1D  | bic        | icarl         |  10 |   0.5823 |   0.6197 |     -0.0374 |  -0.0704 |   -0.0072 |    -0.6926 |    0.0586 |   0.5859 |

### 7.7 Exact paired permutation + Holm (matched seeds, Macro-F1 (primary)): 10/23 headline comparisons significant at Holm p<0.05

Holm correction is applied across all comparisons in this metric. The exact sign-flip test has n=10 paired seeds.

| dataset   | arch   | method_a   | method_b      |   n |   mean_a |   mean_b |   mean_diff |   ci_low |   ci_high |   cohens_d |   p_exact |   p_holm |
|:----------|:-------|:-----------|:--------------|----:|---------:|---------:|------------:|---------:|----------:|-----------:|----------:|---------:|
| CuCD-ID   | MLP    | full       | finetune      |  10 |   0.9529 |   0.0726 |      0.8803 |   0.8352 |    0.9119 |    12.7834 |    0.002  |   0.0449 |
| CuCD-ID   | MLP    | replay     | finetune      |  10 |   0.8408 |   0.0726 |      0.7682 |   0.6395 |    0.8814 |     3.7497 |    0.002  |   0.0449 |
| STIN      | MLP    | full       | finetune      |  10 |   0.588  |   0.1632 |      0.4248 |   0.4005 |    0.4463 |    10.9737 |    0.002  |   0.0449 |
| STIN      | MLP    | replay     | finetune      |  10 |   0.5641 |   0.1632 |      0.4009 |   0.3844 |    0.4167 |    14.1621 |    0.002  |   0.0449 |
| STIN      | GRU    | replay     | finetune      |  10 |   0.5803 |   0.1831 |      0.3972 |   0.3618 |    0.4365 |     6.1456 |    0.002  |   0.0449 |
| STIN      | CNN1D  | replay     | finetune      |  10 |   0.5134 |   0.1232 |      0.3902 |   0.3566 |    0.4188 |     7.4388 |    0.002  |   0.0449 |
| CuCD-ID   | GRU    | icarl      | full          |  10 |   0.9209 |   0.6686 |      0.2522 |   0.2056 |    0.2991 |     3.1158 |    0.002  |   0.0449 |
| CuCD-ID   | MLP    | icarl      | replay        |  10 |   0.9432 |   0.8408 |      0.1024 |   0.0023 |    0.2309 |     0.5247 |    0.123  |   1      |
| STIN      | MLP    | bic        | replay        |  10 |   0.6329 |   0.5641 |      0.0688 |   0.0588 |    0.0819 |     3.3942 |    0.002  |   0.0449 |
| STIN      | MLP    | bic        | joint_scratch |  10 |   0.6329 |   0.5734 |      0.0595 |   0.0345 |    0.0861 |     1.3659 |    0.002  |   0.0449 |
| STIN      | MLP    | bic        | full          |  10 |   0.6329 |   0.588  |      0.045  |   0.0232 |    0.0691 |     1.1427 |    0.002  |   0.0449 |
| STIN      | CNN1D  | joint      | joint_scratch |  10 |   0.6126 |   0.5726 |      0.04   |   0.0215 |    0.0584 |     1.2512 |    0.0059 |   0.0762 |
| STIN      | CNN1D  | icarl      | replay        |  10 |   0.5514 |   0.5134 |      0.0379 |   0.007  |    0.0699 |     0.7119 |    0.0645 |   0.6445 |
| CuCD-ID   | CNN1D  | bic        | replay        |  10 |   0.8067 |   0.7774 |      0.0293 |  -0.0273 |    0.0873 |     0.2994 |    0.3574 |   1      |
| STIN      | GRU    | joint      | joint_scratch |  10 |   0.6082 |   0.589  |      0.0192 |  -0.0076 |    0.0475 |     0.404  |    0.2266 |   1      |
| STIN      | CNN1D  | bic        | replay        |  10 |   0.5312 |   0.5134 |      0.0178 |  -0.0005 |    0.0369 |     0.5665 |    0.1152 |   1      |
| STIN      | GRU    | icarl      | full          |  10 |   0.59   |   0.5722 |      0.0178 |   0.0035 |    0.0307 |     0.7637 |    0.0449 |   0.4941 |
| STIN      | GRU    | icarl      | replay        |  10 |   0.59   |   0.5803 |      0.0097 |  -0.0109 |    0.0327 |     0.265  |    0.3984 |   1      |
| CuCD-ID   | MLP    | icarl      | full          |  10 |   0.9432 |   0.9529 |     -0.0097 |  -0.0788 |    0.0589 |    -0.0822 |    0.7773 |   1      |
| CuCD-ID   | GRU    | icarl      | replay        |  10 |   0.9209 |   0.9384 |     -0.0176 |  -0.0297 |   -0.0059 |    -0.8734 |    0.0273 |   0.3281 |
| CuCD-ID   | MLP    | icarl      | bic           |  10 |   0.9432 |   0.9626 |     -0.0194 |  -0.0548 |    0.0019 |    -0.3595 |    0.2285 |   1      |
| STIN      | CNN1D  | bic        | icarl         |  10 |   0.5312 |   0.5514 |     -0.0201 |  -0.0605 |    0.0149 |    -0.3128 |    0.3809 |   1      |
| CuCD-ID   | CNN1D  | bic        | icarl         |  10 |   0.8067 |   0.8343 |     -0.0276 |  -0.0772 |    0.0235 |    -0.3267 |    0.3223 |   1      |