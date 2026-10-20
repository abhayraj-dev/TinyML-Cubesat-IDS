# Results Summary

_This file is generated automatically by `experiments/generate_summary_report.py` directly from the CSV/JSON outputs of the experiment scripts -- every number below is a real, computed result, not a placeholder._

## 1. Hyperparameter protocol (STIN prior + per-method/arch validation tuning)

Coarse STIN/MLP/full grid: **160** configurations, scored on validation.

Top 5 by validation score (ACC - max(FM,0)):

|    lr |   ewc_lambda |   buffer_capacity |   replay_frac |    ACC |     BWT |     FM |   score |
|------:|-------------:|------------------:|--------------:|-------:|--------:|-------:|--------:|
| 0.001 |          100 |               128 |           0.9 | 0.6422 | -0.2056 | 0.2056 |  0.4367 |
| 0.003 |          100 |               128 |           0.9 | 0.6541 | -0.2656 | 0.2656 |  0.3885 |
| 0.003 |         5000 |               128 |           0.9 | 0.5956 | -0.25   | 0.25   |  0.3456 |
| 0.003 |         1000 |               128 |           0.9 | 0.5881 | -0.2644 | 0.2644 |  0.3237 |
| 0.001 |          100 |              1024 |           0.7 | 0.6444 | -0.3256 | 0.3256 |  0.3189 |

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

| arch   | method   |   val_score | hparams                                                                      |
|:-------|:---------|------------:|:-----------------------------------------------------------------------------|
| CNN1D  | bic      |      0.611  | {"buffer_capacity": 512, "distill_lambda": 2.0, "lr": 0.01}                  |
| CNN1D  | ewc      |     -0.6849 | {"ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.001}                           |
| CNN1D  | finetune |     -0.6881 | {"epochs": 50, "lr": 0.001}                                                  |
| CNN1D  | full     |      0.2438 | {"buffer_capacity": 256, "ewc_lambda": 100, "lr": 0.01, "replay_frac": 0.9}  |
| CNN1D  | icarl    |      0.2974 | {"buffer_capacity": 512, "distill_lambda": 0.5, "lr": 0.01}                  |
| CNN1D  | lwf      |     -0.727  | {"distill_T": 1.5, "distill_lambda": 0.5, "lr": 0.001}                       |
| CNN1D  | replay   |      0.2469 | {"buffer_capacity": 128, "lr": 0.01, "replay_frac": 0.9}                     |
| GRU    | bic      |      0.7701 | {"buffer_capacity": 512, "distill_lambda": 1.0, "lr": 0.01}                  |
| GRU    | ewc      |     -0.0412 | {"ewc_gamma": 2.0, "ewc_lambda": 5000, "lr": 0.001}                          |
| GRU    | finetune |     -0.7045 | {"epochs": 25, "lr": 0.001}                                                  |
| GRU    | full     |      0.5244 | {"buffer_capacity": 128, "ewc_lambda": 100, "lr": 0.01, "replay_frac": 0.7}  |
| GRU    | icarl    |      0.7597 | {"buffer_capacity": 512, "distill_lambda": 0.5, "lr": 0.01}                  |
| GRU    | lwf      |     -0.674  | {"distill_T": 3.0, "distill_lambda": 2.0, "lr": 0.001}                       |
| GRU    | replay   |      0.7743 | {"buffer_capacity": 512, "lr": 0.01, "replay_frac": 0.7}                     |
| MLP    | bic      |      0.9887 | {"buffer_capacity": 512, "distill_lambda": 1.0, "lr": 0.01}                  |
| MLP    | ewc      |     -0.7496 | {"ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.01}                            |
| MLP    | finetune |     -0.7496 | {"epochs": 25, "lr": 0.01}                                                   |
| MLP    | full     |      0.972  | {"buffer_capacity": 512, "ewc_lambda": 1000, "lr": 0.01, "replay_frac": 0.7} |
| MLP    | icarl    |      0.9809 | {"buffer_capacity": 512, "distill_lambda": 1.0, "lr": 0.01}                  |
| MLP    | lwf      |     -0.7474 | {"distill_T": 2.0, "distill_lambda": 0.5, "lr": 0.001}                       |
| MLP    | replay   |      0.9094 | {"buffer_capacity": 512, "lr": 0.003, "replay_frac": 0.9}                    |

Every main/Phase-2 run below uses its matching frozen config; test data is never used for selection.

## 2. Main experiment matrix

Total individual runs: **540** (2 datasets x 3 architectures x 9 methods x 10 seeds).


### CuCD-ID

**Average Accuracy (ACC), mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.7451 | 0.2602 |     0.2657 | 0.624  |  0.704  |  0.9197 |          0.8687 | 0.2501 |   0.6187 |
| GRU    | 0.836  | 0.4016 |     0.2608 | 0.7168 |  0.8949 |  0.9673 |          0.9793 | 0.2981 |   0.8475 |
| MLP    | 0.9564 | 0.25   |     0.25   | 0.8634 |  0.9413 |  0.9948 |          0.9942 | 0.2584 |   0.9194 |

**Backward Transfer (BWT), mean over 10 seeds (negative = forgetting):**

| arch   |     bic |     ewc |   finetune |    full |   icarl |   joint |   joint_scratch |     lwf |   replay |
|:-------|--------:|--------:|-----------:|--------:|--------:|--------:|----------------:|--------:|---------:|
| CNN1D  | -0.2709 | -0.9675 |    -0.9725 | -0.483  | -0.3876 | -0.0332 |         -0.0679 | -0.9809 |  -0.5005 |
| GRU    | -0.1834 | -0.4648 |    -0.9805 | -0.3575 | -0.1236 | -0.0035 |         -0.007  | -0.9007 |  -0.1894 |
| MLP    | -0.0347 | -0.9998 |    -0.9998 | -0.179  | -0.0761 | -0.002  |         -0.0014 | -0.9887 |  -0.104  |

**Macro-F1 (primary, BiC-corrected), mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.7681 | 0.0963 |     0.1077 | 0.6364 |  0.7098 |  0.9298 |          0.8802 | 0.0726 |   0.6308 |
| GRU    | 0.8483 | 0.3885 |     0.109  | 0.7408 |  0.8958 |  0.9714 |          0.9815 | 0.1612 |   0.8718 |
| MLP    | 0.9626 | 0.0726 |     0.0726 | 0.873  |  0.9432 |  0.9953 |          0.9947 | 0.0895 |   0.9279 |

**Balanced Accuracy, mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.773  | 0.2142 |     0.2218 | 0.6694 |  0.7268 |  0.9345 |          0.8856 | 0.2002 |   0.6894 |
| GRU    | 0.8491 | 0.4283 |     0.2207 | 0.7553 |  0.8879 |  0.9726 |          0.9821 | 0.2607 |   0.8723 |
| MLP    | 0.9624 | 0.2    |     0.2    | 0.8875 |  0.9487 |  0.9956 |          0.995  | 0.2118 |   0.9334 |

**Binary detection (1-p(normal), CuCD-ID only; STIN has no benign class):**

| arch   | method        |   DET_AUROC_mean |   DET_AUPR_mean |   DET_FPR95_mean |
|:-------|:--------------|-----------------:|----------------:|-----------------:|
| CNN1D  | bic           |           0.9998 |          1      |           0.0003 |
| CNN1D  | ewc           |           0.9872 |          0.9964 |           0.0607 |
| CNN1D  | finetune      |           0.9904 |          0.9972 |           0.0485 |
| CNN1D  | full          |           0.9999 |          1      |           0.0005 |
| CNN1D  | icarl         |           0.9997 |          0.9999 |           0.0012 |
| CNN1D  | joint         |           0.9999 |          1      |           0.0001 |
| CNN1D  | joint_scratch |           0.9978 |          0.9994 |           0.0064 |
| CNN1D  | lwf           |           0.9705 |          0.9918 |           0.1644 |
| CNN1D  | replay        |           0.9995 |          0.9998 |           0.0027 |
| GRU    | bic           |           0.9908 |          0.9969 |           0.0436 |
| GRU    | ewc           |           0.9015 |          0.9667 |           0.407  |
| GRU    | finetune      |           0.6215 |          0.8415 |           0.7891 |
| GRU    | full          |           0.9928 |          0.9975 |           0.0251 |
| GRU    | icarl         |           0.9939 |          0.9977 |           0.0215 |
| GRU    | joint         |           0.9984 |          0.9981 |           0.0033 |
| GRU    | joint_scratch |           0.999  |          0.9997 |           0.0025 |
| GRU    | lwf           |           0.721  |          0.8931 |           0.6683 |
| GRU    | replay        |           0.9971 |          0.9988 |           0.0076 |
| MLP    | bic           |           0.9999 |          1      |           0.0005 |
| MLP    | ewc           |           0.7968 |          0.8411 |           0.5934 |
| MLP    | finetune      |           0.7968 |          0.8411 |           0.5934 |
| MLP    | full          |           0.9998 |          0.9998 |           0.0006 |
| MLP    | icarl         |           0.9999 |          1      |           0.0003 |
| MLP    | joint         |           1      |          0.9998 |           0.0001 |
| MLP    | joint_scratch |           1      |          1      |           0.0001 |
| MLP    | lwf           |           0.9998 |          1      |           0.0004 |
| MLP    | replay        |           1      |          1      |           0.0001 |

### STIN

**Average Accuracy (ACC), mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.5776 | 0.2239 |     0.2513 | 0.518  |  0.5827 |  0.5432 |          0.4099 | 0.242  |   0.5711 |
| GRU    | 0.6427 | 0.307  |     0.3078 | 0.6236 |  0.6552 |  0.653  |          0.5792 | 0.3022 |   0.6206 |
| MLP    | 0.68   | 0.2822 |     0.2873 | 0.6464 |  0.6637 |  0.6499 |          0.6432 | 0.2833 |   0.6242 |

**Backward Transfer (BWT), mean over 10 seeds (negative = forgetting):**

| arch   |     bic |     ewc |   finetune |    full |   icarl |   joint |   joint_scratch |     lwf |   replay |
|:-------|--------:|--------:|-----------:|--------:|--------:|--------:|----------------:|--------:|---------:|
| CNN1D  | -0.2476 | -0.7526 |    -0.8177 | -0.341  | -0.3952 |  0.0505 |         -0.0597 | -0.7718 |  -0.3238 |
| GRU    | -0.2345 | -0.6248 |    -0.6737 | -0.3803 | -0.3766 | -0.0411 |         -0.0894 | -0.5975 |  -0.4037 |
| MLP    | -0.1766 | -0.9447 |    -0.9447 | -0.3609 | -0.3758 | -0.0403 |         -0.0472 | -0.9231 |  -0.3722 |

**Macro-F1 (primary, BiC-corrected), mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.5246 | 0.0985 |     0.1232 | 0.4411 |  0.5095 |  0.4589 |          0.3077 | 0.1136 |   0.4991 |
| GRU    | 0.5826 | 0.188  |     0.1831 | 0.5554 |  0.5821 |  0.5904 |          0.4931 | 0.1815 |   0.5381 |
| MLP    | 0.6329 | 0.1576 |     0.1632 | 0.5751 |  0.5928 |  0.5955 |          0.587  | 0.1596 |   0.5481 |

**Balanced Accuracy, mean over 10 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.5776 | 0.2239 |     0.2513 | 0.518  |  0.5827 |  0.5432 |          0.4099 | 0.242  |   0.5711 |
| GRU    | 0.6427 | 0.307  |     0.3078 | 0.6236 |  0.6552 |  0.653  |          0.5792 | 0.3022 |   0.6206 |
| MLP    | 0.68   | 0.2822 |     0.2873 | 0.6464 |  0.6637 |  0.6499 |          0.6432 | 0.2833 |   0.6242 |

## 3. Memory: persistent deployment vs peak update RAM (method=full)

| dataset   | arch   |   persistent_total_KB |   peak_total_KB |   macs_per_forward |   persistent_model_KB |   persistent_cmmu_KB |
|:----------|:-------|----------------------:|----------------:|-------------------:|----------------------:|---------------------:|
| CuCD-ID   | CNN1D  |                  7.31 |           41.43 |               4406 |                  0.26 |                 7.05 |
| CuCD-ID   | GRU    |                  5.17 |           35    |               4010 |                  0.43 |                 4.74 |
| CuCD-ID   | MLP    |                 18.59 |           99.43 |               1296 |                  1.32 |                17.27 |
| STIN      | CNN1D  |                  9.53 |           53.19 |               6030 |                  0.31 |                 9.22 |
| STIN      | GRU    |                  6.53 |           42.74 |               3990 |                  0.51 |                 6.03 |
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
| CuCD-ID   | CNN1D  |                32 | 0.5928 | -0.5284 | 0.5284 |
| CuCD-ID   | CNN1D  |                64 | 0.5966 | -0.5235 | 0.5235 |
| CuCD-ID   | CNN1D  |               128 | 0.6162 | -0.492  | 0.492  |
| CuCD-ID   | CNN1D  |               256 | 0.624  | -0.483  | 0.4831 |
| CuCD-ID   | CNN1D  |               512 | 0.5966 | -0.5205 | 0.5205 |
| CuCD-ID   | CNN1D  |              1024 | 0.6078 | -0.5046 | 0.5047 |
| CuCD-ID   | GRU    |                32 | 0.6579 | -0.4396 | 0.4396 |
| CuCD-ID   | GRU    |                64 | 0.6434 | -0.4559 | 0.4559 |
| CuCD-ID   | GRU    |               128 | 0.7168 | -0.3575 | 0.3575 |
| CuCD-ID   | GRU    |               256 | 0.6686 | -0.4198 | 0.4198 |
| CuCD-ID   | GRU    |               512 | 0.7261 | -0.345  | 0.345  |
| CuCD-ID   | GRU    |              1024 | 0.7041 | -0.3759 | 0.3759 |
| CuCD-ID   | MLP    |                32 | 0.6807 | -0.4243 | 0.4244 |
| CuCD-ID   | MLP    |                64 | 0.808  | -0.2542 | 0.2542 |
| CuCD-ID   | MLP    |               128 | 0.7407 | -0.343  | 0.343  |
| CuCD-ID   | MLP    |               256 | 0.8375 | -0.2138 | 0.2138 |
| CuCD-ID   | MLP    |               512 | 0.8634 | -0.179  | 0.179  |
| CuCD-ID   | MLP    |              1024 | 0.8304 | -0.2228 | 0.2228 |
| STIN      | CNN1D  |                32 | 0.5011 | -0.3815 | 0.3815 |
| STIN      | CNN1D  |                64 | 0.4959 | -0.4172 | 0.4172 |
| STIN      | CNN1D  |               128 | 0.517  | -0.3381 | 0.3381 |
| STIN      | CNN1D  |               256 | 0.518  | -0.341  | 0.341  |
| STIN      | CNN1D  |               512 | 0.5024 | -0.3832 | 0.3832 |
| STIN      | CNN1D  |              1024 | 0.4906 | -0.403  | 0.403  |
| STIN      | GRU    |                32 | 0.5865 | -0.419  | 0.419  |
| STIN      | GRU    |                64 | 0.6077 | -0.3999 | 0.3999 |
| STIN      | GRU    |               128 | 0.6236 | -0.3803 | 0.3803 |
| STIN      | GRU    |               256 | 0.5973 | -0.3877 | 0.3877 |
| STIN      | GRU    |               512 | 0.597  | -0.3935 | 0.3935 |
| STIN      | GRU    |              1024 | 0.6163 | -0.4019 | 0.4019 |
| STIN      | MLP    |                32 | 0.6121 | -0.4181 | 0.4181 |
| STIN      | MLP    |                64 | 0.6528 | -0.3494 | 0.3494 |
| STIN      | MLP    |               128 | 0.6497 | -0.3383 | 0.3383 |
| STIN      | MLP    |               256 | 0.6552 | -0.3566 | 0.3566 |
| STIN      | MLP    |               512 | 0.6464 | -0.3609 | 0.3609 |
| STIN      | MLP    |              1024 | 0.6535 | -0.3776 | 0.3776 |

Per-architecture ACC range across all tested buffer capacities:

| dataset   | arch   |    min |    max |   span |
|:----------|:-------|-------:|-------:|-------:|
| CuCD-ID   | CNN1D  | 0.5928 | 0.624  | 0.0312 |
| CuCD-ID   | GRU    | 0.6434 | 0.7261 | 0.0827 |
| CuCD-ID   | MLP    | 0.6807 | 0.8634 | 0.1827 |
| STIN      | CNN1D  | 0.4906 | 0.518  | 0.0274 |
| STIN      | GRU    | 0.5865 | 0.6236 | 0.0372 |
| STIN      | MLP    | 0.6121 | 0.6552 | 0.0431 |

### 5.2 EWC lambda sensitivity (both datasets x all 3 architectures, method=full)

| dataset   | arch   |   ewc_lambda |    ACC |     BWT |     FM |
|:----------|:-------|-------------:|-------:|--------:|-------:|
| CuCD-ID   | CNN1D  |            0 | 0.6102 | -0.5138 | 0.5139 |
| CuCD-ID   | CNN1D  |           10 | 0.6256 | -0.4873 | 0.4875 |
| CuCD-ID   | CNN1D  |          100 | 0.624  | -0.483  | 0.4831 |
| CuCD-ID   | CNN1D  |         1000 | 0.5779 | -0.5406 | 0.5406 |
| CuCD-ID   | CNN1D  |         5000 | 0.5632 | -0.5545 | 0.5545 |
| CuCD-ID   | CNN1D  |        20000 | 0.5452 | -0.5794 | 0.5794 |
| CuCD-ID   | CNN1D  |        50000 | 0.5431 | -0.5823 | 0.5823 |
| CuCD-ID   | GRU    |            0 | 0.7724 | -0.2897 | 0.2897 |
| CuCD-ID   | GRU    |           10 | 0.7646 | -0.2999 | 0.2999 |
| CuCD-ID   | GRU    |          100 | 0.7168 | -0.3575 | 0.3575 |
| CuCD-ID   | GRU    |         1000 | 0.5949 | -0.5032 | 0.5032 |
| CuCD-ID   | GRU    |         5000 | 0.5081 | -0.614  | 0.614  |
| CuCD-ID   | GRU    |        20000 | 0.4895 | -0.6202 | 0.6202 |
| CuCD-ID   | GRU    |        50000 | 0.5026 | -0.592  | 0.592  |
| CuCD-ID   | MLP    |            0 | 0.84   | -0.211  | 0.211  |
| CuCD-ID   | MLP    |           10 | 0.8207 | -0.2362 | 0.2362 |
| CuCD-ID   | MLP    |          100 | 0.8565 | -0.1881 | 0.1881 |
| CuCD-ID   | MLP    |         1000 | 0.8634 | -0.179  | 0.179  |
| CuCD-ID   | MLP    |         5000 | 0.8829 | -0.1536 | 0.1537 |
| CuCD-ID   | MLP    |        20000 | 0.8667 | -0.1752 | 0.1752 |
| CuCD-ID   | MLP    |        50000 | 0.8814 | -0.1555 | 0.1555 |
| STIN      | CNN1D  |            0 | 0.579  | -0.3235 | 0.3235 |
| STIN      | CNN1D  |           10 | 0.5176 | -0.3648 | 0.3648 |
| STIN      | CNN1D  |          100 | 0.518  | -0.341  | 0.341  |
| STIN      | CNN1D  |         1000 | 0.5062 | -0.3521 | 0.3521 |
| STIN      | CNN1D  |         5000 | 0.5089 | -0.3555 | 0.3555 |
| STIN      | CNN1D  |        20000 | 0.5017 | -0.3623 | 0.3623 |
| STIN      | CNN1D  |        50000 | 0.5013 | -0.3658 | 0.3658 |
| STIN      | GRU    |            0 | 0.6223 | -0.3728 | 0.3728 |
| STIN      | GRU    |           10 | 0.6289 | -0.3701 | 0.3701 |
| STIN      | GRU    |          100 | 0.6236 | -0.3803 | 0.3803 |
| STIN      | GRU    |         1000 | 0.5728 | -0.4022 | 0.4022 |
| STIN      | GRU    |         5000 | 0.5228 | -0.4446 | 0.4446 |
| STIN      | GRU    |        20000 | 0.4639 | -0.4878 | 0.4878 |
| STIN      | GRU    |        50000 | 0.4418 | -0.5126 | 0.5126 |
| STIN      | MLP    |            0 | 0.6136 | -0.4344 | 0.4344 |
| STIN      | MLP    |           10 | 0.6492 | -0.3747 | 0.3747 |
| STIN      | MLP    |          100 | 0.6568 | -0.3556 | 0.3556 |
| STIN      | MLP    |         1000 | 0.6464 | -0.3609 | 0.3609 |
| STIN      | MLP    |         5000 | 0.6565 | -0.3679 | 0.3679 |
| STIN      | MLP    |        20000 | 0.6503 | -0.3742 | 0.3742 |
| STIN      | MLP    |        50000 | 0.6598 | -0.3598 | 0.3598 |

Per-(dataset, architecture) ACC range across all tested lambdas:

| dataset   | arch   |    min |    max |   span |
|:----------|:-------|-------:|-------:|-------:|
| CuCD-ID   | CNN1D  | 0.5431 | 0.6256 | 0.0825 |
| CuCD-ID   | GRU    | 0.4895 | 0.7724 | 0.2829 |
| CuCD-ID   | MLP    | 0.8207 | 0.8829 | 0.0622 |
| STIN      | CNN1D  | 0.5013 | 0.579  | 0.0777 |
| STIN      | GRU    | 0.4418 | 0.6289 | 0.1871 |
| STIN      | MLP    | 0.6136 | 0.6598 | 0.0462 |

MLP's span is small on both datasets, while GRU and CNN1D both show a much larger span, and in both cases ACC *decreases* as lambda increases (visible directly in the per-lambda table above and in `figures/ewc_lambda_sensitivity.png`). This is why final numbers use frozen per-method x per-architecture validation configs instead of one shared configuration.

EWC here is online new-task Fisher (Fisher estimated from current-task train/val samples with the running estimate retained, not re-estimated from old-task data; update `F_new = γ·F_old + F_current`). Frozen per-architecture γ weighting: MLP γ=0.5, GRU γ=2.0, CNN1D γ=0.5. γ is treated as a validation-tuned weighting parameter rather than a conventional decay factor (GRU γ=2.0 amplifies retained Fisher by validation selection; a γ≤1 restriction would be stricter but would require rerunning EWC experiments).


### 5.3 Task-order sensitivity, legacy 3-perm (CuCD-ID, MLP, method=full)

_Legacy confounded design (budgets move with the task); superseded by the disentangled 12-permutation study in §7.3. Retained for provenance only._

| order        |   ACC_mean |   ACC_std |   BWT_mean |   BWT_std |
|:-------------|-----------:|----------:|-----------:|----------:|
| [0, 1, 2, 3] |     0.8634 |    0.1474 |    -0.179  |    0.1971 |
| [1, 3, 0, 2] |     0.9765 |    0.0266 |    -0.0281 |    0.0359 |
| [3, 2, 1, 0] |     0.9959 |    0.0023 |    -0.0036 |    0.0024 |

ACC spread across the 3 legacy orderings: **0.1325** (range 0.8634 - 0.9959). See §7.3 for the disentangled measurement (spread 0.0495).


## 6. Per-class F1 -- direct evidence of forgetting

Section 2's ACC/BWT/FM numbers are task-level aggregates; this section makes the same phenomenon visible at the level of individual attack classes by evaluating the final trained model (all classes seen so far, single shared head) and computing a per-class precision/recall/F1 breakdown. See `figures/per_class_f1_*.png` and `figures/confusion_matrix_*.png` for the visual version.


### 6.1 STIN / GRU, mean F1 per task-of-origin

| task_name     |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:--------------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| T0_flood_base | 0.7233 | 0.2029 |     0.1709 | 0.588  |  0.5305 |  0.7588 |          0.7335 | 0.1619 |   0.579  |
| T1_flood_ext  | 0.5367 | 0.0143 |     0      | 0.4831 |  0.5652 |  0.7388 |          0.5218 | 0.0456 |   0.3969 |
| T2_app_layer  | 0.4876 | 0.3469 |     0.3784 | 0.5952 |  0.6505 |  0.2735 |          0.2239 | 0.3371 |   0.6383 |

### 6.2 CuCD-ID / MLP, mean F1 per task-of-origin

| task_name             |    bic |   ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:----------------------|-------:|------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| T0_launch             | 0.9917 | 0     |      0     | 0.984  |  0.9891 |  0.9968 |          0.9966 | 0.0392 |   0.9928 |
| T1_data_injection     | 0.9241 | 0     |      0     | 0.6318 |  0.8276 |  0.9961 |          0.9933 | 0      |   0.7797 |
| T2_defence_impairment | 0.9868 | 0     |      0     | 0.9251 |  0.9916 |  0.9955 |          0.997  | 0.0001 |   0.9877 |
| T3_storage_exhaustion | 0.9184 | 0.363 |      0.363 | 0.8401 |  0.9185 |  0.9914 |          0.9898 | 0.3689 |   0.8863 |

For CuCD-ID, `finetune` and `ewc` collapse every task except the most-recently-learned one to *exactly* F1 = 0.000 -- the model stops predicting those classes at all. For STIN, the same collapse occurs but lands on whichever task ends up dominating the shared head's bias after the last update (not always precisely the final task, and not always exactly 0.000 for every other task -- see the table above), which is itself expected: STIN's per-task class count (3 new classes/task) and harder feature separability make the 'recency bias' target noisier across seeds than CuCD-ID's cleaner 1-2-classes/task structure. In both datasets, `replay`, `full`, and `joint` retain non-zero, generally high F1 across every task -- the direct, per-class confirmation of the class-incremental collapse summarised by the task-level ACC/BWT/FM numbers in Section 2.


## 7. Phase-2 evidence (all numbers computed from their CSVs)


### 7.1 Few-shot uplink sweep, CuCD-ID (RQ4)

| arch   | method   |      5 |     10 |     25 |     50 |    100 |    150 |    300 |
|:-------|:---------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| GRU    | icarl    | 0.4236 | 0.4549 | 0.5162 | 0.6978 | 0.8181 | 0.902  | 0.936  |
| GRU    | replay   | 0.4623 | 0.4756 | 0.4862 | 0.5285 | 0.6566 | 0.8561 | 0.9641 |
| MLP    | bic      | 0.4692 | 0.6944 | 0.6667 | 0.8103 | 0.9755 | 0.9873 | 0.9767 |
| MLP    | full     | 0.4445 | 0.4792 | 0.4792 | 0.5866 | 0.7709 | 0.9334 | 0.9897 |
| MLP    | icarl    | 0.4672 | 0.6268 | 0.5758 | 0.7736 | 0.976  | 0.9913 | 0.9912 |
| MLP    | replay   | 0.521  | 0.4931 | 0.4941 | 0.6666 | 0.7614 | 0.9347 | 0.994  |

Adaptation uplink KB for T1..T3 per K: K=5: 0.3KB, K=10: 0.7KB, K=25: 1.7KB, K=50: 3.4KB, K=100: 6.7KB, K=150: 10.1KB, K=300: 20.2KB.

Total uplink KB **including onboard calibration data** (BiC/Fisher val samples, Option A) per K: K=5: 10.4KB, K=10: 10.8KB, K=25: 11.8KB, K=50: 13.5KB, K=100: 16.9KB, K=150: 20.2KB, K=300: 30.3KB.

MLP @K=50 (knee): bic=0.810, full=0.587, icarl=0.774, replay=0.667. 
MLP @K=5 (extreme): bic=0.469, full=0.445, icarl=0.467, replay=0.521.


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
| fixed_position  | replay   | 0.9766 | 0.0291 |
| moves_with_task | icarl    | 0.9769 | 0.0242 |
| moves_with_task | replay   | 0.9784 | 0.0299 |

Aggregate difference between budget modes is small (fixed_position=0.973 vs moves_with_task=0.978; spread of per-order means across 12 permutations: **0.0591**), although individual runs vary more (raw ACC range **0.1761**). The legacy confounded headline is retired.


### 7.4 IID vs forward-temporal-purged, CuCD-ID/MLP

| method        | protocol        |   mean |    std |
|:--------------|:----------------|-------:|-------:|
| bic           | iid             | 0.9564 | 0.0526 |
| bic           | temporal_purged | 0.2788 | 0.131  |
| ewc           | iid             | 0.25   | 0      |
| ewc           | temporal_purged | 0.234  | 0.0475 |
| finetune      | iid             | 0.25   | 0      |
| finetune      | temporal_purged | 0.234  | 0.0475 |
| full          | iid             | 0.8634 | 0.1474 |
| full          | temporal_purged | 0.3056 | 0.0845 |
| icarl         | iid             | 0.9413 | 0.0874 |
| icarl         | temporal_purged | 0.2292 | 0.1177 |
| joint         | iid             | 0.9948 | 0.0024 |
| joint         | temporal_purged | 0.1496 | 0.0629 |
| joint_scratch | iid             | 0.9942 | 0.0066 |
| joint_scratch | temporal_purged | 0.159  | 0.0623 |
| lwf           | iid             | 0.2584 | 0.0251 |
| lwf           | temporal_purged | 0.2051 | 0.1366 |
| replay        | iid             | 0.9194 | 0.08   |
| replay        | temporal_purged | 0.2099 | 0.0918 |

IID = random-split benchmark; temporal-purged = forward temporal generalization under distribution shift with reduced window overlap (separate estimands, not interpolated). Purge: 0.02 rad on TimeRadians (~22s equivalent). See DATALOG.md for the drift diagnosis.


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
| joint_scratch | drop_0.1     |           0.9381 |
| joint_scratch | drop_0.3     |           0.802  |
| joint_scratch | gauss_0.1    |           0.997  |
| joint_scratch | gauss_0.5    |           0.9755 |
| joint_scratch | int8_sim     |           0.9975 |
| joint_scratch | nowindow     |           0.7434 |
| replay        | clean        |           0.9685 |
| replay        | drop_0.1     |           0.9128 |
| replay        | drop_0.3     |           0.784  |
| replay        | gauss_0.1    |           0.967  |
| replay        | gauss_0.5    |           0.93   |
| replay        | int8_sim     |           0.9711 |
| replay        | nowindow     |           0.6834 |

Mean pooled-ACC drop vs clean (all methods): drop_0.1=+0.059, drop_0.3=+0.192, gauss_0.1=+0.001, gauss_0.5=+0.033, int8_sim=-0.001, nowindow=+0.285.


### 7.6 Activation memory ledger (analytic, batch=1, SIM_ = host-simulated)

| arch   |   act_fwd_floats |   SIM_act_fwd_KB |   SIM_act_train_KB | seq_len   |
|:-------|-----------------:|-----------------:|-------------------:|:----------|
| MLP    |              101 |            0.395 |              0.789 |           |
| GRU    |              775 |            3.027 |              6.055 | 11.0000   |
| CNN1D  |              719 |            2.809 |              5.617 |           |



### 7.7 Exact paired permutation + Holm (matched seeds, ACC): 4/9 headline comparisons significant at Holm p<0.05

| dataset   | arch   | method_a   | method_b   |   n |   mean_a |   mean_b |   mean_diff |   ci_low |   ci_high |   cohens_d |   p_exact |   p_holm |
|:----------|:-------|:-----------|:-----------|----:|---------:|---------:|------------:|---------:|----------:|-----------:|----------:|---------:|
| CuCD-ID   | MLP    | replay     | finetune   |  10 |   0.9194 |   0.25   |      0.6694 |   0.6195 |    0.7114 |     8.3669 |    0.002  |   0.0176 |
| CuCD-ID   | MLP    | full       | finetune   |  10 |   0.8634 |   0.25   |      0.6134 |   0.5252 |    0.6946 |     4.1617 |    0.002  |   0.0176 |
| CuCD-ID   | GRU    | icarl      | full       |  10 |   0.8949 |   0.7168 |      0.1781 |   0.1238 |    0.232  |     1.905  |    0.002  |   0.0176 |
| CuCD-ID   | CNN1D  | bic        | replay     |  10 |   0.7451 |   0.6187 |      0.1263 |   0.0816 |    0.1705 |     1.6414 |    0.0039 |   0.0234 |
| CuCD-ID   | MLP    | icarl      | full       |  10 |   0.9413 |   0.8634 |      0.078  |  -0.0121 |    0.1595 |     0.5289 |    0.1309 |   0.4297 |
| CuCD-ID   | GRU    | icarl      | replay     |  10 |   0.8949 |   0.8475 |      0.0475 |   0.0033 |    0.0984 |     0.5826 |    0.0898 |   0.4297 |
| CuCD-ID   | CNN1D  | bic        | icarl      |  10 |   0.7451 |   0.704  |      0.041  |   0.0041 |    0.079  |     0.6214 |    0.0859 |   0.4297 |
| CuCD-ID   | MLP    | icarl      | replay     |  10 |   0.9413 |   0.9194 |      0.0219 |  -0.0412 |    0.0802 |     0.2101 |    0.4863 |   0.7305 |
| CuCD-ID   | MLP    | icarl      | bic        |  10 |   0.9413 |   0.9564 |     -0.0151 |  -0.045  |    0.0042 |    -0.3321 |    0.3652 |   0.7305 |