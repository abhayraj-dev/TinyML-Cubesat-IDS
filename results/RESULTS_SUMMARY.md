# Results Summary

_This file is generated automatically by `experiments/generate_summary_report.py` directly from the CSV/JSON outputs of the experiment scripts -- every number below is a real, computed result, not a placeholder._


**Data provenance.** CuCD-ID is the labelled NOS3/cFS software-in-the-loop CubeSat telemetry dataset of Fayyaz et al. (Data in Brief 2026, doi:10.1016/j.dib.2026.112598; Mendeley doi:10.17632/7n2d42pm3n.3); all telemetry is simulated, not captured from an orbiting spacecraft. STIN (SAT20/TER20) is the satellite/terrestrial flow dataset of Li et al. (IEEE Access 2020, doi:10.1109/ACCESS.2020.3041641; github.com/kun9717/STIN-data-set) and is used as a cross-domain stress test. Headline tables below use the IID random split; Sections 5.3 and 7.4 quantify the resulting leakage/feature-reliance caveats.


**Scope and claims.** This is a resource-aware benchmark and negative-result report, not a claim that a new combined method wins. `full` is evaluated as a baseline combination; the main matrix, temporal-purged study, feature-provenance ablation, feature-order study, and task-order study are separate estimands with different claim boundaries.

## 1. Hyperparameter protocol (STIN prior + per-method/arch validation tuning)

Coarse STIN/MLP/full grid: **160** configurations, scored on validation.

Top 5 by validation score (ACC - max(FM,0)):

|    lr |   ewc_lambda |   buffer_capacity |   replay_frac |    ACC |     BWT |     FM |   score |
|------:|-------------:|------------------:|--------------:|-------:|--------:|-------:|--------:|
| 0.001 |         1000 |               256 |           0.9 | 0.6822 | -0.1456 | 0.1456 |  0.5367 |
| 0.001 |          100 |              1024 |           0.9 | 0.6926 | -0.1633 | 0.1633 |  0.5293 |
| 0.001 |          100 |               128 |           0.9 | 0.6511 | -0.1856 | 0.1856 |  0.4656 |
| 0.003 |          100 |              1024 |           0.7 | 0.7007 | -0.2378 | 0.2378 |  0.463  |
| 0.003 |          100 |              1024 |           0.9 | 0.6985 | -0.2478 | 0.2478 |  0.4507 |

**Coarse STIN prior (NOT the final configs):**

```json
{
  "lr": 0.001,
  "ewc_lambda": 1000.0,
  "buffer_capacity": 256,
  "replay_frac": 0.9
}
```


**Final frozen configs** (`best_hparams_per_method_arch.json`, per method x architecture, tuned on CuCD-ID validation, 3 seeds each):

| arch   | method        |   val_score | hparams                                                                                                     |
|:-------|:--------------|------------:|:------------------------------------------------------------------------------------------------------------|
| CNN1D  | bic           |      0.7702 | {"buffer_capacity": 512, "distill_lambda": 0.5, "epochs": 50, "lr": 0.01}                                   |
| CNN1D  | ewc           |     -0.7252 | {"epochs": 25, "ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.001}                                            |
| CNN1D  | finetune      |     -0.6931 | {"epochs": 50, "lr": 0.001}                                                                                 |
| CNN1D  | full          |      0.3061 | {"buffer_capacity": 512, "epochs": 50, "ewc_gamma": 1.0, "ewc_lambda": 100, "lr": 0.01, "replay_frac": 0.7} |
| CNN1D  | icarl         |      0.5342 | {"buffer_capacity": 512, "distill_lambda": 0.5, "epochs": 50, "lr": 0.01}                                   |
| CNN1D  | joint         |      0.9919 | {"epochs": 50, "lr": 0.01}                                                                                  |
| CNN1D  | joint_scratch |      0.9696 | {"epochs": 50, "lr": 0.01}                                                                                  |
| CNN1D  | lwf           |     -0.7252 | {"distill_T": 1.5, "distill_lambda": 0.5, "epochs": 25, "lr": 0.001}                                        |
| CNN1D  | replay        |      0.5628 | {"buffer_capacity": 128, "epochs": 50, "lr": 0.01, "replay_frac": 0.9}                                      |
| GRU    | bic           |      0.9119 | {"buffer_capacity": 512, "distill_lambda": 0.5, "epochs": 50, "lr": 0.01}                                   |
| GRU    | ewc           |      0.0894 | {"epochs": 25, "ewc_gamma": 1.0, "ewc_lambda": 5000, "lr": 0.001}                                           |
| GRU    | finetune      |     -0.7339 | {"epochs": 25, "lr": 0.001}                                                                                 |
| GRU    | full          |      0.1166 | {"buffer_capacity": 512, "epochs": 25, "ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.01, "replay_frac": 0.7} |
| GRU    | icarl         |      0.8443 | {"buffer_capacity": 512, "distill_lambda": 0.5, "epochs": 50, "lr": 0.01}                                   |
| GRU    | joint         |      0.9701 | {"epochs": 25, "lr": 0.003}                                                                                 |
| GRU    | joint_scratch |      0.9945 | {"epochs": 50, "lr": 0.003}                                                                                 |
| GRU    | lwf           |     -0.6471 | {"distill_T": 3.0, "distill_lambda": 2.0, "epochs": 25, "lr": 0.001}                                        |
| GRU    | replay        |      0.8555 | {"buffer_capacity": 512, "epochs": 50, "lr": 0.01, "replay_frac": 0.9}                                      |
| MLP    | bic           |      0.9919 | {"buffer_capacity": 512, "distill_lambda": 1.0, "epochs": 50, "lr": 0.003}                                  |
| MLP    | ewc           |     -0.7046 | {"epochs": 50, "ewc_gamma": 0.5, "ewc_lambda": 100, "lr": 0.001}                                            |
| MLP    | finetune      |     -0.7468 | {"epochs": 25, "lr": 0.001}                                                                                 |
| MLP    | full          |      0.9706 | {"buffer_capacity": 256, "epochs": 50, "ewc_gamma": 1.0, "ewc_lambda": 100, "lr": 0.01, "replay_frac": 0.7} |
| MLP    | icarl         |      0.9853 | {"buffer_capacity": 512, "distill_lambda": 0.5, "epochs": 50, "lr": 0.003}                                  |
| MLP    | joint         |      0.9964 | {"epochs": 50, "lr": 0.001}                                                                                 |
| MLP    | joint_scratch |      0.9956 | {"epochs": 25, "lr": 0.003}                                                                                 |
| MLP    | lwf           |     -0.7383 | {"distill_T": 1.5, "distill_lambda": 1.0, "epochs": 50, "lr": 0.001}                                        |
| MLP    | replay        |      0.9814 | {"buffer_capacity": 512, "epochs": 50, "lr": 0.003, "replay_frac": 0.9}                                     |

Every main/Phase-2 run below uses its matching frozen config; test data is never used for selection.


**Tuning scope.** All final method x architecture configs are tuned on CuCD-ID validation only; STIN is a cross-dataset transfer test. Tuning seeds 0-2 overlap the confirmatory matrix seeds 0-14. Because scoring uses the validation selection half and never test rows, this is not test leakage, but the seeds are not fully independent. The `full` grid is smaller than the single-method grids (see `experiments/tuning_grids.py`).

## 2. Main experiment matrix

Total individual runs: **810** (2 datasets x 3 architectures x 9 methods x 15 seeds).


### CuCD-ID

**Average Accuracy (ACC), mean over 15 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.8729 | 0.25   |     0.2797 | 0.6194 |  0.796  |  0.9917 |          0.984  | 0.25   |   0.7089 |
| GRU    | 0.9195 | 0.4384 |     0.2621 | 0.6275 |  0.912  |  0.9683 |          0.9886 | 0.2961 |   0.9224 |
| MLP    | 0.9923 | 0.2744 |     0.2625 | 0.9531 |  0.9903 |  0.9959 |          0.9966 | 0.2588 |   0.9894 |

**Backward Transfer (BWT), mean over 15 seeds (negative = forgetting):**

| arch   |     bic |     ewc |   finetune |    full |   icarl |   joint |   joint_scratch |     lwf |   replay |
|:-------|--------:|--------:|-----------:|--------:|--------:|--------:|----------------:|--------:|---------:|
| CNN1D  | -0.0856 | -0.9784 |    -0.9537 | -0.4905 | -0.2668 | -0.0024 |         -0.0064 | -0.9784 |  -0.3837 |
| GRU    | -0.05   | -0.4033 |    -0.978  | -0.4681 | -0.0997 | -0.0093 |         -0.0031 | -0.8956 |  -0.0899 |
| MLP    | -0.0056 | -0.9674 |    -0.9831 | -0.0602 | -0.0109 | -0.0015 |         -0.0018 | -0.9882 |  -0.0113 |

**Macro-F1 (primary, BiC-corrected), mean over 15 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.8857 | 0.0725 |     0.1331 | 0.6343 |  0.8186 |  0.9923 |          0.9852 | 0.0722 |   0.7269 |
| GRU    | 0.9226 | 0.4386 |     0.104  | 0.6438 |  0.907  |  0.9723 |          0.9897 | 0.1644 |   0.9343 |
| MLP    | 0.9924 | 0.1067 |     0.094  | 0.9562 |  0.9893 |  0.9963 |          0.9969 | 0.0879 |   0.9906 |

**Balanced Accuracy, mean over 15 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.8898 | 0.2001 |     0.2414 | 0.6667 |  0.8176 |  0.9926 |          0.9856 | 0.2    |   0.762  |
| GRU    | 0.9232 | 0.4737 |     0.2194 | 0.6737 |  0.9011 |  0.973  |          0.9899 | 0.2576 |   0.9339 |
| MLP    | 0.9923 | 0.2247 |     0.2175 | 0.9593 |  0.9884 |  0.9964 |          0.997  | 0.2125 |   0.9903 |

**Binary detection (1-p(normal), CuCD-ID only; STIN has no benign class):**

| arch   | method        |   DET_AUROC_mean |   DET_AUPR_mean |   DET_FPR95_mean |
|:-------|:--------------|-----------------:|----------------:|-----------------:|
| CNN1D  | bic           |           0.9997 |          0.9999 |           0.0014 |
| CNN1D  | ewc           |           0.9749 |          0.9929 |           0.1348 |
| CNN1D  | finetune      |           0.9914 |          0.9975 |           0.0438 |
| CNN1D  | full          |           0.9996 |          0.9999 |           0.0003 |
| CNN1D  | icarl         |           0.9997 |          0.9999 |           0.001  |
| CNN1D  | joint         |           1      |          1      |           0      |
| CNN1D  | joint_scratch |           0.9999 |          1      |           0.0003 |
| CNN1D  | lwf           |           0.9726 |          0.9924 |           0.1526 |
| CNN1D  | replay        |           0.9994 |          0.9998 |           0.0026 |
| GRU    | bic           |           0.9929 |          0.9973 |           0.0288 |
| GRU    | ewc           |           0.9339 |          0.98   |           0.2999 |
| GRU    | finetune      |           0.7069 |          0.8772 |           0.6068 |
| GRU    | full          |           0.9942 |          0.998  |           0.0206 |
| GRU    | icarl         |           0.9934 |          0.9975 |           0.026  |
| GRU    | joint         |           0.9984 |          0.9987 |           0.0029 |
| GRU    | joint_scratch |           0.9994 |          0.9998 |           0.002  |
| GRU    | lwf           |           0.7996 |          0.9278 |           0.5555 |
| GRU    | replay        |           0.997  |          0.9987 |           0.0074 |
| MLP    | bic           |           1      |          1      |           0      |
| MLP    | ewc           |           0.9962 |          0.9934 |           0.0102 |
| MLP    | finetune      |           0.9976 |          0.9993 |           0.0122 |
| MLP    | full          |           0.9998 |          0.9998 |           0.0009 |
| MLP    | icarl         |           1      |          1      |           0.0001 |
| MLP    | joint         |           1      |          1      |           0      |
| MLP    | joint_scratch |           1      |          1      |           0      |
| MLP    | lwf           |           0.9951 |          0.9968 |           0.0322 |
| MLP    | replay        |           1      |          1      |           0.0002 |

### STIN

**Average Accuracy (ACC), mean over 15 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.6286 | 0.2406 |     0.262  | 0.5416 |  0.6072 |  0.648  |          0.6161 | 0.2485 |   0.5892 |
| GRU    | 0.6731 | 0.3091 |     0.2941 | 0.5889 |  0.6643 |  0.6596 |          0.6486 | 0.2703 |   0.6588 |
| MLP    | 0.6866 | 0.2693 |     0.2882 | 0.6509 |  0.6578 |  0.6415 |          0.6544 | 0.2849 |   0.6477 |

**Backward Transfer (BWT), mean over 15 seeds (negative = forgetting):**

| arch   |     bic |     ewc |   finetune |    full |   icarl |   joint |   joint_scratch |     lwf |   replay |
|:-------|--------:|--------:|-----------:|--------:|--------:|--------:|----------------:|--------:|---------:|
| CNN1D  | -0.2589 | -0.7597 |    -0.81   | -0.4483 | -0.4029 | -0.0494 |         -0.0531 | -0.786  |  -0.3744 |
| GRU    | -0.2056 | -0.6352 |    -0.6868 | -0.3981 | -0.3735 | -0.0575 |         -0.1444 | -0.6515 |  -0.3651 |
| MLP    | -0.1509 | -0.9325 |    -0.9168 | -0.3832 | -0.384  | -0.0807 |         -0.1249 | -0.9365 |  -0.3729 |

**Macro-F1 (primary, BiC-corrected), mean over 15 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.5744 | 0.117  |     0.13   | 0.4651 |  0.537  |  0.5929 |          0.5497 | 0.1235 |   0.5076 |
| GRU    | 0.6204 | 0.1882 |     0.1733 | 0.515  |  0.5925 |  0.5923 |          0.5855 | 0.1493 |   0.5908 |
| MLP    | 0.6476 | 0.1474 |     0.168  | 0.5822 |  0.5897 |  0.5878 |          0.5975 | 0.1635 |   0.5809 |

**Balanced Accuracy, mean over 15 seeds:**

| arch   |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:-------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| CNN1D  | 0.6286 | 0.2406 |     0.262  | 0.5416 |  0.6072 |  0.648  |          0.6161 | 0.2485 |   0.5892 |
| GRU    | 0.6731 | 0.3091 |     0.2941 | 0.5889 |  0.6643 |  0.6596 |          0.6486 | 0.2703 |   0.6588 |
| MLP    | 0.6866 | 0.2693 |     0.2882 | 0.6509 |  0.6578 |  0.6415 |          0.6544 | 0.2849 |   0.6477 |

## 3. Memory: persistent deployment vs peak update RAM (method=full)

| dataset   | arch   |   persistent_total_KB |   retained_update_state_KB |   peak_total_KB |   macs_per_forward |   persistent_model_KB |   persistent_cmmu_KB |
|:----------|:-------|----------------------:|---------------------------:|----------------:|-------------------:|----------------------:|---------------------:|
| CuCD-ID   | CNN1D  |                 13.31 |                      70.31 |          180.41 |               4406 |                  0.26 |                13.05 |
| CuCD-ID   | GRU    |                 14.17 |                      70.31 |          238.16 |               4010 |                  0.43 |                13.74 |
| CuCD-ID   | MLP    |                 12.59 |                      70.31 |          150.16 |               1296 |                  1.32 |                11.27 |
| STIN      | CNN1D  |                 17.53 |                     168.75 |          314.41 |               6030 |                  0.31 |                17.22 |
| STIN      | GRU    |                 18.53 |                     168.75 |          348.34 |               3990 |                  0.51 |                18.03 |
| STIN      | MLP    |                 16.17 |                     168.75 |          268.91 |               1616 |                  1.63 |                14.54 |

All configurations fit within the **assumed 60 KB TinyML persistent-state budget** (a conservative study constraint, not a universal CubeSat limit; largest persistent: **18.53 KB**, 30.9% of budget). Estimated peak update RAM is larger (max **348.34 KB** host-estimated, counting FP32 weights, gradients, Adam states, batch-32 activations, replay/teacher, and retained calibration state) -- a reported constraint on onboard training, not covered by the persistent number. The `retained_update_state_KB` column is the cumulative calibration half (not the experimental selection half) retained across task updates by EWC/full/BiC; it is not deployed inference state.


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
| CuCD-ID   | CNN1D  |                32 | 0.62   | -0.4911 | 0.4911 |
| CuCD-ID   | CNN1D  |                64 | 0.6154 | -0.495  | 0.495  |
| CuCD-ID   | CNN1D  |               128 | 0.6141 | -0.4993 | 0.4993 |
| CuCD-ID   | CNN1D  |               256 | 0.6257 | -0.4825 | 0.4825 |
| CuCD-ID   | CNN1D  |               512 | 0.6194 | -0.4905 | 0.4905 |
| CuCD-ID   | CNN1D  |              1024 | 0.6223 | -0.4887 | 0.4887 |
| CuCD-ID   | GRU    |                32 | 0.4741 | -0.6579 | 0.6579 |
| CuCD-ID   | GRU    |                64 | 0.5051 | -0.6179 | 0.6179 |
| CuCD-ID   | GRU    |               128 | 0.5613 | -0.5509 | 0.5509 |
| CuCD-ID   | GRU    |               256 | 0.6409 | -0.4503 | 0.4503 |
| CuCD-ID   | GRU    |               512 | 0.6275 | -0.4681 | 0.4681 |
| CuCD-ID   | GRU    |              1024 | 0.6588 | -0.426  | 0.426  |
| CuCD-ID   | MLP    |                32 | 0.7742 | -0.2971 | 0.2971 |
| CuCD-ID   | MLP    |                64 | 0.8235 | -0.2307 | 0.2307 |
| CuCD-ID   | MLP    |               128 | 0.9254 | -0.0958 | 0.0958 |
| CuCD-ID   | MLP    |               256 | 0.9531 | -0.0602 | 0.0602 |
| CuCD-ID   | MLP    |               512 | 0.971  | -0.0358 | 0.0359 |
| CuCD-ID   | MLP    |              1024 | 0.9796 | -0.0242 | 0.0242 |
| STIN      | CNN1D  |                32 | 0.5162 | -0.4896 | 0.4896 |
| STIN      | CNN1D  |                64 | 0.5423 | -0.4474 | 0.4474 |
| STIN      | CNN1D  |               128 | 0.5357 | -0.4567 | 0.4567 |
| STIN      | CNN1D  |               256 | 0.5438 | -0.4352 | 0.4352 |
| STIN      | CNN1D  |               512 | 0.5416 | -0.4483 | 0.4483 |
| STIN      | CNN1D  |              1024 | 0.5513 | -0.4362 | 0.4362 |
| STIN      | GRU    |                32 | 0.5603 | -0.4204 | 0.4204 |
| STIN      | GRU    |                64 | 0.6003 | -0.3716 | 0.3716 |
| STIN      | GRU    |               128 | 0.5958 | -0.3866 | 0.3866 |
| STIN      | GRU    |               256 | 0.6036 | -0.3776 | 0.3776 |
| STIN      | GRU    |               512 | 0.5889 | -0.3981 | 0.3981 |
| STIN      | GRU    |              1024 | 0.6063 | -0.3647 | 0.3647 |
| STIN      | MLP    |                32 | 0.612  | -0.4082 | 0.4082 |
| STIN      | MLP    |                64 | 0.6241 | -0.3906 | 0.3906 |
| STIN      | MLP    |               128 | 0.6378 | -0.3722 | 0.3722 |
| STIN      | MLP    |               256 | 0.6509 | -0.3832 | 0.3832 |
| STIN      | MLP    |               512 | 0.6508 | -0.3702 | 0.3702 |
| STIN      | MLP    |              1024 | 0.6563 | -0.3645 | 0.3645 |

Per-architecture ACC range across all tested buffer capacities:

| dataset   | arch   |    min |    max |   span |
|:----------|:-------|-------:|-------:|-------:|
| CuCD-ID   | CNN1D  | 0.6141 | 0.6257 | 0.0115 |
| CuCD-ID   | GRU    | 0.4741 | 0.6588 | 0.1846 |
| CuCD-ID   | MLP    | 0.7742 | 0.9796 | 0.2054 |
| STIN      | CNN1D  | 0.5162 | 0.5513 | 0.0351 |
| STIN      | GRU    | 0.5603 | 0.6063 | 0.046  |
| STIN      | MLP    | 0.612  | 0.6563 | 0.0443 |

### 5.2 EWC lambda sensitivity (both datasets x all 3 architectures, method=full)

| dataset   | arch   |   ewc_lambda |    ACC |     BWT |     FM |
|:----------|:-------|-------------:|-------:|--------:|-------:|
| CuCD-ID   | CNN1D  |            0 | 0.6819 | -0.4202 | 0.4202 |
| CuCD-ID   | CNN1D  |           10 | 0.6356 | -0.4784 | 0.4785 |
| CuCD-ID   | CNN1D  |          100 | 0.6194 | -0.4905 | 0.4905 |
| CuCD-ID   | CNN1D  |         1000 | 0.5871 | -0.5248 | 0.5248 |
| CuCD-ID   | CNN1D  |         5000 | 0.5667 | -0.5529 | 0.5529 |
| CuCD-ID   | CNN1D  |        20000 | 0.5596 | -0.563  | 0.563  |
| CuCD-ID   | CNN1D  |        50000 | 0.557  | -0.567  | 0.567  |
| CuCD-ID   | GRU    |            0 | 0.8612 | -0.173  | 0.173  |
| CuCD-ID   | GRU    |           10 | 0.771  | -0.2839 | 0.2839 |
| CuCD-ID   | GRU    |          100 | 0.6275 | -0.4681 | 0.4681 |
| CuCD-ID   | GRU    |         1000 | 0.5527 | -0.557  | 0.557  |
| CuCD-ID   | GRU    |         5000 | 0.508  | -0.6087 | 0.6087 |
| CuCD-ID   | GRU    |        20000 | 0.4918 | -0.6216 | 0.6216 |
| CuCD-ID   | GRU    |        50000 | 0.4759 | -0.6349 | 0.6349 |
| CuCD-ID   | MLP    |            0 | 0.9467 | -0.0689 | 0.0689 |
| CuCD-ID   | MLP    |           10 | 0.957  | -0.0546 | 0.0546 |
| CuCD-ID   | MLP    |          100 | 0.9531 | -0.0602 | 0.0602 |
| CuCD-ID   | MLP    |         1000 | 0.9513 | -0.0621 | 0.0621 |
| CuCD-ID   | MLP    |         5000 | 0.9517 | -0.0612 | 0.0612 |
| CuCD-ID   | MLP    |        20000 | 0.9491 | -0.0637 | 0.0637 |
| CuCD-ID   | MLP    |        50000 | 0.9318 | -0.085  | 0.0851 |
| STIN      | CNN1D  |            0 | 0.6116 | -0.3882 | 0.3888 |
| STIN      | CNN1D  |           10 | 0.5621 | -0.4411 | 0.4411 |
| STIN      | CNN1D  |          100 | 0.5416 | -0.4483 | 0.4483 |
| STIN      | CNN1D  |         1000 | 0.5215 | -0.4642 | 0.4642 |
| STIN      | CNN1D  |         5000 | 0.5177 | -0.4571 | 0.4571 |
| STIN      | CNN1D  |        20000 | 0.5163 | -0.4613 | 0.4613 |
| STIN      | CNN1D  |        50000 | 0.5154 | -0.4627 | 0.4627 |
| STIN      | GRU    |            0 | 0.6195 | -0.3852 | 0.3852 |
| STIN      | GRU    |           10 | 0.6273 | -0.3729 | 0.3729 |
| STIN      | GRU    |          100 | 0.5889 | -0.3981 | 0.3981 |
| STIN      | GRU    |         1000 | 0.559  | -0.4252 | 0.4252 |
| STIN      | GRU    |         5000 | 0.5229 | -0.4492 | 0.4492 |
| STIN      | GRU    |        20000 | 0.4712 | -0.4615 | 0.4615 |
| STIN      | GRU    |        50000 | 0.4577 | -0.4843 | 0.4843 |
| STIN      | MLP    |            0 | 0.6098 | -0.4201 | 0.4201 |
| STIN      | MLP    |           10 | 0.6464 | -0.3577 | 0.3577 |
| STIN      | MLP    |          100 | 0.6509 | -0.3832 | 0.3832 |
| STIN      | MLP    |         1000 | 0.6545 | -0.3724 | 0.3724 |
| STIN      | MLP    |         5000 | 0.6433 | -0.3661 | 0.3661 |
| STIN      | MLP    |        20000 | 0.6463 | -0.3881 | 0.3881 |
| STIN      | MLP    |        50000 | 0.655  | -0.3615 | 0.3615 |

Per-(dataset, architecture) ACC range across all tested lambdas:

| dataset   | arch   |    min |    max |   span |
|:----------|:-------|-------:|-------:|-------:|
| CuCD-ID   | CNN1D  | 0.557  | 0.6819 | 0.1249 |
| CuCD-ID   | GRU    | 0.4759 | 0.8612 | 0.3853 |
| CuCD-ID   | MLP    | 0.9318 | 0.957  | 0.0252 |
| STIN      | CNN1D  | 0.5154 | 0.6116 | 0.0962 |
| STIN      | GRU    | 0.4577 | 0.6273 | 0.1696 |
| STIN      | MLP    | 0.6098 | 0.655  | 0.0452 |

MLP's span is small on both datasets, while GRU and CNN1D both show a much larger span, and in both cases ACC *decreases* as lambda increases (visible directly in the per-lambda table above and in `figures/ewc_lambda_sensitivity.png`). This is why final numbers use frozen per-method x per-architecture validation configs instead of one shared configuration.

EWC here is online diagonal Fisher with a running estimate (`F_new = γ·F_old + F_current`). The Fisher sample is drawn from a class-balanced mix of the current task's training data and the cumulative calibration pool of all tasks seen so far (not the current task alone), so old-task curvature is not dropped at each consolidation. Frozen per-architecture EWC-only γ weighting: MLP γ=0.5, GRU γ=1.0, CNN1D γ=0.5. γ is treated as a validation-tuned weighting parameter rather than a conventional decay factor; the combined `full` configuration is tuned separately.


### 5.3 Feature-provenance leakage ablation (CuCD-ID/MLP, 15 seeds)

Retrained models after removing or isolating feature groups (not test-time corruption; 6 unique variants x 3 methods x 15 seeds = 270 runs). Deltas are relative to the all-feature run for the same method; negative values mean the removed features were helping the model. Group-level effects are in Section 7.10.

| variant              | method        |   n_features |   ACC_mean |   MacroF1_mean |   ACC_mean_delta_vs_all |   MacroF1_mean_delta_vs_all |
|:---------------------|:--------------|-------------:|-----------:|---------------:|------------------------:|----------------------------:|
| all                  | bic           |           22 |     0.9923 |         0.9924 |                  0      |                      0      |
| all                  | joint_scratch |           22 |     0.9966 |         0.9969 |                  0      |                      0      |
| all                  | replay        |           22 |     0.9894 |         0.9906 |                  0      |                      0      |
| no_ids               | bic           |           19 |     0.9903 |         0.9906 |                 -0.002  |                     -0.0018 |
| no_ids               | joint_scratch |           19 |     0.9956 |         0.996  |                 -0.0011 |                     -0.0009 |
| no_ids               | replay        |           19 |     0.9658 |         0.9693 |                 -0.0236 |                     -0.0213 |
| no_time              | bic           |           21 |     0.9889 |         0.989  |                 -0.0034 |                     -0.0034 |
| no_time              | joint_scratch |           21 |     0.9949 |         0.9953 |                 -0.0017 |                     -0.0015 |
| no_time              | replay        |           21 |     0.9901 |         0.9907 |                  0.0007 |                      0.0001 |
| no_time_ids_window   | bic           |            9 |     0.9153 |         0.898  |                 -0.0771 |                     -0.0944 |
| no_time_ids_window   | joint_scratch |            9 |     0.9723 |         0.9757 |                 -0.0243 |                     -0.0211 |
| no_time_ids_window   | replay        |            9 |     0.9229 |         0.9338 |                 -0.0666 |                     -0.0569 |
| no_window            | bic           |           13 |     0.9384 |         0.9421 |                 -0.0539 |                     -0.0503 |
| no_window            | joint_scratch |           13 |     0.9912 |         0.9922 |                 -0.0054 |                     -0.0047 |
| no_window            | replay        |           13 |     0.9042 |         0.922  |                 -0.0852 |                     -0.0686 |
| only_time_ids_window | bic           |           13 |     0.9843 |         0.9844 |                 -0.008  |                     -0.008  |
| only_time_ids_window | joint_scratch |           13 |     0.9956 |         0.9959 |                 -0.001  |                     -0.0009 |
| only_time_ids_window | replay        |           13 |     0.9694 |         0.9701 |                 -0.02   |                     -0.0206 |

This quantifies how much random-split performance rides on the circular time field, CCSDS identifiers, and 20-second window aggregates. It is a reliance measure, not proof that every removed feature is illegitimate: `no_ids` also removes protocol information a real detector may legitimately use. The IID headline remains a benchmark estimand; the temporal-purged table in Section 7.4 is the separate deployment-shift estimand.


### 5.4 Task-order sensitivity, legacy 3-perm (CuCD-ID, MLP, method=full)

_Legacy confounded design (budgets move with the task); superseded by the disentangled 12-permutation study in §7.3. Retained for provenance only._

| order        |   ACC_mean |   ACC_std |   BWT_mean |   BWT_std |
|:-------------|-----------:|----------:|-----------:|----------:|
| [0, 1, 2, 3] |     0.8634 |    0.1474 |    -0.179  |    0.1971 |
| [1, 3, 0, 2] |     0.9765 |    0.0266 |    -0.0281 |    0.0359 |
| [3, 2, 1, 0] |     0.9959 |    0.0023 |    -0.0036 |    0.0024 |

ACC spread across the 3 legacy orderings: **0.1325** (range 0.8634 - 0.9959). See §7.3 for the disentangled measurement (computed from `task_order_disentangled.csv`, not hard-coded).


### 5.5 Priority-2 controls: matched replay budget and EWC lambda

Main-table methods are not equal-memory. `replay_matched` uses the `full` buffer capacity/replay fraction; `ewc_l0`/`ewc_l10` and `full_l0`/`full_l10` switch EWC off or weak. These are separate runs on the same locked splits/seeds (15 seeds).

| dataset   | arch   | condition      |    ACC |
|:----------|:-------|:---------------|-------:|
| CuCD-ID   | CNN1D  | ewc_l0         | 0.2534 |
| CuCD-ID   | CNN1D  | ewc_l10        | 0.2504 |
| CuCD-ID   | CNN1D  | full_l0        | 0.6819 |
| CuCD-ID   | CNN1D  | full_l10       | 0.6356 |
| CuCD-ID   | CNN1D  | replay_matched | 0.6819 |
| CuCD-ID   | GRU    | ewc_l0         | 0.2621 |
| CuCD-ID   | GRU    | ewc_l10        | 0.292  |
| CuCD-ID   | GRU    | full_l0        | 0.8612 |
| CuCD-ID   | GRU    | full_l10       | 0.771  |
| CuCD-ID   | GRU    | replay_matched | 0.9124 |
| CuCD-ID   | MLP    | ewc_l0         | 0.2608 |
| CuCD-ID   | MLP    | ewc_l10        | 0.2868 |
| CuCD-ID   | MLP    | full_l0        | 0.9467 |
| CuCD-ID   | MLP    | full_l10       | 0.957  |
| CuCD-ID   | MLP    | replay_matched | 0.9754 |
| STIN      | CNN1D  | ewc_l0         | 0.2443 |
| STIN      | CNN1D  | ewc_l10        | 0.2411 |
| STIN      | CNN1D  | full_l0        | 0.6116 |
| STIN      | CNN1D  | full_l10       | 0.5621 |
| STIN      | CNN1D  | replay_matched | 0.6116 |
| STIN      | GRU    | ewc_l0         | 0.2941 |
| STIN      | GRU    | ewc_l10        | 0.3212 |
| STIN      | GRU    | full_l0        | 0.6195 |
| STIN      | GRU    | full_l10       | 0.6273 |
| STIN      | GRU    | replay_matched | 0.6497 |
| STIN      | MLP    | ewc_l0         | 0.2948 |
| STIN      | MLP    | ewc_l10        | 0.2831 |
| STIN      | MLP    | full_l0        | 0.6098 |
| STIN      | MLP    | full_l10       | 0.6464 |
| STIN      | MLP    | replay_matched | 0.6256 |

Paired exact tests (Holm over controls): **7/30** comparisons survive p_holm<0.05. `replay_matched vs replay` isolates the replay-budget difference; `*_l0/*_l10 vs ewc/full` isolates the EWC term. See `results/stats_baseline_controls.csv` for CIs/effect sizes.


These controls are a tuning diagnostic, not a method win: in several cells switching EWC off (`full_l0`) exceeds the tuned `full` reference, and on CuCD-ID/GRU it remains below matched replay. Treat the selected EWC strength as a limitation of the tuning grid, not as evidence that the combined method is superior.


## 6. Per-class F1 -- direct evidence of forgetting

Section 2's ACC/BWT/FM numbers are task-level aggregates; this section makes the same phenomenon visible at the level of individual attack classes by evaluating the final trained model (all classes seen so far, single shared head) and computing a per-class precision/recall/F1 breakdown. See `figures/per_class_f1_*.png` and `figures/confusion_matrix_*.png` for the visual version.


### 6.1 STIN / GRU, mean F1 per task-of-origin

| task_name     |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:--------------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| T0_flood_base | 0.73   | 0.1947 |     0.1501 | 0.5325 |  0.5425 |  0.7626 |          0.7537 | 0.1327 |   0.5859 |
| T1_flood_ext  | 0.6156 | 0.0308 |     0      | 0.4623 |  0.5744 |  0.721  |          0.6185 | 0.0269 |   0.529  |
| T2_app_layer  | 0.5158 | 0.339  |     0.3698 | 0.5501 |  0.6607 |  0.2934 |          0.3842 | 0.2884 |   0.6574 |

### 6.2 CuCD-ID / MLP, mean F1 per task-of-origin

| task_name             |    bic |    ewc |   finetune |   full |   icarl |   joint |   joint_scratch |    lwf |   replay |
|:----------------------|-------:|-------:|-----------:|-------:|--------:|--------:|----------------:|-------:|---------:|
| T0_launch             | 0.9946 | 0.0323 |     0.0508 | 0.989  |  0.9903 |  0.9971 |          0.9974 | 0.0366 |   0.9955 |
| T1_data_injection     | 0.9958 | 0      |     0      | 0.9033 |  0.994  |  0.9976 |          0.9978 | 0      |   0.9892 |
| T2_defence_impairment | 0.9908 | 0.0934 |     0      | 0.9596 |  0.9912 |  0.9959 |          0.9961 | 0      |   0.9912 |
| T3_storage_exhaustion | 0.9863 | 0.3754 |     0.3683 | 0.9399 |  0.9805 |  0.9937 |          0.9956 | 0.3661 |   0.9819 |

For CuCD-ID, the memoryless methods (`finetune`, `ewc`) collapse most classes to zero or near-zero F1 while retaining some ability on the most recent class; the exact per-class pattern is shown above and varies by method and seed. On STIN the same severe recency bias appears, but it is noisier because each task introduces several classes and the feature set is less separable. Memory-based and joint references retain non-zero F1 across most tasks. This is the direct per-class confirmation of the task-level results in Section 2; do not infer a universal exact-zero pattern from one method.


## 7. Phase-2 evidence (all numbers computed from their CSVs)


### 7.1 Few-shot uplink sweep, CuCD-ID (RQ4)

Two calibration protocols are reported: **fixed** (150 calibration + 150 selection rows/class) and **scaled** (K calibration + K selection rows/class). Calibration payload is charged only to methods that consume it (EWC/full/BiC).

| calibration_protocol   | arch   | method   |      5 |     10 |     25 |     50 |    100 |    150 |    300 |
|:-----------------------|:-------|:---------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| fixed                  | GRU    | icarl    | 0.4708 | 0.6041 | 0.7439 | 0.8684 | 0.9012 | 0.918  | 0.9404 |
| fixed                  | GRU    | replay   | 0.4276 | 0.4887 | 0.4923 | 0.681  | 0.8283 | 0.9249 | 0.9714 |
| fixed                  | MLP    | bic      | 0.5904 | 0.8799 | 0.8899 | 0.9544 | 0.984  | 0.9937 | 0.9933 |
| fixed                  | MLP    | full     | 0.5138 | 0.5    | 0.6291 | 0.7524 | 0.8869 | 0.9349 | 0.9815 |
| fixed                  | MLP    | icarl    | 0.4972 | 0.7333 | 0.7747 | 0.901  | 0.9795 | 0.9916 | 0.9931 |
| fixed                  | MLP    | replay   | 0.5133 | 0.5237 | 0.5062 | 0.7213 | 0.8952 | 0.991  | 0.9958 |
| scaled                 | GRU    | icarl    | 0.443  | 0.6443 | 0.7107 | 0.8414 | 0.8502 | 0.918  | 0.9207 |
| scaled                 | GRU    | replay   | 0.4947 | 0.4762 | 0.5118 | 0.5634 | 0.7989 | 0.9249 | 0.966  |
| scaled                 | MLP    | bic      | 0.6539 | 0.8933 | 0.9139 | 0.9705 | 0.9882 | 0.9937 | 0.9951 |
| scaled                 | MLP    | full     | 0.4597 | 0.4977 | 0.5216 | 0.7171 | 0.9094 | 0.9349 | 0.9389 |
| scaled                 | MLP    | icarl    | 0.5033 | 0.692  | 0.8024 | 0.9107 | 0.9718 | 0.9916 | 0.9948 |
| scaled                 | MLP    | replay   | 0.5225 | 0.5036 | 0.5442 | 0.7481 | 0.9218 | 0.991  | 0.9951 |

Mean total uplink KB across method/architecture rows, including method-appropriate calibration payload, **fixed** protocol per K: K=5: 3.7KB, K=10: 4.0KB, K=25: 5.0KB, K=50: 6.7KB, K=100: 10.1KB, K=150: 13.5KB, K=300: 23.6KB.


Mean total uplink KB across method/architecture rows, including method-appropriate calibration payload, **scaled** protocol per K: K=5: 0.5KB, K=10: 0.9KB, K=25: 2.2KB, K=50: 4.5KB, K=100: 9.0KB, K=150: 13.5KB, K=300: 26.9KB.

MLP @K=50 (fixed calibration): bic=0.954, full=0.752, icarl=0.901, replay=0.721.

MLP @K=5 (extreme, fixed calibration): bic=0.590, full=0.514, icarl=0.497, replay=0.513.

Largest MLP/BiC Macro-F1 step is at K=10 (K=50: 0.958; K=100: 0.985). Both fixed and scaled curves are in the table; the knee is read from the data, not hardcoded.


### 7.2 Per-arch tune, CuCD-ID/CNN (exploratory, validation-scored, not applied)

| method        |    lr |   epochs |    ACC |
|:--------------|------:|---------:|-------:|
| icarl         | 0.001 |       25 | 0.2742 |
| icarl         | 0.001 |       50 | 0.3356 |
| icarl         | 0.003 |       25 | 0.3833 |
| icarl         | 0.003 |       50 | 0.5714 |
| icarl         | 0.01  |       25 | 0.6769 |
| icarl         | 0.01  |       50 | 0.7975 |
| joint_scratch | 0.001 |       25 | 0.6983 |
| joint_scratch | 0.001 |       50 | 0.8111 |
| joint_scratch | 0.003 |       25 | 0.8686 |
| joint_scratch | 0.003 |       50 | 0.9231 |
| joint_scratch | 0.01  |       25 | 0.9306 |
| joint_scratch | 0.01  |       50 | 0.98   |
| replay        | 0.001 |       25 | 0.3869 |
| replay        | 0.001 |       50 | 0.4475 |
| replay        | 0.003 |       25 | 0.4719 |
| replay        | 0.003 |       50 | 0.5189 |
| replay        | 0.01  |       25 | 0.6181 |
| replay        | 0.01  |       50 | 0.8117 |

- replay: best lr=0.01, epochs=50 -> ACC=0.812.

- icarl: best lr=0.01, epochs=50 -> ACC=0.797.

- joint_scratch: best lr=0.01, epochs=50 -> ACC=0.980.



### 7.3 Disentangled task-order study, CuCD-ID/MLP (RQ5)

| budget_mode     | method   |   mean |    std |
|:----------------|:---------|-------:|-------:|
| fixed_position  | icarl    | 0.9898 | 0.0139 |
| fixed_position  | replay   | 0.9887 | 0.0143 |
| moves_with_task | icarl    | 0.9925 | 0.0042 |
| moves_with_task | replay   | 0.9856 | 0.0193 |

**Degenerate vs primary estimand** (single-class first task => zero-gradient first update):

| budget_mode     | degenerate_single_class_first   |   mean |    std |   count |
|:----------------|:--------------------------------|-------:|-------:|--------:|
| fixed_position  | False                           | 0.9876 | 0.0144 |      36 |
| fixed_position  | True                            | 0.991  | 0.0136 |      36 |
| moves_with_task | False                           | 0.9876 | 0.0144 |      36 |
| moves_with_task | True                            | 0.9905 | 0.0142 |      36 |

**Primary** (two-class T0 first, 6 permutations): per-order mean spread **0.0146**.

The two primary budget modes are numerically identical by construction: every primary order starts with T0, so both modes assign 700 rows to position 0 and 150 rows thereafter. They are not independent evidence; the reported spread is order sensitivity only. The degenerate orders, where T0 is not first, do differ.

**Degenerate** (single-class first, 6 permutations): per-order mean spread 0.0089; these first updates are no-ops, so they are reported separately and excluded from the primary claim.

Individual-run ACC range across all permutations: **0.0871**. The legacy confounded headline is retired.


### 7.4 IID vs forward-temporal protocols, CuCD-ID (3 architectures)

| method        | protocol                               |   mean |    std |
|:--------------|:---------------------------------------|-------:|-------:|
| bic           | iid                                    | 0.9282 | 0.0594 |
| bic           | temporal_purged                        | 0.2325 | 0.1054 |
| bic           | temporal_purged_fair                   | 0.7153 | 0.1434 |
| bic           | temporal_purged_fair_no_leaky_features | 0.5145 | 0.1028 |
| bic           | temporal_purged_no_leaky_features      | 0.2385 | 0.1007 |
| ewc           | iid                                    | 0.3209 | 0.0983 |
| ewc           | temporal_purged                        | 0.1882 | 0.0929 |
| ewc           | temporal_purged_fair                   | 0.2938 | 0.0788 |
| ewc           | temporal_purged_fair_no_leaky_features | 0.2574 | 0.053  |
| ewc           | temporal_purged_no_leaky_features      | 0.2372 | 0.0552 |
| finetune      | iid                                    | 0.2681 | 0.0332 |
| finetune      | temporal_purged                        | 0.2215 | 0.0469 |
| finetune      | temporal_purged_fair                   | 0.2636 | 0.0324 |
| finetune      | temporal_purged_fair_no_leaky_features | 0.2909 | 0.0696 |
| finetune      | temporal_purged_no_leaky_features      | 0.245  | 0.0492 |
| full          | iid                                    | 0.7333 | 0.1749 |
| full          | temporal_purged                        | 0.1839 | 0.0967 |
| full          | temporal_purged_fair                   | 0.5637 | 0.1524 |
| full          | temporal_purged_fair_no_leaky_features | 0.5252 | 0.1517 |
| full          | temporal_purged_no_leaky_features      | 0.2461 | 0.0844 |
| icarl         | iid                                    | 0.8995 | 0.0863 |
| icarl         | temporal_purged                        | 0.1923 | 0.0954 |
| icarl         | temporal_purged_fair                   | 0.7072 | 0.152  |
| icarl         | temporal_purged_fair_no_leaky_features | 0.5309 | 0.0993 |
| icarl         | temporal_purged_no_leaky_features      | 0.2394 | 0.1001 |
| joint         | iid                                    | 0.9853 | 0.0143 |
| joint         | temporal_purged                        | 0.1713 | 0.0889 |
| joint         | temporal_purged_fair                   | 0.7908 | 0.1031 |
| joint         | temporal_purged_fair_no_leaky_features | 0.6385 | 0.102  |
| joint         | temporal_purged_no_leaky_features      | 0.2394 | 0.1115 |
| joint_scratch | iid                                    | 0.9897 | 0.0092 |
| joint_scratch | temporal_purged                        | 0.1783 | 0.0801 |
| joint_scratch | temporal_purged_fair                   | 0.8233 | 0.0903 |
| joint_scratch | temporal_purged_fair_no_leaky_features | 0.6631 | 0.1051 |
| joint_scratch | temporal_purged_no_leaky_features      | 0.1636 | 0.1003 |
| lwf           | iid                                    | 0.2683 | 0.0331 |
| lwf           | temporal_purged                        | 0.2003 | 0.0768 |
| lwf           | temporal_purged_fair                   | 0.2701 | 0.0468 |
| lwf           | temporal_purged_fair_no_leaky_features | 0.2772 | 0.0589 |
| lwf           | temporal_purged_no_leaky_features      | 0.2528 | 0.0133 |
| replay        | iid                                    | 0.8736 | 0.1307 |
| replay        | temporal_purged                        | 0.189  | 0.0995 |
| replay        | temporal_purged_fair                   | 0.6987 | 0.1151 |
| replay        | temporal_purged_fair_no_leaky_features | 0.6106 | 0.1372 |
| replay        | temporal_purged_no_leaky_features      | 0.1848 | 0.1087 |

Cold-start temporal + no time/ID/window features:

| method        |   mean |    std |
|:--------------|-------:|-------:|
| bic           | 0.2385 | 0.1007 |
| ewc           | 0.2372 | 0.0552 |
| finetune      | 0.245  | 0.0492 |
| full          | 0.2461 | 0.0844 |
| icarl         | 0.2394 | 0.1001 |
| joint         | 0.2394 | 0.1115 |
| joint_scratch | 0.1636 | 0.1003 |
| lwf           | 0.2528 | 0.0133 |
| replay        | 0.1848 | 0.1087 |

Fair random-past temporal + no time/ID/window features:

| method        |   mean |    std |
|:--------------|-------:|-------:|
| bic           | 0.5145 | 0.1028 |
| ewc           | 0.2574 | 0.053  |
| finetune      | 0.2909 | 0.0696 |
| full          | 0.5252 | 0.1517 |
| icarl         | 0.5309 | 0.0993 |
| joint         | 0.6385 | 0.102  |
| joint_scratch | 0.6631 | 0.1051 |
| lwf           | 0.2772 | 0.0589 |
| replay        | 0.6106 | 0.1372 |

IID = random-split benchmark. `temporal_purged` is the original cold-start arm: train draws the earliest head of each class's past block, validation is the contiguous block ending one purge before test. `temporal_purged_fair` draws train and validation from the purged past and locks test in the future (70/10/20). Purge: 0.02 rad on TimeRadians (a proxy for the documented ~20-22 s window horizon; not a wall-clock guarantee). Blocks are ordered on unwrapped circular time so an orbital wrap does not scramble chronology.

**Interpretation.** Mean deployable-method ACC is **0.6131 (IID)**, **0.2011 (cold-start temporal)**, and **0.5018 (fair random-past temporal)**. The cold-start arm isolates support: training only on the earliest head makes forward generalization look near chance even for `joint_scratch`, so the cold-start gap is not a pure drift or forgetting estimate. The fair arm recovers most of the IID accuracy for replay/BiC, but `finetune` remains near chance and the no-leaky fair arm falls to **0.4295**, showing that temporal shift and time/ID/window feature reliance remain separate limitations.

Cold-start is therefore a support-controlled baseline, not the deployment estimate. Random-past validation also avoids the empty-validation cells that the contiguous block creates for narrow classes; the builder warns for the cold-start arm.

**Matched-split control.** Re-running the cold-start arm with the same 70/10/20 test block and random validation as the fair arm leaves MLP `joint_scratch` at **0.106** and BiC at **0.171**, versus **0.870** and **0.894** under random-past training (3 seeds; `temporal_sampling_diagnosis.csv`). Test/validation block choice is therefore not the main driver of the recovery.


#### 7.4.1 Temporal leakage/validity audit (read-only, no models trained)

This audit replays the exact random per-class splits used by `build_cucd_id_tasks` (including the per-task RNG advance) and verifies row-level parity before measuring the nearest training row **actually drawn for the model** (not the larger pre-draw train pool), using circular `TimeRadians` distance. It is diagnostic: it does not estimate a leakage-free accuracy.

| class              | train/test rows   |   median dt (rad) | IQR (rad)         |   max (rad) | test rows <= 0.02 rad   |
|:-------------------|:------------------|------------------:|:------------------|------------:|:------------------------|
| normal             | 700/1463          |          8.2e-05  | 3.00e-05–1.35e-04 |     0.00087 | 100.0% (100.0–100.0)    |
| command_flooding   | 700/1087          |          2.3e-05  | 1.50e-05–5.20e-05 |     0.00328 | 100.0% (100.0–100.0)    |
| data_injection     | 150/1292          |          0.000356 | 1.63e-04–6.90e-04 |     0.00506 | 100.0% (100.0–100.0)    |
| defence_impairment | 150/1413          |          0.0048   | 2.18e-03–9.61e-03 |     0.0547  | 94.6% (92.9–97.5)       |
| storage_exhaustion | 150/1485          |          0.00343  | 1.48e-03–6.54e-03 |     0.0431  | 98.6% (95.5–100.0)      |

`train` is the number of rows drawn per class (700 for T0, 150 for later tasks); `test` is the frozen random test split. Percentages are pooled over 15 splits, with the per-split range in parentheses.


**Class occupancy** (unwrapped chronological time).

| class              |   rows | time span (rad)   | blocks occupied   |   largest within-class gap (rad) |
|:-------------------|-------:|:------------------|:------------------|---------------------------------:|
| normal             |   4878 | 7.966–8.104       | 1 (B2)            |                           0.0007 |
| command_flooding   |   3622 | 7.613–8.054       | 1 (B2)            |                           0.0724 |
| data_injection     |   4307 | 5.156–5.321       | 1 (B0)            |                           0.0046 |
| defence_impairment |   4709 | 6.894–9.006       | 4 (B1,B2,B3,B4)   |                           0.0262 |
| storage_exhaustion |   4949 | 5.795–7.234       | 1 (B1)            |                           0.0041 |

**Natural time blocks** (gap > 0.02 rad; unwrapped chronological time).

| block   | time span (rad)   |   rows |   classes | class counts                                                |
|:--------|:------------------|-------:|----------:|:------------------------------------------------------------|
| B0      | 5.156–5.321       |   4307 |         1 | data_injection=4307                                         |
| B1      | 5.795–7.452       |   6299 |         2 | defence_impairment=1350; storage_exhaustion=4949            |
| B2      | 7.478–8.710       |  11292 |         3 | normal=4878; command_flooding=3622; defence_impairment=2792 |
| B3      | 8.736–8.794       |    135 |         1 | defence_impairment=135                                      |
| B4      | 8.819–9.006       |    432 |         1 | defence_impairment=432                                      |

**Within-class early-vs-late feature drift** (split at each class's median time; pooled within-phase SD; `TimeRadians` excluded).

| class              | early/late rows   |   mean abs z |   max abs z | largest shift                       |
|:-------------------|:------------------|-------------:|------------:|:------------------------------------|
| normal             | 2439/2439         |        0.452 |       3.233 | MemoryPercpuMB (3.23)               |
| command_flooding   | 1811/1811         |        0.52  |       2.705 | SlidingWindowMinIntervalSec (2.70)  |
| data_injection     | 2154/2153         |        0.824 |       4.866 | SlidingWindowMeanIntervalSec (4.87) |
| defence_impairment | 2355/2354         |        0.465 |       2.689 | MemoryPercpuMB (2.69)               |
| storage_exhaustion | 2475/2474         |        0.204 |       1.166 | MemoryPercpuMB (1.17)               |

**Reading.** Across 15 random splits, **98.6%** of all test rows have a training row within the 0.02 rad purge horizon (per-class 94.6–100.0%). For `normal`, `command_flooding`, and `data_injection`, every test row is within that horizon. For `defence_impairment` and `storage_exhaustion`, the 150-row training draw leaves some tail rows beyond 0.02 rad, but their medians remain 0.003–0.005 rad and their maxima 0.04–0.06 rad. This is interpolation inside the same telemetry window, before accounting for overlapping 20-second aggregates.

The split is also not a scenario split. There are only **5** natural time blocks under the 0.02 rad gap rule, and **normal, command_flooding, data_injection, storage_exhaustion** are confined to a single block. A leave-one-block-out estimate therefore cannot form a complete four-task scenario evaluation. Within-class early/late drift is large (mean |z| 0.20–0.82; peak |z| 4.87 on `SlidingWindowMeanIntervalSec`), so the cold-start temporal gap is a joint support + drift + scaler/prior-shift estimand. The fair random-past arm removes the cold-start support component but not drift or feature reliance. Neither the IID number nor the IID-minus-temporal gap should be read as a causal leakage correction.

See `figures/temporal_leakage_audit.png` and `results/temporal_leakage_audit.json`.


### 7.5 Robustness + simulated-quantization ledger, CuCD-ID/MLP (RQ6/RQ7)

_`int8_sim` is a simulated post-training INT8 round-trip accuracy proxy (FP32 weights only); no INT8 kernels were executed and no hardware deployment is claimed._

| method        | corruption   |   ACC_final_pool |
|:--------------|:-------------|-----------------:|
| bic           | clean        |           0.9942 |
| bic           | drop_0.1     |           0.9365 |
| bic           | drop_0.3     |           0.8008 |
| bic           | gauss_0.1    |           0.9937 |
| bic           | gauss_0.5    |           0.97   |
| bic           | int8_sim     |           0.9942 |
| bic           | nowindow     |           0.748  |
| icarl         | clean        |           0.992  |
| icarl         | drop_0.1     |           0.9381 |
| icarl         | drop_0.3     |           0.8046 |
| icarl         | gauss_0.1    |           0.9917 |
| icarl         | gauss_0.5    |           0.9623 |
| icarl         | int8_sim     |           0.9921 |
| icarl         | nowindow     |           0.6431 |
| joint_scratch | clean        |           0.9964 |
| joint_scratch | drop_0.1     |           0.9395 |
| joint_scratch | drop_0.3     |           0.8043 |
| joint_scratch | gauss_0.1    |           0.9962 |
| joint_scratch | gauss_0.5    |           0.9746 |
| joint_scratch | int8_sim     |           0.9964 |
| joint_scratch | nowindow     |           0.7278 |
| replay        | clean        |           0.9923 |
| replay        | drop_0.1     |           0.9319 |
| replay        | drop_0.3     |           0.7953 |
| replay        | gauss_0.1    |           0.991  |
| replay        | gauss_0.5    |           0.9459 |
| replay        | int8_sim     |           0.9927 |
| replay        | nowindow     |           0.7118 |

Mean pooled-ACC drop vs clean (all methods): drop_0.1=+0.057, drop_0.3=+0.192, gauss_0.1=+0.001, gauss_0.5=+0.031, int8_sim=-0.000, nowindow=+0.286.


### 7.6 Activation memory ledger (analytic, batch=1 and batch=32; SIM_ = host-simulated)

| arch   |   act_fwd_floats |   SIM_act_fwd_KB |   SIM_act_train_KB | seq_len   |   SIM_act_fwd_KB_batch32 |   SIM_act_train_KB_batch32 |
|:-------|-----------------:|-----------------:|-------------------:|:----------|-------------------------:|---------------------------:|
| MLP    |              101 |            0.395 |              0.789 |           |                   12.625 |                      25.25 |
| GRU    |              775 |            3.027 |              6.055 | 11.0000   |                   96.875 |                     193.75 |
| CNN1D  |              719 |            2.809 |              5.617 |           |                   89.875 |                     179.75 |

The full persistent/update ledger (`results/memory_ledger.csv`) is derived from the main matrix; this table isolates activations.


### 7.7 Exact paired permutation + Holm (matched seeds, ACC): 162/216 pre-specified pairwise comparisons significant at Holm p<0.05

The family is all unordered method pairs within each dataset x architecture (216 comparisons per metric), fixed before looking at results; Holm correction is applied across the whole family. The exact sign-flip test has n=15 paired seeds. This is exploratory, not a winner-selected headline list; significance is not scientific importance, so read the effect sizes and confidence intervals rather than the count of significant comparisons.

| dataset   | arch   | method_a   | method_b      |   n |   mean_a |   mean_b |   mean_diff |   ci_low |   ci_high |   cohens_d |   p_exact |   p_holm |
|:----------|:-------|:-----------|:--------------|----:|---------:|---------:|------------:|---------:|----------:|-----------:|----------:|---------:|
| CuCD-ID   | MLP    | replay     | lwf           |  15 |   0.9894 |   0.2588 |      0.7306 |   0.7144 |    0.7404 |    24.7157 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | replay     | ewc           |  15 |   0.9894 |   0.2744 |      0.7151 |   0.6844 |    0.7364 |    13.0424 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | full       | lwf           |  15 |   0.9531 |   0.2588 |      0.6943 |   0.6574 |    0.72   |    10.3816 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | lwf           |  15 |   0.9224 |   0.2961 |      0.6262 |   0.6029 |    0.645  |    14.6159 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | ewc           |  15 |   0.9224 |   0.4384 |      0.4839 |   0.4445 |    0.5223 |     5.9923 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | lwf           |  15 |   0.7089 |   0.25   |      0.4589 |   0.4195 |    0.5025 |     5.3742 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | ewc           |  15 |   0.7089 |   0.25   |      0.4589 |   0.4194 |    0.5025 |     5.3728 |    0.0001 |   0.0132 |
| STIN      | GRU    | replay     | lwf           |  15 |   0.6588 |   0.2703 |      0.3885 |   0.3573 |    0.4186 |     6.1486 |    0.0001 |   0.0132 |
| STIN      | MLP    | replay     | ewc           |  15 |   0.6477 |   0.2693 |      0.3784 |   0.3643 |    0.395  |    12.1621 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | lwf           |  15 |   0.6194 |   0.25   |      0.3694 |   0.324  |    0.4102 |     4.1378 |    0.0001 |   0.0132 |
| STIN      | MLP    | full       | lwf           |  15 |   0.6509 |   0.2849 |      0.366  |   0.3559 |    0.3748 |    18.7197 |    0.0001 |   0.0132 |
| STIN      | MLP    | replay     | lwf           |  15 |   0.6477 |   0.2849 |      0.3628 |   0.3554 |    0.3705 |    23.5315 |    0.0001 |   0.0132 |
| STIN      | GRU    | replay     | ewc           |  15 |   0.6588 |   0.3091 |      0.3496 |   0.32   |    0.38   |     5.679  |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | ewc           |  15 |   0.5892 |   0.2406 |      0.3486 |   0.3242 |    0.372  |     7.1498 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | lwf           |  15 |   0.5892 |   0.2485 |      0.3407 |   0.3161 |    0.3647 |     6.8336 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | lwf           |  15 |   0.6275 |   0.2961 |      0.3314 |   0.2878 |    0.3721 |     3.8408 |    0.0001 |   0.0132 |
| STIN      | GRU    | full       | lwf           |  15 |   0.5889 |   0.2703 |      0.3186 |   0.2811 |    0.3523 |     4.4061 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | full          |  15 |   0.9224 |   0.6275 |      0.2948 |   0.2597 |    0.3322 |     3.9422 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | full       | lwf           |  15 |   0.5416 |   0.2485 |      0.2932 |   0.2747 |    0.3141 |     7.2171 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | lwf           |  15 |   0.4384 |   0.2961 |      0.1423 |   0.108  |    0.1786 |     1.9728 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | full          |  15 |   0.7089 |   0.6194 |      0.0896 |   0.0412 |    0.1407 |     0.8697 |    0.0029 |   0.1463 |
| CuCD-ID   | CNN1D  | bic        | icarl         |  15 |   0.8729 |   0.796  |      0.0769 |   0.0592 |    0.0945 |     2.1127 |    0.0001 |   0.0132 |
| STIN      | GRU    | replay     | full          |  15 |   0.6588 |   0.5889 |      0.0699 |   0.0438 |    0.0994 |     1.2406 |    0.0002 |   0.0132 |
| STIN      | CNN1D  | replay     | full          |  15 |   0.5892 |   0.5416 |      0.0476 |   0.0254 |    0.0693 |     1.0432 |    0.0017 |   0.0906 |
| STIN      | MLP    | bic        | joint         |  15 |   0.6866 |   0.6415 |      0.0451 |   0.0328 |    0.0569 |     1.809  |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | lwf           |  15 |   0.3091 |   0.2703 |      0.0389 |   0.0116 |    0.0683 |     0.6717 |    0.0176 |   0.7585 |
| CuCD-ID   | MLP    | replay     | full          |  15 |   0.9894 |   0.9531 |      0.0363 |   0.0134 |    0.0715 |     0.5687 |    0.0003 |   0.0189 |
| STIN      | MLP    | bic        | joint_scratch |  15 |   0.6866 |   0.6544 |      0.0322 |   0.0125 |    0.052  |     0.8032 |    0.0099 |   0.4351 |
| STIN      | CNN1D  | joint      | joint_scratch |  15 |   0.648  |   0.6161 |      0.032  |   0.0198 |    0.0434 |     1.3082 |    0.0004 |   0.0252 |
| CuCD-ID   | CNN1D  | finetune   | lwf           |  15 |   0.2797 |   0.25   |      0.0297 |   0.0145 |    0.0483 |     0.8586 |    0.0005 |   0.0283 |
| CuCD-ID   | CNN1D  | finetune   | ewc           |  15 |   0.2797 |   0.25   |      0.0296 |   0.0145 |    0.0482 |     0.8595 |    0.0005 |   0.0283 |
| STIN      | MLP    | bic        | icarl         |  15 |   0.6866 |   0.6578 |      0.0288 |   0.0202 |    0.037  |     1.6861 |    0.0001 |   0.0132 |
| STIN      | GRU    | bic        | joint_scratch |  15 |   0.6731 |   0.6486 |      0.0244 |   0.0022 |    0.0448 |     0.5598 |    0.0492 |   1      |
| STIN      | GRU    | finetune   | lwf           |  15 |   0.2941 |   0.2703 |      0.0239 |  -0.0074 |    0.049  |     0.412  |    0.1328 |   1      |
| STIN      | CNN1D  | bic        | icarl         |  15 |   0.6286 |   0.6072 |      0.0214 |   0.0065 |    0.0364 |     0.6903 |    0.019  |   0.7998 |
| STIN      | CNN1D  | finetune   | ewc           |  15 |   0.262  |   0.2406 |      0.0214 |   0.0021 |    0.0398 |     0.5558 |    0.0535 |   1      |
| STIN      | MLP    | finetune   | ewc           |  15 |   0.2882 |   0.2693 |      0.0189 |   0.0063 |    0.0352 |     0.6409 |    0.0065 |   0.2976 |
| STIN      | MLP    | icarl      | joint         |  15 |   0.6578 |   0.6415 |      0.0163 |   0.0017 |    0.0308 |     0.541  |    0.0566 |   1      |
| STIN      | GRU    | icarl      | joint_scratch |  15 |   0.6643 |   0.6486 |      0.0157 |  -0.0003 |    0.0327 |     0.4668 |    0.0928 |   1      |
| CuCD-ID   | MLP    | ewc        | lwf           |  15 |   0.2744 |   0.2588 |      0.0156 |   0.0017 |    0.0315 |     0.5094 |    0.0527 |   1      |
| STIN      | CNN1D  | finetune   | lwf           |  15 |   0.262  |   0.2485 |      0.0135 |  -0.0014 |    0.0286 |     0.4351 |    0.0813 |   1      |
| STIN      | GRU    | bic        | joint         |  15 |   0.6731 |   0.6596 |      0.0134 |  -0.008  |    0.0343 |     0.3077 |    0.2521 |   1      |
| STIN      | CNN1D  | bic        | joint_scratch |  15 |   0.6286 |   0.6161 |      0.0125 |  -0.0115 |    0.0356 |     0.2627 |    0.3251 |   1      |
| STIN      | GRU    | joint      | joint_scratch |  15 |   0.6596 |   0.6486 |      0.011  |  -0.0087 |    0.0304 |     0.276  |    0.2996 |   1      |
| CuCD-ID   | GRU    | replay     | icarl         |  15 |   0.9224 |   0.912  |      0.0103 |  -0.0022 |    0.0238 |     0.3903 |    0.1557 |   1      |
| STIN      | GRU    | replay     | joint_scratch |  15 |   0.6588 |   0.6486 |      0.0101 |  -0.0071 |    0.0285 |     0.2762 |    0.3    |   1      |
| STIN      | MLP    | full       | joint         |  15 |   0.6509 |   0.6415 |      0.0094 |  -0.0088 |    0.0252 |     0.2674 |    0.3148 |   1      |
| STIN      | GRU    | bic        | icarl         |  15 |   0.6731 |   0.6643 |      0.0088 |  -0.0038 |    0.0206 |     0.3488 |    0.1987 |   1      |
| CuCD-ID   | CNN1D  | joint      | joint_scratch |  15 |   0.9917 |   0.984  |      0.0077 |   0.0026 |    0.0153 |     0.5878 |    0.0042 |   0.2064 |
| CuCD-ID   | GRU    | bic        | icarl         |  15 |   0.9195 |   0.912  |      0.0075 |   0.003  |    0.0119 |     0.8195 |    0.0078 |   0.3516 |
| STIN      | MLP    | replay     | joint         |  15 |   0.6477 |   0.6415 |      0.0062 |  -0.0107 |    0.0231 |     0.1788 |    0.4993 |   1      |
| STIN      | GRU    | icarl      | joint         |  15 |   0.6643 |   0.6596 |      0.0047 |  -0.0132 |    0.022  |     0.1301 |    0.6161 |   1      |
| CuCD-ID   | MLP    | finetune   | lwf           |  15 |   0.2625 |   0.2588 |      0.0038 |  -0.0005 |    0.0098 |     0.3495 |    0.2734 |   1      |
| STIN      | MLP    | icarl      | joint_scratch |  15 |   0.6578 |   0.6544 |      0.0034 |  -0.0132 |    0.0199 |     0.1005 |    0.699  |   1      |
| STIN      | MLP    | finetune   | lwf           |  15 |   0.2882 |   0.2849 |      0.0033 |  -0.001  |    0.0096 |     0.2904 |    0.3899 |   1      |
| CuCD-ID   | GRU    | replay     | bic           |  15 |   0.9224 |   0.9195 |      0.0029 |  -0.009  |    0.0162 |     0.1116 |    0.6938 |   1      |
| CuCD-ID   | MLP    | bic        | icarl         |  15 |   0.9923 |   0.9903 |      0.002  |  -0      |    0.0044 |     0.4256 |    0.1025 |   1      |
| CuCD-ID   | CNN1D  | ewc        | lwf           |  15 |   0.25   |   0.25   |      0      |   0      |    0.0001 |     0.3442 |    0.5    |   1      |
| CuCD-ID   | MLP    | joint      | joint_scratch |  15 |   0.9959 |   0.9966 |     -0.0007 |  -0.0014 |   -0.0002 |    -0.5937 |    0.0363 |   1      |
| STIN      | GRU    | replay     | joint         |  15 |   0.6588 |   0.6596 |     -0.0009 |  -0.0221 |    0.0184 |    -0.0215 |    0.9377 |   1      |
| CuCD-ID   | MLP    | replay     | icarl         |  15 |   0.9894 |   0.9903 |     -0.0009 |  -0.0044 |    0.002  |    -0.1402 |    0.6371 |   1      |
| CuCD-ID   | MLP    | replay     | bic           |  15 |   0.9894 |   0.9923 |     -0.0029 |  -0.0062 |   -0.0004 |    -0.4746 |    0.0486 |   1      |
| STIN      | MLP    | replay     | full          |  15 |   0.6477 |   0.6509 |     -0.0032 |  -0.0144 |    0.0081 |    -0.1373 |    0.5994 |   1      |
| STIN      | MLP    | full       | joint_scratch |  15 |   0.6509 |   0.6544 |     -0.0035 |  -0.0182 |    0.0118 |    -0.1093 |    0.6726 |   1      |
| CuCD-ID   | MLP    | bic        | joint         |  15 |   0.9923 |   0.9959 |     -0.0036 |  -0.0045 |   -0.0027 |    -1.9671 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | bic        | joint_scratch |  15 |   0.9923 |   0.9966 |     -0.0043 |  -0.0055 |   -0.0032 |    -1.813  |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | icarl      | joint         |  15 |   0.9903 |   0.9959 |     -0.0055 |  -0.0083 |   -0.0033 |    -1.0595 |    0.0001 |   0.0132 |
| STIN      | GRU    | replay     | icarl         |  15 |   0.6588 |   0.6643 |     -0.0056 |  -0.0239 |    0.0134 |    -0.1466 |    0.5694 |   1      |
| CuCD-ID   | MLP    | icarl      | joint_scratch |  15 |   0.9903 |   0.9966 |     -0.0063 |  -0.0092 |   -0.004  |    -1.1594 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | replay     | joint         |  15 |   0.9894 |   0.9959 |     -0.0065 |  -0.0104 |   -0.0037 |    -0.9199 |    0.0001 |   0.0132 |
| STIN      | MLP    | replay     | joint_scratch |  15 |   0.6477 |   0.6544 |     -0.0066 |  -0.0203 |    0.0073 |    -0.2356 |    0.3698 |   1      |
| STIN      | MLP    | full       | icarl         |  15 |   0.6509 |   0.6578 |     -0.0069 |  -0.0192 |    0.0053 |    -0.2742 |    0.3102 |   1      |
| CuCD-ID   | MLP    | replay     | joint_scratch |  15 |   0.9894 |   0.9966 |     -0.0072 |  -0.0109 |   -0.0045 |    -1.0721 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | lwf           |  15 |   0.2406 |   0.2485 |     -0.0079 |  -0.0279 |    0.0127 |    -0.189  |    0.4791 |   1      |
| STIN      | CNN1D  | icarl      | joint_scratch |  15 |   0.6072 |   0.6161 |     -0.0089 |  -0.0298 |    0.0109 |    -0.2139 |    0.423  |   1      |
| STIN      | MLP    | replay     | icarl         |  15 |   0.6477 |   0.6578 |     -0.01   |  -0.0207 |    0.0021 |    -0.4314 |    0.1163 |   1      |
| CuCD-ID   | MLP    | finetune   | ewc           |  15 |   0.2625 |   0.2744 |     -0.0118 |  -0.0278 |    0.0022 |    -0.3816 |    0.1533 |   1      |
| STIN      | MLP    | joint      | joint_scratch |  15 |   0.6415 |   0.6544 |     -0.0129 |  -0.0335 |    0.0084 |    -0.2998 |    0.2617 |   1      |
| STIN      | GRU    | replay     | bic           |  15 |   0.6588 |   0.6731 |     -0.0143 |  -0.0327 |    0.0049 |    -0.3766 |    0.1721 |   1      |
| STIN      | GRU    | finetune   | ewc           |  15 |   0.2941 |   0.3091 |     -0.015  |  -0.0469 |    0.0136 |    -0.2373 |    0.387  |   1      |
| STIN      | MLP    | ewc        | lwf           |  15 |   0.2693 |   0.2849 |     -0.0156 |  -0.0316 |   -0.0045 |    -0.5559 |    0.0034 |   0.1678 |
| STIN      | CNN1D  | replay     | icarl         |  15 |   0.5892 |   0.6072 |     -0.018  |  -0.0369 |    0.0006 |    -0.4712 |    0.0905 |   1      |
| STIN      | CNN1D  | bic        | joint         |  15 |   0.6286 |   0.648  |     -0.0194 |  -0.044  |    0.0068 |    -0.3803 |    0.1643 |   1      |
| CuCD-ID   | GRU    | joint      | joint_scratch |  15 |   0.9683 |   0.9886 |     -0.0203 |  -0.0263 |   -0.0141 |    -1.6195 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | joint_scratch |  15 |   0.5892 |   0.6161 |     -0.0269 |  -0.0514 |   -0.0043 |    -0.5539 |    0.0502 |   1      |
| CuCD-ID   | GRU    | finetune   | lwf           |  15 |   0.2621 |   0.2961 |     -0.034  |  -0.0557 |   -0.0137 |    -0.7996 |    0.0061 |   0.2869 |
| STIN      | MLP    | full       | bic           |  15 |   0.6509 |   0.6866 |     -0.0357 |  -0.0484 |   -0.0254 |    -1.5327 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | full       | icarl         |  15 |   0.9531 |   0.9903 |     -0.0373 |  -0.0727 |   -0.0141 |    -0.5783 |    0.0006 |   0.0342 |
| STIN      | MLP    | replay     | bic           |  15 |   0.6477 |   0.6866 |     -0.0389 |  -0.0503 |   -0.0266 |    -1.6063 |    0.0002 |   0.0132 |
| CuCD-ID   | MLP    | full       | bic           |  15 |   0.9531 |   0.9923 |     -0.0392 |  -0.0746 |   -0.0163 |    -0.6034 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | bic           |  15 |   0.5892 |   0.6286 |     -0.0394 |  -0.062  |   -0.0187 |    -0.8804 |    0.0026 |   0.1365 |
| STIN      | CNN1D  | icarl      | joint         |  15 |   0.6072 |   0.648  |     -0.0408 |  -0.0633 |   -0.0186 |    -0.888  |    0.0049 |   0.2344 |
| CuCD-ID   | MLP    | full       | joint         |  15 |   0.9531 |   0.9959 |     -0.0428 |  -0.0785 |   -0.0199 |    -0.6556 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | full       | joint_scratch |  15 |   0.9531 |   0.9966 |     -0.0436 |  -0.0789 |   -0.0208 |    -0.6731 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | joint         |  15 |   0.9224 |   0.9683 |     -0.0459 |  -0.0556 |   -0.0372 |    -2.4336 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | bic        | joint         |  15 |   0.9195 |   0.9683 |     -0.0488 |  -0.0603 |   -0.0376 |    -2.1134 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | icarl      | joint         |  15 |   0.912  |   0.9683 |     -0.0563 |  -0.0684 |   -0.0449 |    -2.3388 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | joint         |  15 |   0.5892 |   0.648  |     -0.0588 |  -0.0859 |   -0.0331 |    -1.0718 |    0.0009 |   0.047  |
| STIN      | GRU    | full       | joint_scratch |  15 |   0.5889 |   0.6486 |     -0.0598 |  -0.0824 |   -0.0369 |    -1.2809 |    0.0003 |   0.0189 |
| STIN      | CNN1D  | full       | icarl         |  15 |   0.5416 |   0.6072 |     -0.0656 |  -0.0837 |   -0.0484 |    -1.8141 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | joint_scratch |  15 |   0.9224 |   0.9886 |     -0.0662 |  -0.0757 |   -0.0571 |    -3.4166 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | bic        | joint_scratch |  15 |   0.9195 |   0.9886 |     -0.0691 |  -0.08   |   -0.0581 |    -3.0721 |    0.0001 |   0.0132 |
| STIN      | GRU    | full       | joint         |  15 |   0.5889 |   0.6596 |     -0.0708 |  -0.0997 |   -0.044  |    -1.2425 |    0.0002 |   0.0132 |
| STIN      | CNN1D  | full       | joint_scratch |  15 |   0.5416 |   0.6161 |     -0.0744 |  -0.0977 |   -0.0494 |    -1.5168 |    0.0002 |   0.0154 |
| STIN      | GRU    | full       | icarl         |  15 |   0.5889 |   0.6643 |     -0.0755 |  -0.0952 |   -0.0556 |    -1.8762 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | icarl      | joint_scratch |  15 |   0.912  |   0.9886 |     -0.0766 |  -0.0878 |   -0.0655 |    -3.3334 |    0.0001 |   0.0132 |
| STIN      | GRU    | full       | bic           |  15 |   0.5889 |   0.6731 |     -0.0842 |  -0.1097 |   -0.0561 |    -1.529  |    0.0003 |   0.0189 |
| STIN      | CNN1D  | full       | bic           |  15 |   0.5416 |   0.6286 |     -0.087  |  -0.1097 |   -0.0655 |    -1.9137 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | icarl         |  15 |   0.7089 |   0.796  |     -0.0871 |  -0.1274 |   -0.0464 |    -1.0377 |    0.0015 |   0.0791 |
| STIN      | CNN1D  | full       | joint         |  15 |   0.5416 |   0.648  |     -0.1064 |  -0.1296 |   -0.0796 |    -2.105  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | bic        | joint_scratch |  15 |   0.8729 |   0.984  |     -0.1111 |  -0.1389 |   -0.0859 |    -2.0106 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | bic        | joint         |  15 |   0.8729 |   0.9917 |     -0.1188 |  -0.1463 |   -0.0937 |    -2.1843 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | bic           |  15 |   0.7089 |   0.8729 |     -0.1639 |  -0.2036 |   -0.1203 |    -1.9204 |    0.0002 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | ewc           |  15 |   0.2621 |   0.4384 |     -0.1763 |  -0.2117 |   -0.1413 |    -2.4428 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | icarl         |  15 |   0.6194 |   0.796  |     -0.1766 |  -0.2258 |   -0.1313 |    -1.8245 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | icarl      | joint_scratch |  15 |   0.796  |   0.984  |     -0.188  |  -0.2129 |   -0.1642 |    -3.7548 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | full          |  15 |   0.4384 |   0.6275 |     -0.1891 |  -0.2458 |   -0.1303 |    -1.5831 |    0.0002 |   0.0132 |
| CuCD-ID   | CNN1D  | icarl      | joint         |  15 |   0.796  |   0.9917 |     -0.1957 |  -0.2203 |   -0.1723 |    -3.9323 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | bic           |  15 |   0.6194 |   0.8729 |     -0.2535 |  -0.302  |   -0.2109 |    -2.7249 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | joint_scratch |  15 |   0.7089 |   0.984  |     -0.2751 |  -0.3155 |   -0.2312 |    -3.1732 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | full          |  15 |   0.262  |   0.5416 |     -0.2797 |  -0.3026 |   -0.2579 |    -6.2117 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | full          |  15 |   0.3091 |   0.5889 |     -0.2797 |  -0.3092 |   -0.2496 |    -4.5622 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | joint         |  15 |   0.7089 |   0.9917 |     -0.2828 |  -0.3225 |   -0.2389 |    -3.2963 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | icarl         |  15 |   0.6275 |   0.912  |     -0.2845 |  -0.3243 |   -0.2455 |    -3.5241 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | bic           |  15 |   0.6275 |   0.9195 |     -0.292  |  -0.3318 |   -0.2536 |    -3.6513 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | full          |  15 |   0.2941 |   0.5889 |     -0.2948 |  -0.3205 |   -0.2701 |    -5.6359 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | full          |  15 |   0.2406 |   0.5416 |     -0.3011 |  -0.3211 |   -0.2815 |    -7.4947 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | replay        |  15 |   0.262  |   0.5892 |     -0.3273 |  -0.3521 |   -0.3027 |    -6.5087 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | joint_scratch |  15 |   0.3091 |   0.6486 |     -0.3395 |  -0.367  |   -0.3097 |    -5.8102 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | full          |  15 |   0.2797 |   0.6194 |     -0.3397 |  -0.3851 |   -0.2892 |    -3.4187 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | joint         |  15 |   0.6275 |   0.9683 |     -0.3408 |  -0.3772 |   -0.3063 |    -4.6889 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | icarl         |  15 |   0.262  |   0.6072 |     -0.3453 |  -0.3619 |   -0.3289 |   -10.1325 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | joint         |  15 |   0.3091 |   0.6596 |     -0.3505 |  -0.3874 |   -0.3133 |    -4.6398 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | joint         |  15 |   0.2882 |   0.6415 |     -0.3533 |  -0.3701 |   -0.337  |   -10.4136 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | joint_scratch |  15 |   0.262  |   0.6161 |     -0.3541 |  -0.3771 |   -0.3313 |    -7.5071 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | joint_scratch |  15 |   0.2941 |   0.6486 |     -0.3545 |  -0.374  |   -0.3327 |    -8.3011 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | icarl         |  15 |   0.3091 |   0.6643 |     -0.3552 |  -0.3846 |   -0.3269 |    -6.0126 |    0.0001 |   0.0132 |
| STIN      | MLP    | lwf        | joint         |  15 |   0.2849 |   0.6415 |     -0.3566 |  -0.3713 |   -0.3428 |   -12.0873 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | lwf        | icarl         |  15 |   0.2485 |   0.6072 |     -0.3587 |  -0.3753 |   -0.342  |   -10.4923 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | replay        |  15 |   0.2882 |   0.6477 |     -0.3595 |  -0.3678 |   -0.3509 |   -20.9455 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | joint_scratch |  15 |   0.6275 |   0.9886 |     -0.361  |  -0.4008 |   -0.323  |    -4.5098 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | full          |  15 |   0.2882 |   0.6509 |     -0.3627 |  -0.3734 |   -0.3513 |   -15.9681 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | bic           |  15 |   0.3091 |   0.6731 |     -0.364  |  -0.3982 |   -0.3301 |    -5.2265 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | joint_scratch |  15 |   0.6194 |   0.984  |     -0.3646 |  -0.4123 |   -0.3223 |    -3.9198 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | replay        |  15 |   0.2941 |   0.6588 |     -0.3646 |  -0.3952 |   -0.3365 |    -5.9674 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | full          |  15 |   0.2621 |   0.6275 |     -0.3654 |  -0.399  |   -0.3295 |    -5.1091 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | joint         |  15 |   0.2941 |   0.6596 |     -0.3655 |  -0.3998 |   -0.331  |    -5.1839 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | joint_scratch |  15 |   0.2882 |   0.6544 |     -0.3662 |  -0.3832 |   -0.3481 |   -10.2495 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | icarl         |  15 |   0.2406 |   0.6072 |     -0.3667 |  -0.3861 |   -0.3444 |    -8.5935 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | bic           |  15 |   0.262  |   0.6286 |     -0.3667 |  -0.3944 |   -0.3395 |    -6.4648 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | lwf        | joint_scratch |  15 |   0.2485 |   0.6161 |     -0.3676 |  -0.3875 |   -0.3475 |    -8.9257 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | full          |  15 |   0.25   |   0.6194 |     -0.3693 |  -0.4101 |   -0.3239 |    -4.1379 |    0.0001 |   0.0132 |
| STIN      | MLP    | lwf        | joint_scratch |  15 |   0.2849 |   0.6544 |     -0.3694 |  -0.3837 |   -0.3546 |   -12.3327 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | icarl         |  15 |   0.2882 |   0.6578 |     -0.3696 |  -0.3785 |   -0.3593 |   -18.9137 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | icarl         |  15 |   0.2941 |   0.6643 |     -0.3702 |  -0.3951 |   -0.3466 |    -7.4048 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | joint         |  15 |   0.2693 |   0.6415 |     -0.3722 |  -0.3972 |   -0.3522 |    -8.0694 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | joint         |  15 |   0.6194 |   0.9917 |     -0.3723 |  -0.418  |   -0.3312 |    -4.1495 |    0.0001 |   0.0132 |
| STIN      | MLP    | lwf        | icarl         |  15 |   0.2849 |   0.6578 |     -0.3728 |  -0.379  |   -0.3658 |   -27.7047 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | joint_scratch |  15 |   0.2406 |   0.6161 |     -0.3755 |  -0.3932 |   -0.356  |    -9.8123 |    0.0001 |   0.0132 |
| STIN      | GRU    | lwf        | joint_scratch |  15 |   0.2703 |   0.6486 |     -0.3784 |  -0.4036 |   -0.3488 |    -6.8267 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | bic           |  15 |   0.2941 |   0.6731 |     -0.379  |  -0.4094 |   -0.3494 |    -6.1096 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | lwf        | bic           |  15 |   0.2485 |   0.6286 |     -0.3801 |  -0.4059 |   -0.3542 |    -7.1031 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | full          |  15 |   0.2693 |   0.6509 |     -0.3816 |  -0.3977 |   -0.3672 |   -12.2048 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | joint_scratch |  15 |   0.2693 |   0.6544 |     -0.3851 |  -0.4084 |   -0.3637 |    -8.3949 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | joint         |  15 |   0.262  |   0.648  |     -0.3861 |  -0.4075 |   -0.3641 |    -8.5578 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | bic           |  15 |   0.2406 |   0.6286 |     -0.3881 |  -0.4142 |   -0.3609 |    -7.1117 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | icarl         |  15 |   0.2693 |   0.6578 |     -0.3885 |  -0.406  |   -0.3745 |   -12.059  |    0.0001 |   0.0132 |
| STIN      | GRU    | lwf        | joint         |  15 |   0.2703 |   0.6596 |     -0.3894 |  -0.4202 |   -0.3575 |    -6.0397 |    0.0001 |   0.0132 |
| STIN      | GRU    | lwf        | icarl         |  15 |   0.2703 |   0.6643 |     -0.3941 |  -0.4154 |   -0.3702 |    -8.5114 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | bic           |  15 |   0.2882 |   0.6866 |     -0.3984 |  -0.4068 |   -0.3898 |   -23.0085 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | lwf        | joint         |  15 |   0.2485 |   0.648  |     -0.3996 |  -0.4209 |   -0.3776 |    -8.9074 |    0.0001 |   0.0132 |
| STIN      | MLP    | lwf        | bic           |  15 |   0.2849 |   0.6866 |     -0.4017 |  -0.4085 |   -0.3946 |   -28.4368 |    0.0001 |   0.0132 |
| STIN      | GRU    | lwf        | bic           |  15 |   0.2703 |   0.6731 |     -0.4028 |  -0.4311 |   -0.3738 |    -6.7848 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | joint         |  15 |   0.2406 |   0.648  |     -0.4075 |  -0.4262 |   -0.3876 |   -10.3224 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | bic           |  15 |   0.2693 |   0.6866 |     -0.4173 |  -0.434  |   -0.4031 |   -13.215  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | replay        |  15 |   0.2797 |   0.7089 |     -0.4293 |  -0.4769 |   -0.3836 |    -4.4496 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | icarl         |  15 |   0.4384 |   0.912  |     -0.4736 |  -0.5141 |   -0.4346 |    -5.8038 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | bic           |  15 |   0.4384 |   0.9195 |     -0.4811 |  -0.5222 |   -0.4413 |    -5.8012 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | icarl         |  15 |   0.2797 |   0.796  |     -0.5163 |  -0.5486 |   -0.4809 |    -7.4704 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | joint         |  15 |   0.4384 |   0.9683 |     -0.5299 |  -0.5632 |   -0.4968 |    -7.7303 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | icarl         |  15 |   0.25   |   0.796  |     -0.546  |  -0.569  |   -0.522  |   -11.1956 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | lwf        | icarl         |  15 |   0.25   |   0.796  |     -0.546  |  -0.5691 |   -0.5221 |   -11.2088 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | joint_scratch |  15 |   0.4384 |   0.9886 |     -0.5501 |  -0.5836 |   -0.5177 |    -8.0789 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | bic           |  15 |   0.2797 |   0.8729 |     -0.5932 |  -0.6287 |   -0.5545 |    -7.795  |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | lwf        | icarl         |  15 |   0.2961 |   0.912  |     -0.6159 |  -0.6353 |   -0.5952 |   -14.7998 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | bic           |  15 |   0.25   |   0.8729 |     -0.6228 |  -0.6475 |   -0.5961 |   -11.732  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | lwf        | bic           |  15 |   0.25   |   0.8729 |     -0.6229 |  -0.6475 |   -0.5962 |   -11.745  |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | lwf        | bic           |  15 |   0.2961 |   0.9195 |     -0.6234 |  -0.6421 |   -0.6033 |   -15.6448 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | icarl         |  15 |   0.2621 |   0.912  |     -0.6499 |  -0.6671 |   -0.6282 |   -16.1409 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | bic           |  15 |   0.2621 |   0.9195 |     -0.6573 |  -0.6738 |   -0.6371 |   -17.2977 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | replay        |  15 |   0.2621 |   0.9224 |     -0.6602 |  -0.6736 |   -0.6448 |   -22.334  |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | lwf        | joint         |  15 |   0.2961 |   0.9683 |     -0.6722 |  -0.6876 |   -0.6545 |   -19.7795 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | full          |  15 |   0.2744 |   0.9531 |     -0.6787 |  -0.7153 |   -0.6295 |    -7.5612 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | full          |  15 |   0.2625 |   0.9531 |     -0.6905 |  -0.7174 |   -0.6535 |   -10.1338 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | lwf        | joint_scratch |  15 |   0.2961 |   0.9886 |     -0.6924 |  -0.7089 |   -0.6735 |   -19.2773 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | joint_scratch |  15 |   0.2797 |   0.984  |     -0.7043 |  -0.7194 |   -0.6862 |   -20.7908 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | joint         |  15 |   0.2621 |   0.9683 |     -0.7061 |  -0.7173 |   -0.692  |   -27.3857 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | joint         |  15 |   0.2797 |   0.9917 |     -0.712  |  -0.7269 |   -0.694  |   -21.1633 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | icarl         |  15 |   0.2744 |   0.9903 |     -0.716  |  -0.7376 |   -0.6847 |   -12.8613 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | bic           |  15 |   0.2744 |   0.9923 |     -0.718  |  -0.7398 |   -0.6864 |   -12.8605 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | joint         |  15 |   0.2744 |   0.9959 |     -0.7215 |  -0.7436 |   -0.6899 |   -12.9164 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | joint_scratch |  15 |   0.2744 |   0.9966 |     -0.7223 |  -0.7445 |   -0.6907 |   -12.8786 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | joint_scratch |  15 |   0.2621 |   0.9886 |     -0.7264 |  -0.7363 |   -0.711  |   -26.0926 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | replay        |  15 |   0.2625 |   0.9894 |     -0.7269 |  -0.7399 |   -0.7067 |   -20.2614 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | icarl         |  15 |   0.2625 |   0.9903 |     -0.7278 |  -0.7406 |   -0.7071 |   -19.8159 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | bic           |  15 |   0.2625 |   0.9923 |     -0.7298 |  -0.7422 |   -0.7085 |   -19.7078 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | lwf        | icarl         |  15 |   0.2588 |   0.9903 |     -0.7315 |  -0.7411 |   -0.7149 |   -24.0852 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | joint         |  15 |   0.2625 |   0.9959 |     -0.7333 |  -0.7457 |   -0.7121 |   -19.7828 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | lwf        | bic           |  15 |   0.2588 |   0.9923 |     -0.7335 |  -0.7424 |   -0.7172 |   -24.0731 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | joint_scratch |  15 |   0.25   |   0.984  |     -0.734  |  -0.7389 |   -0.7266 |   -57.6526 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | lwf        | joint_scratch |  15 |   0.25   |   0.984  |     -0.734  |  -0.739  |   -0.7266 |   -57.6277 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | joint_scratch |  15 |   0.2625 |   0.9966 |     -0.7341 |  -0.7462 |   -0.7131 |   -19.9122 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | lwf        | joint         |  15 |   0.2588 |   0.9959 |     -0.7371 |  -0.7458 |   -0.7208 |   -24.1313 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | lwf        | joint_scratch |  15 |   0.2588 |   0.9966 |     -0.7378 |  -0.7465 |   -0.7215 |   -24.179  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | joint         |  15 |   0.25   |   0.9917 |     -0.7417 |  -0.7427 |   -0.7407 |  -356.738  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | lwf        | joint         |  15 |   0.25   |   0.9917 |     -0.7417 |  -0.7428 |   -0.7407 |  -345.814  |    0.0001 |   0.0132 |

### 7.7 Exact paired permutation + Holm (matched seeds, Macro-F1 (primary)): 164/216 pre-specified pairwise comparisons significant at Holm p<0.05

The family is all unordered method pairs within each dataset x architecture (216 comparisons per metric), fixed before looking at results; Holm correction is applied across the whole family. The exact sign-flip test has n=15 paired seeds. This is exploratory, not a winner-selected headline list; significance is not scientific importance, so read the effect sizes and confidence intervals rather than the count of significant comparisons.

| dataset   | arch   | method_a   | method_b      |   n |   mean_a |   mean_b |   mean_diff |   ci_low |   ci_high |   cohens_d |   p_exact |   p_holm |
|:----------|:-------|:-----------|:--------------|----:|---------:|---------:|------------:|---------:|----------:|-----------:|----------:|---------:|
| CuCD-ID   | MLP    | replay     | lwf           |  15 |   0.9906 |   0.0879 |      0.9028 |   0.8762 |    0.9179 |    18.5238 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | replay     | ewc           |  15 |   0.9906 |   0.1067 |      0.8839 |   0.84   |    0.9133 |    11.3274 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | full       | lwf           |  15 |   0.9562 |   0.0879 |      0.8683 |   0.8262 |    0.8991 |    11.2951 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | lwf           |  15 |   0.9343 |   0.1644 |      0.7699 |   0.7496 |    0.7895 |    19.0918 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | lwf           |  15 |   0.7269 |   0.0722 |      0.6547 |   0.6137 |    0.6988 |     7.4724 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | ewc           |  15 |   0.7269 |   0.0725 |      0.6544 |   0.6135 |    0.6985 |     7.4728 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | lwf           |  15 |   0.6343 |   0.0722 |      0.5621 |   0.5126 |    0.6059 |     5.7764 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | ewc           |  15 |   0.9343 |   0.4386 |      0.4957 |   0.4497 |    0.5418 |     5.18   |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | lwf           |  15 |   0.6438 |   0.1644 |      0.4794 |   0.432  |    0.521  |     5.2445 |    0.0001 |   0.0132 |
| STIN      | GRU    | replay     | lwf           |  15 |   0.5908 |   0.1493 |      0.4415 |   0.4079 |    0.4752 |     6.4021 |    0.0001 |   0.0132 |
| STIN      | MLP    | replay     | ewc           |  15 |   0.5809 |   0.1474 |      0.4335 |   0.4183 |    0.4492 |    13.8787 |    0.0001 |   0.0132 |
| STIN      | MLP    | full       | lwf           |  15 |   0.5822 |   0.1635 |      0.4187 |   0.4076 |    0.4289 |    19.0147 |    0.0001 |   0.0132 |
| STIN      | MLP    | replay     | lwf           |  15 |   0.5809 |   0.1635 |      0.4174 |   0.4047 |    0.4303 |    15.9867 |    0.0001 |   0.0132 |
| STIN      | GRU    | replay     | ewc           |  15 |   0.5908 |   0.1882 |      0.4026 |   0.3757 |    0.43   |     7.2053 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | ewc           |  15 |   0.5076 |   0.117  |      0.3906 |   0.36   |    0.4201 |     6.2943 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | lwf           |  15 |   0.5076 |   0.1235 |      0.3841 |   0.3531 |    0.4147 |     6.1135 |    0.0001 |   0.0132 |
| STIN      | GRU    | full       | lwf           |  15 |   0.515  |   0.1493 |      0.3657 |   0.3274 |    0.4009 |     4.9217 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | full       | lwf           |  15 |   0.4651 |   0.1235 |      0.3416 |   0.3199 |    0.365  |     7.4497 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | full          |  15 |   0.9343 |   0.6438 |      0.2905 |   0.2519 |    0.3314 |     3.5798 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | lwf           |  15 |   0.4386 |   0.1644 |      0.2742 |   0.2257 |    0.3254 |     2.6781 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | full          |  15 |   0.7269 |   0.6343 |      0.0926 |   0.0421 |    0.1478 |     0.8504 |    0.0031 |   0.1404 |
| STIN      | GRU    | replay     | full          |  15 |   0.5908 |   0.515  |      0.0758 |   0.0475 |    0.1084 |     1.2192 |    0.0002 |   0.0154 |
| CuCD-ID   | CNN1D  | bic        | icarl         |  15 |   0.8857 |   0.8186 |      0.0671 |   0.0487 |    0.0844 |     1.8321 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | lwf           |  15 |   0.1331 |   0.0722 |      0.0609 |   0.0333 |    0.0925 |     1.0066 |    0.0005 |   0.0278 |
| CuCD-ID   | CNN1D  | finetune   | ewc           |  15 |   0.1331 |   0.0725 |      0.0606 |   0.033  |    0.0921 |     1.0015 |    0.001  |   0.0508 |
| STIN      | MLP    | bic        | joint         |  15 |   0.6476 |   0.5878 |      0.0598 |   0.0402 |    0.0786 |     1.4994 |    0.0002 |   0.0154 |
| STIN      | MLP    | bic        | icarl         |  15 |   0.6476 |   0.5897 |      0.0579 |   0.045  |    0.0705 |     2.2179 |    0.0001 |   0.0132 |
| STIN      | MLP    | bic        | joint_scratch |  15 |   0.6476 |   0.5975 |      0.0501 |   0.0223 |    0.0773 |     0.8886 |    0.0047 |   0.2068 |
| STIN      | CNN1D  | joint      | joint_scratch |  15 |   0.5929 |   0.5497 |      0.0432 |   0.0258 |    0.0596 |     1.2222 |    0.0009 |   0.0453 |
| STIN      | CNN1D  | replay     | full          |  15 |   0.5076 |   0.4651 |      0.0425 |   0.0143 |    0.0695 |     0.7387 |    0.014  |   0.5591 |
| STIN      | GRU    | ewc        | lwf           |  15 |   0.1882 |   0.1493 |      0.0389 |   0.0083 |    0.0714 |     0.6062 |    0.0323 |   1      |
| STIN      | CNN1D  | bic        | icarl         |  15 |   0.5744 |   0.537  |      0.0374 |   0.0169 |    0.057  |     0.8982 |    0.0049 |   0.2068 |
| STIN      | GRU    | bic        | joint_scratch |  15 |   0.6204 |   0.5855 |      0.035  |   0.0075 |    0.0619 |     0.6253 |    0.0309 |   1      |
| CuCD-ID   | MLP    | replay     | full          |  15 |   0.9906 |   0.9562 |      0.0345 |   0.0122 |    0.0707 |     0.5215 |    0.0001 |   0.0132 |
| STIN      | GRU    | bic        | joint         |  15 |   0.6204 |   0.5923 |      0.0281 |   0.0004 |    0.0557 |     0.4917 |    0.0778 |   1      |
| STIN      | GRU    | bic        | icarl         |  15 |   0.6204 |   0.5925 |      0.0279 |   0.0131 |    0.0427 |     0.9232 |    0.0048 |   0.2068 |
| CuCD-ID   | GRU    | replay     | icarl         |  15 |   0.9343 |   0.907  |      0.0273 |   0.0171 |    0.038  |     1.2672 |    0.0002 |   0.0154 |
| STIN      | CNN1D  | bic        | joint_scratch |  15 |   0.5744 |   0.5497 |      0.0247 |  -0.0047 |    0.0533 |     0.4179 |    0.1275 |   1      |
| STIN      | GRU    | finetune   | lwf           |  15 |   0.1733 |   0.1493 |      0.024  |  -0.0107 |    0.0529 |     0.3672 |    0.1763 |   1      |
| STIN      | MLP    | finetune   | ewc           |  15 |   0.168  |   0.1474 |      0.0206 |   0.0092 |    0.0338 |     0.8223 |    0.0013 |   0.0641 |
| CuCD-ID   | MLP    | ewc        | lwf           |  15 |   0.1067 |   0.0879 |      0.0188 |   0.0032 |    0.0375 |     0.5276 |    0.0457 |   1      |
| CuCD-ID   | GRU    | bic        | icarl         |  15 |   0.9226 |   0.907  |      0.0156 |   0.0106 |    0.0208 |     1.4775 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | ewc           |  15 |   0.13   |   0.117  |      0.013  |  -0.0058 |    0.0313 |     0.3454 |    0.1998 |   1      |
| CuCD-ID   | GRU    | replay     | bic           |  15 |   0.9343 |   0.9226 |      0.0117 |   0.0017 |    0.0223 |     0.5549 |    0.0448 |   1      |
| CuCD-ID   | CNN1D  | joint      | joint_scratch |  15 |   0.9923 |   0.9852 |      0.0071 |   0.0026 |    0.0137 |     0.6204 |    0.002  |   0.0947 |
| STIN      | GRU    | icarl      | joint_scratch |  15 |   0.5925 |   0.5855 |      0.0071 |  -0.0121 |    0.0267 |     0.178  |    0.4946 |   1      |
| STIN      | GRU    | joint      | joint_scratch |  15 |   0.5923 |   0.5855 |      0.0068 |  -0.0143 |    0.028  |     0.1579 |    0.5455 |   1      |
| STIN      | CNN1D  | finetune   | lwf           |  15 |   0.13   |   0.1235 |      0.0065 |  -0.007  |    0.0197 |     0.2357 |    0.3728 |   1      |
| CuCD-ID   | MLP    | finetune   | lwf           |  15 |   0.094  |   0.0879 |      0.0061 |  -0.0023 |    0.0186 |     0.2749 |    0.3164 |   1      |
| STIN      | GRU    | replay     | joint_scratch |  15 |   0.5908 |   0.5855 |      0.0053 |  -0.0169 |    0.0281 |     0.1141 |    0.6645 |   1      |
| STIN      | MLP    | finetune   | lwf           |  15 |   0.168  |   0.1635 |      0.0045 |  -0.0012 |    0.0134 |     0.2798 |    0.3422 |   1      |
| CuCD-ID   | MLP    | bic        | icarl         |  15 |   0.9924 |   0.9893 |      0.0031 |   0.0013 |    0.0052 |     0.7953 |    0.0049 |   0.2068 |
| STIN      | MLP    | icarl      | joint         |  15 |   0.5897 |   0.5878 |      0.0019 |  -0.0178 |    0.0215 |     0.046  |    0.8589 |   1      |
| CuCD-ID   | MLP    | replay     | icarl         |  15 |   0.9906 |   0.9893 |      0.0014 |  -0.0012 |    0.0038 |     0.2666 |    0.3232 |   1      |
| CuCD-ID   | CNN1D  | ewc        | lwf           |  15 |   0.0725 |   0.0722 |      0.0003 |   0      |    0.0008 |     0.4047 |    0.125  |   1      |
| STIN      | GRU    | icarl      | joint         |  15 |   0.5925 |   0.5923 |      0.0002 |  -0.0215 |    0.022  |     0.005  |    0.9847 |   1      |
| CuCD-ID   | MLP    | joint      | joint_scratch |  15 |   0.9963 |   0.9969 |     -0.0006 |  -0.0012 |   -0      |    -0.4969 |    0.0739 |   1      |
| STIN      | MLP    | replay     | full          |  15 |   0.5809 |   0.5822 |     -0.0014 |  -0.0154 |    0.0123 |    -0.0484 |    0.8526 |   1      |
| STIN      | GRU    | replay     | joint         |  15 |   0.5908 |   0.5923 |     -0.0015 |  -0.0267 |    0.0221 |    -0.0305 |    0.9068 |   1      |
| STIN      | GRU    | replay     | icarl         |  15 |   0.5908 |   0.5925 |     -0.0018 |  -0.0221 |    0.0198 |    -0.0412 |    0.8487 |   1      |
| CuCD-ID   | MLP    | replay     | bic           |  15 |   0.9906 |   0.9924 |     -0.0018 |  -0.0041 |    0.0002 |    -0.4067 |    0.1306 |   1      |
| CuCD-ID   | MLP    | bic        | joint         |  15 |   0.9924 |   0.9963 |     -0.0038 |  -0.0049 |   -0.0029 |    -1.8649 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | bic        | joint_scratch |  15 |   0.9924 |   0.9969 |     -0.0044 |  -0.0057 |   -0.0032 |    -1.762  |    0.0001 |   0.0132 |
| STIN      | MLP    | full       | joint         |  15 |   0.5822 |   0.5878 |     -0.0056 |  -0.0282 |    0.0154 |    -0.1251 |    0.6361 |   1      |
| CuCD-ID   | MLP    | replay     | joint         |  15 |   0.9906 |   0.9963 |     -0.0056 |  -0.0086 |   -0.0034 |    -1.0407 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | replay     | joint_scratch |  15 |   0.9906 |   0.9969 |     -0.0062 |  -0.009  |   -0.0041 |    -1.215  |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | lwf           |  15 |   0.117  |   0.1235 |     -0.0065 |  -0.0285 |    0.0159 |    -0.1434 |    0.5805 |   1      |
| STIN      | MLP    | replay     | joint         |  15 |   0.5809 |   0.5878 |     -0.007  |  -0.0303 |    0.0159 |    -0.1462 |    0.5735 |   1      |
| CuCD-ID   | MLP    | icarl      | joint         |  15 |   0.9893 |   0.9963 |     -0.007  |  -0.0095 |   -0.0049 |    -1.4569 |    0.0001 |   0.0132 |
| STIN      | MLP    | full       | icarl         |  15 |   0.5822 |   0.5897 |     -0.0075 |  -0.0208 |    0.0055 |    -0.277  |    0.304  |   1      |
| CuCD-ID   | MLP    | icarl      | joint_scratch |  15 |   0.9893 |   0.9969 |     -0.0076 |  -0.0102 |   -0.0054 |    -1.5134 |    0.0001 |   0.0132 |
| STIN      | MLP    | icarl      | joint_scratch |  15 |   0.5897 |   0.5975 |     -0.0078 |  -0.0267 |    0.0111 |    -0.2026 |    0.4402 |   1      |
| STIN      | MLP    | replay     | icarl         |  15 |   0.5809 |   0.5897 |     -0.0088 |  -0.0232 |    0.0075 |    -0.2824 |    0.29   |   1      |
| STIN      | MLP    | joint      | joint_scratch |  15 |   0.5878 |   0.5975 |     -0.0097 |  -0.0355 |    0.0154 |    -0.1856 |    0.482  |   1      |
| STIN      | CNN1D  | icarl      | joint_scratch |  15 |   0.537  |   0.5497 |     -0.0127 |  -0.0354 |    0.0086 |    -0.2815 |    0.2972 |   1      |
| CuCD-ID   | MLP    | finetune   | ewc           |  15 |   0.094  |   0.1067 |     -0.0127 |  -0.0351 |    0.0092 |    -0.2828 |    0.2622 |   1      |
| STIN      | GRU    | finetune   | ewc           |  15 |   0.1733 |   0.1882 |     -0.0148 |  -0.048  |    0.0149 |    -0.2256 |    0.4105 |   1      |
| STIN      | MLP    | full       | joint_scratch |  15 |   0.5822 |   0.5975 |     -0.0153 |  -0.0338 |    0.0034 |    -0.3929 |    0.1489 |   1      |
| STIN      | MLP    | ewc        | lwf           |  15 |   0.1474 |   0.1635 |     -0.0161 |  -0.0284 |   -0.0067 |    -0.7219 |    0.0008 |   0.0428 |
| STIN      | MLP    | replay     | joint_scratch |  15 |   0.5809 |   0.5975 |     -0.0167 |  -0.0335 |    0.0002 |    -0.4815 |    0.0836 |   1      |
| CuCD-ID   | GRU    | joint      | joint_scratch |  15 |   0.9723 |   0.9897 |     -0.0174 |  -0.0225 |   -0.012  |    -1.6114 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | bic        | joint         |  15 |   0.5744 |   0.5929 |     -0.0185 |  -0.0465 |    0.0129 |    -0.3092 |    0.2505 |   1      |
| STIN      | CNN1D  | replay     | icarl         |  15 |   0.5076 |   0.537  |     -0.0293 |  -0.0565 |   -0.0021 |    -0.5233 |    0.0634 |   1      |
| STIN      | GRU    | replay     | bic           |  15 |   0.5908 |   0.6204 |     -0.0297 |  -0.0533 |   -0.0064 |    -0.6238 |    0.0265 |   1      |
| CuCD-ID   | MLP    | full       | icarl         |  15 |   0.9562 |   0.9893 |     -0.0331 |  -0.0696 |   -0.0106 |    -0.4956 |    0.0006 |   0.0342 |
| CuCD-ID   | MLP    | full       | bic           |  15 |   0.9562 |   0.9924 |     -0.0362 |  -0.0727 |   -0.0139 |    -0.539  |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | joint         |  15 |   0.9343 |   0.9723 |     -0.0379 |  -0.0458 |   -0.0306 |    -2.4462 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | full       | joint         |  15 |   0.9562 |   0.9963 |     -0.0401 |  -0.0768 |   -0.0177 |    -0.5942 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | full       | joint_scratch |  15 |   0.9562 |   0.9969 |     -0.0407 |  -0.0771 |   -0.0184 |    -0.6085 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | joint_scratch |  15 |   0.5076 |   0.5497 |     -0.042  |  -0.0738 |   -0.0113 |    -0.6458 |    0.0261 |   1      |
| CuCD-ID   | GRU    | bic        | joint         |  15 |   0.9226 |   0.9723 |     -0.0497 |  -0.0593 |   -0.04   |    -2.5053 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | replay     | joint_scratch |  15 |   0.9343 |   0.9897 |     -0.0553 |  -0.0629 |   -0.0481 |    -3.5715 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | icarl      | joint         |  15 |   0.537  |   0.5929 |     -0.0559 |  -0.0824 |   -0.0291 |    -1.0257 |    0.0016 |   0.0791 |
| CuCD-ID   | GRU    | finetune   | lwf           |  15 |   0.104  |   0.1644 |     -0.0604 |  -0.0894 |   -0.029  |    -0.9871 |    0.0034 |   0.1511 |
| CuCD-ID   | GRU    | icarl      | joint         |  15 |   0.907  |   0.9723 |     -0.0653 |  -0.0762 |   -0.0544 |    -2.9041 |    0.0001 |   0.0132 |
| STIN      | MLP    | full       | bic           |  15 |   0.5822 |   0.6476 |     -0.0654 |  -0.0822 |   -0.0494 |    -1.9704 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | replay     | bic           |  15 |   0.5076 |   0.5744 |     -0.0667 |  -0.0995 |   -0.0361 |    -1.0075 |    0.0016 |   0.0778 |
| STIN      | MLP    | replay     | bic           |  15 |   0.5809 |   0.6476 |     -0.0668 |  -0.0879 |   -0.044  |    -1.4919 |    0.0003 |   0.0177 |
| CuCD-ID   | GRU    | bic        | joint_scratch |  15 |   0.9226 |   0.9897 |     -0.0671 |  -0.0767 |   -0.0574 |    -3.4182 |    0.0001 |   0.0132 |
| STIN      | GRU    | full       | joint_scratch |  15 |   0.515  |   0.5855 |     -0.0705 |  -0.0973 |   -0.0446 |    -1.3005 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | full       | icarl         |  15 |   0.4651 |   0.537  |     -0.0718 |  -0.089  |   -0.0556 |    -2.1044 |    0.0001 |   0.0132 |
| STIN      | GRU    | full       | joint         |  15 |   0.515  |   0.5923 |     -0.0773 |  -0.1092 |   -0.0479 |    -1.2419 |    0.0002 |   0.0132 |
| STIN      | GRU    | full       | icarl         |  15 |   0.515  |   0.5925 |     -0.0776 |  -0.098  |   -0.0561 |    -1.8171 |    0.0002 |   0.0154 |
| CuCD-ID   | GRU    | icarl      | joint_scratch |  15 |   0.907  |   0.9897 |     -0.0827 |  -0.0932 |   -0.0721 |    -3.7962 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | full       | joint_scratch |  15 |   0.4651 |   0.5497 |     -0.0845 |  -0.1101 |   -0.0564 |    -1.5498 |    0.0002 |   0.0154 |
| STIN      | CNN1D  | replay     | joint         |  15 |   0.5076 |   0.5929 |     -0.0852 |  -0.1203 |   -0.0505 |    -1.1808 |    0.0006 |   0.0342 |
| CuCD-ID   | CNN1D  | replay     | icarl         |  15 |   0.7269 |   0.8186 |     -0.0917 |  -0.1346 |   -0.0493 |    -1.0521 |    0.0012 |   0.0623 |
| CuCD-ID   | CNN1D  | bic        | joint_scratch |  15 |   0.8857 |   0.9852 |     -0.0995 |  -0.1246 |   -0.0774 |    -2.0307 |    0.0001 |   0.0132 |
| STIN      | GRU    | full       | bic           |  15 |   0.515  |   0.6204 |     -0.1055 |  -0.1336 |   -0.0747 |    -1.7459 |    0.0002 |   0.0132 |
| CuCD-ID   | CNN1D  | bic        | joint         |  15 |   0.8857 |   0.9923 |     -0.1066 |  -0.131  |   -0.0846 |    -2.2192 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | full       | bic           |  15 |   0.4651 |   0.5744 |     -0.1092 |  -0.1338 |   -0.0865 |    -2.2332 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | full       | joint         |  15 |   0.4651 |   0.5929 |     -0.1277 |  -0.1532 |   -0.0971 |    -2.2421 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | bic           |  15 |   0.7269 |   0.8857 |     -0.1588 |  -0.198  |   -0.1158 |    -1.8746 |    0.0002 |   0.0132 |
| CuCD-ID   | CNN1D  | icarl      | joint_scratch |  15 |   0.8186 |   0.9852 |     -0.1666 |  -0.1921 |   -0.1436 |    -3.348  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | icarl      | joint         |  15 |   0.8186 |   0.9923 |     -0.1737 |  -0.1988 |   -0.151  |    -3.5338 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | icarl         |  15 |   0.6343 |   0.8186 |     -0.1843 |  -0.2379 |   -0.1358 |    -1.735  |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | full          |  15 |   0.4386 |   0.6438 |     -0.2052 |  -0.2687 |   -0.1393 |    -1.5349 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | bic           |  15 |   0.6343 |   0.8857 |     -0.2514 |  -0.3029 |   -0.207  |    -2.5466 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | joint_scratch |  15 |   0.7269 |   0.9852 |     -0.2583 |  -0.2998 |   -0.2143 |    -2.9327 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | icarl         |  15 |   0.6438 |   0.907  |     -0.2632 |  -0.3049 |   -0.2216 |    -3.1093 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | replay     | joint         |  15 |   0.7269 |   0.9923 |     -0.2654 |  -0.3067 |   -0.2212 |    -3.0244 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | bic           |  15 |   0.6438 |   0.9226 |     -0.2788 |  -0.3207 |   -0.2379 |    -3.3162 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | full          |  15 |   0.1882 |   0.515  |     -0.3268 |  -0.3576 |   -0.2953 |    -5.1156 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | joint         |  15 |   0.6438 |   0.9723 |     -0.3284 |  -0.3675 |   -0.2917 |    -4.2447 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | ewc           |  15 |   0.104  |   0.4386 |     -0.3346 |  -0.3809 |   -0.2887 |    -3.5588 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | full          |  15 |   0.13   |   0.4651 |     -0.3351 |  -0.3589 |   -0.3128 |    -7.168  |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | full          |  15 |   0.1733 |   0.515  |     -0.3416 |  -0.3682 |   -0.3156 |    -6.2078 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | full       | joint_scratch |  15 |   0.6438 |   0.9897 |     -0.3459 |  -0.3875 |   -0.3054 |    -4.1367 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | full          |  15 |   0.117  |   0.4651 |     -0.3481 |  -0.3687 |   -0.3278 |    -8.402  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | joint_scratch |  15 |   0.6343 |   0.9852 |     -0.3509 |  -0.4019 |   -0.3065 |    -3.5196 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | full       | joint         |  15 |   0.6343 |   0.9923 |     -0.358  |  -0.4074 |   -0.3141 |    -3.673  |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | replay        |  15 |   0.13   |   0.5076 |     -0.3776 |  -0.4073 |   -0.3486 |    -6.2656 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | joint_scratch |  15 |   0.1882 |   0.5855 |     -0.3973 |  -0.4226 |   -0.3691 |    -7.2412 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | joint         |  15 |   0.1882 |   0.5923 |     -0.4041 |  -0.4406 |   -0.3673 |    -5.4308 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | icarl         |  15 |   0.1882 |   0.5925 |     -0.4044 |  -0.4315 |   -0.3785 |    -7.4819 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | icarl         |  15 |   0.13   |   0.537  |     -0.407  |  -0.4226 |   -0.3916 |   -12.8537 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | joint_scratch |  15 |   0.1733 |   0.5855 |     -0.4121 |  -0.4363 |   -0.3873 |    -8.1019 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | replay        |  15 |   0.168  |   0.5809 |     -0.4129 |  -0.4262 |   -0.3998 |   -15.3961 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | lwf        | icarl         |  15 |   0.1235 |   0.537  |     -0.4134 |  -0.4302 |   -0.3963 |   -11.9142 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | full          |  15 |   0.168  |   0.5822 |     -0.4143 |  -0.4259 |   -0.402  |   -16.9936 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | replay        |  15 |   0.1733 |   0.5908 |     -0.4174 |  -0.4527 |   -0.3861 |    -6.0627 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | joint         |  15 |   0.1733 |   0.5923 |     -0.419  |  -0.4577 |   -0.3789 |    -5.155  |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | icarl         |  15 |   0.1733 |   0.5925 |     -0.4192 |  -0.4475 |   -0.3927 |    -7.4467 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | joint_scratch |  15 |   0.13   |   0.5497 |     -0.4196 |  -0.4412 |   -0.3986 |    -9.5217 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | joint         |  15 |   0.168  |   0.5878 |     -0.4199 |  -0.4414 |   -0.398  |    -9.4626 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | icarl         |  15 |   0.117  |   0.537  |     -0.42   |  -0.4402 |   -0.3968 |    -9.46   |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | icarl         |  15 |   0.168  |   0.5897 |     -0.4217 |  -0.4315 |   -0.4097 |   -18.4876 |    0.0001 |   0.0132 |
| STIN      | MLP    | lwf        | joint         |  15 |   0.1635 |   0.5878 |     -0.4243 |  -0.4441 |   -0.4053 |   -10.6134 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | lwf        | joint_scratch |  15 |   0.1235 |   0.5497 |     -0.4261 |  -0.4482 |   -0.4045 |    -9.4507 |    0.0001 |   0.0132 |
| STIN      | MLP    | lwf        | icarl         |  15 |   0.1635 |   0.5897 |     -0.4262 |  -0.432  |   -0.4199 |   -34.558  |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | joint_scratch |  15 |   0.168  |   0.5975 |     -0.4296 |  -0.4512 |   -0.4068 |    -9.5218 |    0.0001 |   0.0132 |
| STIN      | GRU    | ewc        | bic           |  15 |   0.1882 |   0.6204 |     -0.4323 |  -0.4656 |   -0.3995 |    -6.3295 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | joint_scratch |  15 |   0.117  |   0.5497 |     -0.4327 |  -0.4527 |   -0.4104 |    -9.8687 |    0.0001 |   0.0132 |
| STIN      | MLP    | lwf        | joint_scratch |  15 |   0.1635 |   0.5975 |     -0.434  |  -0.4519 |   -0.4158 |   -11.7    |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | full          |  15 |   0.1474 |   0.5822 |     -0.4349 |  -0.4471 |   -0.4221 |   -16.9609 |    0.0001 |   0.0132 |
| STIN      | GRU    | lwf        | joint_scratch |  15 |   0.1493 |   0.5855 |     -0.4362 |  -0.4607 |   -0.4086 |    -8.3173 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | joint         |  15 |   0.1474 |   0.5878 |     -0.4405 |  -0.4644 |   -0.4192 |    -9.5435 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | icarl         |  15 |   0.1474 |   0.5897 |     -0.4423 |  -0.4555 |   -0.4304 |   -17.3546 |    0.0001 |   0.0132 |
| STIN      | GRU    | lwf        | joint         |  15 |   0.1493 |   0.5923 |     -0.443  |  -0.4758 |   -0.4084 |    -6.384  |    0.0001 |   0.0132 |
| STIN      | GRU    | lwf        | icarl         |  15 |   0.1493 |   0.5925 |     -0.4432 |  -0.4649 |   -0.4191 |    -9.5317 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | bic           |  15 |   0.13   |   0.5744 |     -0.4444 |  -0.474  |   -0.4148 |    -7.2338 |    0.0001 |   0.0132 |
| STIN      | GRU    | finetune   | bic           |  15 |   0.1733 |   0.6204 |     -0.4471 |  -0.483  |   -0.413  |    -6.1054 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | joint_scratch |  15 |   0.1474 |   0.5975 |     -0.4501 |  -0.4727 |   -0.4286 |    -9.9319 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | lwf        | bic           |  15 |   0.1235 |   0.5744 |     -0.4508 |  -0.4801 |   -0.4212 |    -7.3689 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | bic           |  15 |   0.117  |   0.5744 |     -0.4574 |  -0.4867 |   -0.4273 |    -7.5063 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | finetune   | joint         |  15 |   0.13   |   0.5929 |     -0.4629 |  -0.4865 |   -0.4386 |    -9.2871 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | icarl         |  15 |   0.4386 |   0.907  |     -0.4684 |  -0.5172 |   -0.4204 |    -4.6908 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | lwf        | joint         |  15 |   0.1235 |   0.5929 |     -0.4693 |  -0.495  |   -0.4435 |    -8.8814 |    0.0001 |   0.0132 |
| STIN      | GRU    | lwf        | bic           |  15 |   0.1493 |   0.6204 |     -0.4711 |  -0.5028 |   -0.4395 |    -7.2267 |    0.0001 |   0.0132 |
| STIN      | CNN1D  | ewc        | joint         |  15 |   0.117  |   0.5929 |     -0.4759 |  -0.4962 |   -0.4538 |   -10.9415 |    0.0001 |   0.0132 |
| STIN      | MLP    | finetune   | bic           |  15 |   0.168  |   0.6476 |     -0.4797 |  -0.4947 |   -0.4631 |   -14.781  |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | bic           |  15 |   0.4386 |   0.9226 |     -0.484  |  -0.5323 |   -0.4369 |    -4.8943 |    0.0001 |   0.0132 |
| STIN      | MLP    | lwf        | bic           |  15 |   0.1635 |   0.6476 |     -0.4841 |  -0.497  |   -0.4716 |   -18.4993 |    0.0001 |   0.0132 |
| STIN      | MLP    | ewc        | bic           |  15 |   0.1474 |   0.6476 |     -0.5003 |  -0.518  |   -0.4829 |   -13.9879 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | full          |  15 |   0.1331 |   0.6343 |     -0.5012 |  -0.5584 |   -0.4374 |    -4.0635 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | joint         |  15 |   0.4386 |   0.9723 |     -0.5337 |  -0.5762 |   -0.4916 |    -6.0589 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | full          |  15 |   0.104  |   0.6438 |     -0.5398 |  -0.5729 |   -0.5047 |    -7.6901 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | ewc        | joint_scratch |  15 |   0.4386 |   0.9897 |     -0.5511 |  -0.5937 |   -0.5091 |    -6.2887 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | full          |  15 |   0.0725 |   0.6343 |     -0.5617 |  -0.6055 |   -0.5123 |    -5.7799 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | replay        |  15 |   0.1331 |   0.7269 |     -0.5938 |  -0.6488 |   -0.5391 |    -5.252  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | icarl         |  15 |   0.1331 |   0.8186 |     -0.6855 |  -0.7294 |   -0.637  |    -7.3233 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | lwf        | icarl         |  15 |   0.1644 |   0.907  |     -0.7426 |  -0.7666 |   -0.7175 |   -14.7116 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | icarl         |  15 |   0.0725 |   0.8186 |     -0.7461 |  -0.7684 |   -0.7216 |   -15.4405 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | lwf        | icarl         |  15 |   0.0722 |   0.8186 |     -0.7464 |  -0.7687 |   -0.722  |   -15.4586 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | bic           |  15 |   0.1331 |   0.8857 |     -0.7526 |  -0.7962 |   -0.7052 |    -8.0034 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | lwf        | bic           |  15 |   0.1644 |   0.9226 |     -0.7582 |  -0.7788 |   -0.7368 |   -17.4857 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | icarl         |  15 |   0.104  |   0.907  |     -0.803  |  -0.8289 |   -0.7691 |   -13.0037 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | lwf        | joint         |  15 |   0.1644 |   0.9723 |     -0.8079 |  -0.827  |   -0.7874 |   -19.8831 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | bic           |  15 |   0.0725 |   0.8857 |     -0.8131 |  -0.8348 |   -0.7894 |   -17.296  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | lwf        | bic           |  15 |   0.0722 |   0.8857 |     -0.8135 |  -0.835  |   -0.7897 |   -17.3547 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | bic           |  15 |   0.104  |   0.9226 |     -0.8186 |  -0.8431 |   -0.7872 |   -14.0817 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | lwf        | joint_scratch |  15 |   0.1644 |   0.9897 |     -0.8253 |  -0.8459 |   -0.8048 |   -19.7976 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | replay        |  15 |   0.104  |   0.9343 |     -0.8303 |  -0.852  |   -0.8029 |   -16.4507 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | full          |  15 |   0.1067 |   0.9562 |     -0.8495 |  -0.8947 |   -0.7903 |    -7.8762 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | joint_scratch |  15 |   0.1331 |   0.9852 |     -0.8521 |  -0.879  |   -0.8211 |   -14.4353 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | finetune   | joint         |  15 |   0.1331 |   0.9923 |     -0.8592 |  -0.8865 |   -0.828  |   -14.3693 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | full          |  15 |   0.094  |   0.9562 |     -0.8622 |  -0.8957 |   -0.8196 |   -10.8328 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | joint         |  15 |   0.104  |   0.9723 |     -0.8683 |  -0.8883 |   -0.8416 |   -17.919  |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | icarl         |  15 |   0.1067 |   0.9893 |     -0.8826 |  -0.912  |   -0.8383 |   -11.2169 |    0.0001 |   0.0132 |
| CuCD-ID   | GRU    | finetune   | joint_scratch |  15 |   0.104  |   0.9897 |     -0.8857 |  -0.9058 |   -0.8573 |   -17.2623 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | bic           |  15 |   0.1067 |   0.9924 |     -0.8857 |  -0.9153 |   -0.8411 |   -11.2211 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | joint         |  15 |   0.1067 |   0.9963 |     -0.8896 |  -0.9193 |   -0.8449 |   -11.2451 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | ewc        | joint_scratch |  15 |   0.1067 |   0.9969 |     -0.8902 |  -0.9201 |   -0.8453 |   -11.2189 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | icarl         |  15 |   0.094  |   0.9893 |     -0.8953 |  -0.9159 |   -0.8619 |   -15.6622 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | replay        |  15 |   0.094  |   0.9906 |     -0.8966 |  -0.9174 |   -0.8638 |   -15.8983 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | bic           |  15 |   0.094  |   0.9924 |     -0.8984 |  -0.919  |   -0.8647 |   -15.592  |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | lwf        | icarl         |  15 |   0.0879 |   0.9893 |     -0.9014 |  -0.9165 |   -0.8745 |   -18.1707 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | joint         |  15 |   0.094  |   0.9963 |     -0.9023 |  -0.923  |   -0.8685 |   -15.6025 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | finetune   | joint_scratch |  15 |   0.094  |   0.9969 |     -0.9029 |  -0.9234 |   -0.8693 |   -15.6816 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | lwf        | bic           |  15 |   0.0879 |   0.9924 |     -0.9045 |  -0.9195 |   -0.8775 |   -18.1438 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | lwf        | joint         |  15 |   0.0879 |   0.9963 |     -0.9084 |  -0.9235 |   -0.8812 |   -18.1281 |    0.0001 |   0.0132 |
| CuCD-ID   | MLP    | lwf        | joint_scratch |  15 |   0.0879 |   0.9969 |     -0.909  |  -0.9241 |   -0.8818 |   -18.1315 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | joint_scratch |  15 |   0.0725 |   0.9852 |     -0.9127 |  -0.917  |   -0.9061 |   -81.5134 |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | lwf        | joint_scratch |  15 |   0.0722 |   0.9852 |     -0.913  |  -0.9173 |   -0.9064 |   -81.208  |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | ewc        | joint         |  15 |   0.0725 |   0.9923 |     -0.9197 |  -0.9207 |   -0.9188 |  -467.85   |    0.0001 |   0.0132 |
| CuCD-ID   | CNN1D  | lwf        | joint         |  15 |   0.0722 |   0.9923 |     -0.9201 |  -0.921  |   -0.9192 |  -508.955  |    0.0001 |   0.0132 |

### 7.8 iCaRL-lite + NCM evaluation

The original NCM evaluation reruns the frozen iCaRL-lite configuration with embedding-space herding and a post-hoc cosine NCM head fitted only on stored exemplars. Test rows are never used to fit the NCM means. This is the NCM result used in the main comparison, but it changes both the feature/herding path and the classifier, so it cannot by itself isolate the classifier change; Section 7.8.1 supplies the matched control.

**NCM ACC, mean over 15 seeds:**

| dataset   | arch   |   icarl_ncm |
|:----------|:-------|------------:|
| CuCD-ID   | CNN1D  |      0.795  |
| CuCD-ID   | GRU    |      0.8671 |
| CuCD-ID   | MLP    |      0.896  |
| STIN      | CNN1D  |      0.567  |
| STIN      | GRU    |      0.6606 |
| STIN      | MLP    |      0.632  |

**Paired ACC difference vs softmax-head baselines (same seeds; confounded by herding + classifier):**

| dataset   | arch   |     bic |    full |   icarl |   replay |
|:----------|:-------|--------:|--------:|--------:|---------:|
| CuCD-ID   | CNN1D  | -0.0779 |  0.1756 | -0.001  |   0.0861 |
| CuCD-ID   | GRU    | -0.0524 |  0.2396 | -0.0449 |  -0.0553 |
| CuCD-ID   | MLP    | -0.0963 | -0.0571 | -0.0944 |  -0.0934 |
| STIN      | CNN1D  | -0.0616 |  0.0253 | -0.0402 |  -0.0222 |
| STIN      | GRU    | -0.0125 |  0.0717 | -0.0038 |   0.0018 |
| STIN      | MLP    | -0.0546 | -0.0189 | -0.0258 |  -0.0157 |

The NCM head is not a free win: the largest average drop in this original comparison is **-0.0963 ACC** on CuCD-ID/MLP vs `bic`. The matched controls below show how much of this is the classifier versus the embedding/herding change.


#### 7.8.1 NCM sanity/control: buffer, embedding herding, and classifier

The control re-runs iCaRL with the same input features under four paired conditions: the original input-space herding + softmax head, an embedding-space herding + softmax head, and NCM fitted on the buffer or on all training data; a joint all-data model is included as a second control. The NCM metric is task-balanced ACC, matching the existing NCM files; overall sample accuracy is retained as `*_SampleACC` in `ncm_sanity_summary.csv`.

| dataset   | arch   | stage           | input_softmax   |   embed_softmax | ncm_buffer   |   ncm_all | existing_ncm   | ncm_buf-emb   | ncm_all-emb   | buf-all   |
|:----------|:-------|:----------------|:----------------|----------------:|:-------------|----------:|:---------------|:--------------|:--------------|:----------|
| CuCD-ID   | CNN1D  | icarl_embedding | 0.7960          |          0.8109 | 0.7950       |    0.7952 | 0.7950         | -0.0159       | -0.0157       | -0.0002   |
| CuCD-ID   | GRU    | icarl_embedding | 0.9120          |          0.9308 | 0.8671       |    0.8653 | 0.8671         | -0.0637       | -0.0655       | 0.0018    |
| CuCD-ID   | MLP    | icarl_embedding | 0.9903          |          0.9825 | 0.8960       |    0.8948 | 0.8960         | -0.0866       | -0.0878       | 0.0012    |
| STIN      | CNN1D  | icarl_embedding | 0.6072          |          0.6191 | 0.5670       |    0.5667 | 0.5670         | -0.0522       | -0.0524       | 0.0003    |
| STIN      | GRU    | icarl_embedding | 0.6643          |          0.6668 | 0.6606       |    0.6541 | 0.6606         | -0.0062       | -0.0127       | 0.0065    |
| STIN      | MLP    | icarl_embedding | 0.6578          |          0.6622 | 0.6320       |    0.632  | 0.6320         | -0.0302       | -0.0302       | -0.0000   |
| CuCD-ID   | CNN1D  | joint_scratch   |                 |          0.9816 |              |    0.7972 |                |               |               |           |
| CuCD-ID   | GRU    | joint_scratch   |                 |          0.9914 |              |    0.9915 |                |               |               |           |
| CuCD-ID   | MLP    | joint_scratch   |                 |          0.9961 |              |    0.9135 |                |               |               |           |
| STIN      | CNN1D  | joint_scratch   |                 |          0.6125 |              |    0.5572 |                |               |               |           |
| STIN      | GRU    | joint_scratch   |                 |          0.6794 |              |    0.6887 |                |               |               |           |
| STIN      | MLP    | joint_scratch   |                 |          0.6566 |              |    0.6207 |                |               |               |           |

**Buffer is not the limitation.** NCM-all minus NCM-buffer has maximum absolute value **0.0065 ACC** across all six cells, so a larger or all-data class-mean estimate would not close the gap.

**The NCM gap is classifier/feature-space.** NCM-all is below the embedding-herding softmax head in 6/6 iCaRL cells; the largest gaps are on CuCD-ID/MLP and STIN/CNN1D. The joint GRU cell is essentially at parity. This is a negative/mixed NCM result, not evidence of NCM superiority.


**Joint all-data NCM minus softmax (same features and training data, only the head changes):**

| dataset   | arch   |   NCM - softmax |
|:----------|:-------|----------------:|
| CuCD-ID   | CNN1D  |         -0.1844 |
| CuCD-ID   | GRU    |          0.0001 |
| CuCD-ID   | MLP    |         -0.0826 |
| STIN      | CNN1D  |         -0.0553 |
| STIN      | GRU    |          0.0094 |
| STIN      | MLP    |         -0.0359 |

This is the cleanest classifier-only contrast: the joint GRU is at parity, while CuCD-ID MLP/CNN1D and STIN CNN1D/MLP prefer the softmax head. Together with NCM-all approximately NCM-buffer, this rules out a buffer-size explanation and points to the cosine class-mean geometry / feature scaling.

See `figures/ncm_sanity_control.png`. _The old sentence that this comparison 'isolates the classifier change' is incorrect and has been removed: the original NCM run changes the herding path as well as the head. Use Section 7.8.1 for the matched control._


### 7.9 XAI: SHAP vs IG agreement and faithfulness

KernelSHAP (256 sampled coalitions) and exact-input-gradient IG are computed on the same class-stratified test rows for BiC, iCaRL-lite+NCM and offline Scratch. The main text uses **correctly classified rows only** and reports Top5Ratio, deletion AUC and insertion AUC; agreement and completeness are diagnostics, not proof of causal feature use. The primary background is a real training medoid, not the zero vector, because zero is degenerate for cosine/NCM paths; a class-balanced mean of per-class medoids is run as a sensitivity check. IG uses 512 steps.

**Main XAI metrics, correct-only rows (mean over 3 seeds):**

| dataset   | arch   | method        |   n_samples_mean |   top5_ratio_mean |   ig_deletion_auc_mean |   shap_deletion_auc_mean |   ig_insertion_auc_mean |   shap_insertion_auc_mean |
|:----------|:-------|:--------------|-----------------:|------------------:|-----------------------:|-------------------------:|------------------------:|--------------------------:|
| CuCD-ID   | GRU    | bic           |          45      |            0.5463 |                 0.2681 |                   0.2907 |                  0.6892 |                    0.6931 |
| CuCD-ID   | GRU    | icarl_ncm     |          41.6667 |            0.5327 |                 0.3072 |                   0.293  |                  0.747  |                    0.749  |
| CuCD-ID   | GRU    | joint_scratch |          47.6667 |            0.5384 |                 0.2949 |                   0.313  |                  0.8687 |                    0.8721 |
| CuCD-ID   | MLP    | bic           |          47.6667 |            0.7932 |                 0.2783 |                   0.2964 |                  0.8952 |                    0.8904 |
| CuCD-ID   | MLP    | icarl_ncm     |          45      |            0.5964 |                 0.2838 |                   0.3388 |                  0.8443 |                    0.8096 |
| CuCD-ID   | MLP    | joint_scratch |          48      |            0.8083 |                 0.279  |                   0.2854 |                  0.9257 |                    0.9166 |
| STIN      | GRU    | bic           |          32      |            0.5828 |                 0.1354 |                   0.1296 |                  0.6649 |                    0.6318 |
| STIN      | GRU    | icarl_ncm     |          31      |            0.5878 |                 0.1818 |                   0.1882 |                  0.6675 |                    0.6438 |
| STIN      | GRU    | joint_scratch |          33.3333 |            0.5957 |                 0.1592 |                   0.1615 |                  0.718  |                    0.6602 |
| STIN      | MLP    | bic           |          32.3333 |            0.7175 |                 0.1515 |                   0.1614 |                  0.695  |                    0.6817 |
| STIN      | MLP    | icarl_ncm     |          28.3333 |            0.626  |                 0.156  |                   0.1843 |                  0.4955 |                    0.478  |
| STIN      | MLP    | joint_scratch |          31.3333 |            0.6961 |                 0.1397 |                   0.1473 |                  0.7231 |                    0.7055 |

Highest correct-only Top5Ratio: **CuCD-ID/MLP/joint_scratch** (0.808). Lower Top5Ratio for NCM reflects attribution disagreement/stability, not necessarily lower accuracy; deletion/insertion are reported alongside it. Correct-only and all-row tables are in `results/XAI_SUPPLEMENT.md`.


**Correctness split.** Error-only rows are reported in the supplement; cells with fewer than three errors are unstable/NaN and are not used for claims. See `figures/xai_main_metrics.png`.


#### 7.9.1 Completeness, KernelSHAP efficiency, and baseline sensitivity

IG completes the medoid-to-input path to within **0.0044** mean relative error (worst cell) after 512 steps. KernelSHAP's efficiency constraint is enforced through its KKT system, so its completeness residual is exact by construction and is only an implementation sanity check, not evidence of attribution accuracy; agreement and faithfulness metrics are the accuracy evidence. The original zero baseline was degenerate for NCM (cosine of a zero vector), which produced artificially near-zero IG attributions; the medoid baseline fixes this.

**Background sensitivity (secondary class-balanced medoid mean minus primary training medoid):**

| dataset   | arch   | method        |   top5_ratio_mean_primary |   top5_ratio_mean_secondary |   top5_ratio_mean_delta |   ig_deletion_auc_mean_delta |   shap_deletion_auc_mean_delta |
|:----------|:-------|:--------------|--------------------------:|----------------------------:|------------------------:|-----------------------------:|-------------------------------:|
| CuCD-ID   | GRU    | bic           |                    0.5444 |                      0.5319 |                 -0.0125 |                       0.0274 |                         0.0498 |
| CuCD-ID   | GRU    | icarl_ncm     |                    0.5347 |                      0.5167 |                 -0.0181 |                      -0.023  |                         0.0092 |
| CuCD-ID   | GRU    | joint_scratch |                    0.5389 |                      0.5181 |                 -0.0208 |                      -0.0329 |                        -0.0512 |
| CuCD-ID   | MLP    | bic           |                    0.7931 |                      0.6944 |                 -0.0986 |                      -0.0123 |                        -0.0034 |
| CuCD-ID   | MLP    | icarl_ncm     |                    0.6056 |                      0.6097 |                  0.0042 |                       0.0162 |                         0.0112 |
| CuCD-ID   | MLP    | joint_scratch |                    0.8083 |                      0.7569 |                 -0.0514 |                      -0.0039 |                         0.0045 |
| STIN      | GRU    | bic           |                    0.5833 |                      0.6403 |                  0.0569 |                       0.0079 |                         0.0089 |
| STIN      | GRU    | icarl_ncm     |                    0.5931 |                      0.5778 |                 -0.0153 |                       0.0025 |                        -0.0072 |
| STIN      | GRU    | joint_scratch |                    0.5847 |                      0.5625 |                 -0.0222 |                      -0.0055 |                        -0.0125 |
| STIN      | MLP    | bic           |                    0.7097 |                      0.7556 |                  0.0458 |                      -0.0036 |                        -0.0043 |
| STIN      | MLP    | icarl_ncm     |                    0.6167 |                      0.6347 |                  0.0181 |                       0      |                        -0.0036 |
| STIN      | MLP    | joint_scratch |                    0.6986 |                      0.7458 |                  0.0472 |                      -0.0032 |                        -0.0022 |

Across 12 cells, mean |delta Top5Ratio| = 0.0343 (max 0.0986); mean |delta IG deletion AUC| = 0.0115; mean |delta SHAP deletion AUC| = 0.0140. The qualitative ranking is not driven by one background row.


#### 7.9.2 Feature-group attribution and deletion

Groups are the disjoint provenance families in `src/feature_groups.py`. Shares are mean absolute attribution fractions within a run; deletion drop is the drop in target-class probability when the group is replaced by the medoid background. `insertion_gain` is measured from the all-baseline probability (`inserted_prob - baseline_prob`). `insertion_gap_vs_full` is the older `inserted_prob - base_prob` contrast and is negative by construction for partial restorations, so it is supplementary.

| dataset   | group            |   ig_share |   shap_share |   deletion_drop |   insertion_gain |
|:----------|:-----------------|-----------:|-------------:|----------------:|-----------------:|
| CuCD-ID   | time             |     0.1714 |       0.1657 |          0.2773 |           0.1653 |
| CuCD-ID   | ids              |     0.0735 |       0.0512 |          0.0098 |           0.0232 |
| CuCD-ID   | window           |     0.5029 |       0.4958 |          0.39   |           0.332  |
| CuCD-ID   | raw_telemetry    |     0.2521 |       0.2873 |          0.1217 |           0.0731 |
| STIN      | flow_volume      |     0.3548 |       0.2985 |          0.2826 |           0.0698 |
| STIN      | packet_length    |     0.5369 |       0.4431 |          0.3943 |           0.1663 |
| STIN      | iat_timing       |     0.0188 |       0.0288 |          0.0042 |          -0.0007 |
| STIN      | tcp_flags        |     0      |       0.1246 |          0      |           0      |
| STIN      | tcp_window_block |     0.0895 |       0.105  |          0.1168 |           0.0679 |

On CuCD-ID, window aggregates and time contribute most to attribution and deletion; raw telemetry is secondary. Under the primary training-medoid background, the top feature is not universally `TimeRadians`: for CuCD-ID/GRU with BiC or iCaRL+NCM, `UniqueMessageIDsInWindow` ranks first, while MLP cells and GRU/joint_scratch rank `TimeRadians` first. Under the class-balanced sensitivity background the GRU ranking can shift to `TimeRadians` or `MemoryAnonMB`, so the robust reading is time/window aggregates, not one stable feature. STIN is a network-flow family analysis (no time/ID/window provenance groups), so its group shares are not a leakage ablation. See `figures/xai_feature_groups.png`.


### 7.10 Feature-group reliance: retrained ablation

Retrained CuCD-ID/MLP ablation: each row retrains without the named group (or with only that group). `ACC_delta_vs_all` is relative to the all-feature model for the same method; negative means the group was helping. The former `only_raw_telemetry` alias was removed because it was feature-wise identical to `no_time_ids_window`; the table therefore contains six unique variants and 270 runs.

| group                  | variant              | method        |   n_features |   ACC_mean |   ACC_delta_vs_all |   MacroF1_delta_vs_all |
|:-----------------------|:---------------------|:--------------|-------------:|-----------:|-------------------:|-----------------------:|
| time                   | no_time              | bic           |           21 |     0.9889 |            -0.0034 |                -0.0034 |
| ids                    | no_ids               | bic           |           19 |     0.9903 |            -0.002  |                -0.0018 |
| window                 | no_window            | bic           |           13 |     0.9384 |            -0.0539 |                -0.0503 |
| all_leakage_provenance | no_time_ids_window   | bic           |            9 |     0.9153 |            -0.0771 |                -0.0944 |
| raw_telemetry          | only_time_ids_window | bic           |           13 |     0.9843 |            -0.008  |                -0.008  |
| time                   | no_time              | joint_scratch |           21 |     0.9949 |            -0.0017 |                -0.0015 |
| ids                    | no_ids               | joint_scratch |           19 |     0.9956 |            -0.0011 |                -0.0009 |
| window                 | no_window            | joint_scratch |           13 |     0.9912 |            -0.0054 |                -0.0047 |
| all_leakage_provenance | no_time_ids_window   | joint_scratch |            9 |     0.9723 |            -0.0243 |                -0.0211 |
| raw_telemetry          | only_time_ids_window | joint_scratch |           13 |     0.9956 |            -0.001  |                -0.0009 |
| time                   | no_time              | replay        |           21 |     0.9901 |             0.0007 |                 0.0001 |
| ids                    | no_ids               | replay        |           19 |     0.9658 |            -0.0236 |                -0.0213 |
| window                 | no_window            | replay        |           13 |     0.9042 |            -0.0852 |                -0.0686 |
| all_leakage_provenance | no_time_ids_window   | replay        |            9 |     0.9229 |            -0.0666 |                -0.0569 |
| raw_telemetry          | only_time_ids_window | replay        |           13 |     0.9694 |            -0.02   |                -0.0206 |

The large drop for `no_time_ids_window` and the small drop for `only_time_ids_window` (raw telemetry removed, leakage groups retained) show that the leakage-bearing provenance fields carry most of the random-split advantage. The `raw_telemetry` row in this table is the `only_time_ids_window` variant. This is a reliance measure, not proof that every removed field is illegitimate.


### 7.11 Budget-Aware Update Admission Control (BUAC)


Fair random-past temporal scope (held-out seeds 10-14; budget 256 KB memory / 64 KB uplink):

| arch   | gate             |   n |   safe_n |   unsafe_n |   safe_accept |   unsafe_accept |   balanced_acc |   precision |   auroc |
|:-------|:-----------------|----:|---------:|-----------:|--------------:|----------------:|---------------:|------------:|--------:|
| CNN1D  | buac             |  90 |       22 |         68 |         0.636 |           0.132 |          0.752 |       0.609 |   0.867 |
| CNN1D  | buac_iid_trained |  90 |       22 |         68 |         0.818 |           0.176 |          0.821 |       0.6   |   0.888 |
| CNN1D  | rule_buac        |  90 |       22 |         68 |         0.818 |           0.176 |          0.821 |       0.6   |   0.88  |
| GRU    | buac             |  90 |       25 |         65 |         0.56  |           0.169 |          0.695 |       0.56  |   0.834 |
| GRU    | buac_iid_trained |  90 |       25 |         65 |         0.64  |           0.215 |          0.712 |       0.533 |   0.833 |
| GRU    | rule_buac        |  90 |       25 |         65 |         0.56  |           0.154 |          0.703 |       0.583 |   0.818 |
| MLP    | buac             |  90 |       39 |         51 |         0.949 |           0.078 |          0.935 |       0.902 |   0.977 |
| MLP    | buac_iid_trained |  90 |       39 |         51 |         1     |           0.078 |          0.961 |       0.907 |   0.968 |
| MLP    | rule_buac        |  90 |       39 |         51 |         0.949 |           0.078 |          0.935 |       0.902 |   0.982 |

All protocols pooled (IID + cold-start + fair):

| arch   | gate             |   n |   safe_accept |   unsafe_accept |   balanced_acc |   auroc |
|:-------|:-----------------|----:|--------------:|----------------:|---------------:|--------:|
| CNN1D  | buac             | 270 |         0.406 |           0.044 |          0.681 |   0.934 |
| CNN1D  | buac_iid_trained | 270 |         0.812 |           0.199 |          0.807 |   0.817 |
| CNN1D  | rule_buac        | 270 |         0.766 |           0.058 |          0.854 |   0.917 |
| GRU    | buac             | 270 |         0.522 |           0.055 |          0.734 |   0.927 |
| GRU    | buac_iid_trained | 270 |         0.609 |           0.109 |          0.75  |   0.92  |
| GRU    | rule_buac        | 270 |         0.522 |           0.05  |          0.736 |   0.925 |
| MLP    | buac             | 270 |         0.882 |           0.023 |          0.93  |   0.965 |
| MLP    | buac_iid_trained | 270 |         0.978 |           0.339 |          0.82  |   0.758 |
| MLP    | rule_buac        | 270 |         0.882 |           0.023 |          0.93  |   0.968 |

MLP budget sensitivity at 64 KB uplink (retrospective threshold; not a deployment guarantee):

|   mem_budget_KB |   gate_safe_accept |   gate_unsafe_accept |   gate_balanced_acc |
|----------------:|-------------------:|---------------------:|--------------------:|
|              64 |              0     |                0     |               0.5   |
|             128 |              0.108 |                0     |               0.554 |
|             256 |              0.882 |                0.023 |               0.93  |
|             512 |              0.882 |                0.023 |               0.93  |
**Interpretation.** Under the cold-start temporal protocol the gate accepts nothing, and the fixed rule also accepts nothing (safe acceptance 0.000-0.000 across architectures). That regime is behaviorally equivalent to always-reject, so it cannot separate a validity-aware policy from a trivial one. Under the fair random-past protocol the learned gate and the fixed rule do accept updates, but transfer is architecture-dependent (BUAC unsafe acceptance 0.078-0.169); the fixed rule's unsafe acceptance is 0.078-0.176, and an IID-only-trained gate is 0.078-0.215. Only the MLP gate stays near the 5% unsafe target; GRU/CNN exceed it, and the fixed rule lacks the threshold calibration that the learned gate uses. These are stage-wise, offline admission decisions: rejecting an update does not branch later training, and ground-assisted coverage/utility is not evaluated.

The gate uses only pre-test signals: selection-half gain/retention, forward-time flag, current-feature drift, and the memory/uplink budget. `r_fwd_flag` is a protocol proxy, not an independent physical-risk sensor. See `buac_*_decisions.csv` for per-stage decisions and `buac_*_budget_sensitivity.csv` for the budget sweep. See `figures/buac_gate.png` for the fair-protocol safe/unsafe acceptance bars.


### 7.12 External stress test: SCAD scenario-disjoint telemetry

SCAD is an external, CC BY 4.0 satellite-telemetry dataset with scenario-level reaction-wheel attack labels; it is not a second CuCD-ID confirmation. We build a three-task class-incremental sequence (normal+RWA, RWB, RWC) over a single trajectory-group split: 84 training, 12 validation and 24 test groups. The 80 retained features exclude timestamps, scenario identifiers and absolute orbital state, and are centred on each scenario's first 300 attack-free rows. Task 0 uses 700 samples per class and later tasks 150, matching the CuCD-ID budget. Means are over 15 seeds.

| method        |   ACC |   ACC sd |    BWT |    FM |   Macro-F1 |   Balanced ACC |
|:--------------|------:|---------:|-------:|------:|-----------:|---------------:|
| joint_scratch | 0.563 |    0.04  | -0.058 | 0.058 |      0.608 |          0.622 |
| joint         | 0.545 |    0.035 | -0.024 | 0.024 |      0.588 |          0.603 |
| icarl         | 0.506 |    0.032 | -0.422 | 0.422 |      0.49  |          0.491 |
| bic           | 0.498 |    0.042 | -0.198 | 0.198 |      0.524 |          0.535 |
| replay        | 0.462 |    0.024 | -0.396 | 0.396 |      0.472 |          0.506 |
| full          | 0.434 |    0.035 | -0.503 | 0.503 |      0.401 |          0.416 |
| ewc           | 0.334 |    0.002 | -0.913 | 0.913 |      0.106 |          0.252 |
| lwf           | 0.333 |    0     | -0.925 | 0.925 |      0.1   |          0.25  |
| finetune      | 0.333 |    0     | -0.918 | 0.918 |      0.1   |          0.25  |

With 150 shots per new class, the joint-scratch ceiling is **0.563**, and replay reaches **0.462**. BiC-lite (0.498) and iCaRL-lite (0.506) lead the deployable methods, while fine-tuning, EWC and LwF sit at chance. The ranking is budget-dependent: with 700 shots per class, replay reaches **0.585** and joint-scratch **0.662**, but BiC-lite (0.539) and iCaRL-lite (0.563) fall behind replay.


SCAD MLP, 700 shots per class:

| method        |   ACC |    BWT |    FM |
|:--------------|------:|-------:|------:|
| joint_scratch | 0.662 | -0.127 | 0.127 |
| replay        | 0.585 | -0.334 | 0.334 |
| icarl         | 0.563 | -0.433 | 0.433 |
| bic           | 0.539 | -0.157 | 0.157 |

SCAD GRU/CNN cross-architecture check (150 shots):

| arch   | method        |   ACC |    BWT |     FM |
|:-------|:--------------|------:|-------:|-------:|
| CNN1D  | replay        | 0.338 | -0.657 |  0.657 |
| CNN1D  | icarl         | 0.334 | -0.824 |  0.824 |
| CNN1D  | bic           | 0.291 | -0.201 |  0.201 |
| CNN1D  | joint_scratch | 0.244 |  0.021 | -0.006 |
| GRU    | icarl         | 0.37  | -0.495 |  0.495 |
| GRU    | joint_scratch | 0.355 | -0.022 |  0.024 |
| GRU    | replay        | 0.348 | -0.358 |  0.358 |
| GRU    | bic           | 0.344 | -0.179 |  0.179 |

The GRU and CNN remain at or near the four-class chance level (0.25). This is a negative transfer result, not evidence that the CuCD-ID ranking generalises. The external test is therefore reported as a boundary condition: it strengthens the paper's validity argument only because it is allowed to fail, and it shows that scenario-disjoint SCAD telemetry is harder than the random CuCD-ID split.


## 8. Claim boundaries and known limitations

This report supports a benchmark/diagnostic reading, not a deployment or method-superiority claim. The main boundaries:

- **No scenario-level validation.** CuCD-ID has no scenario/run identifier in the released CSV. A true scenario split is not possible; the time-block proxy is a sensitivity estimand only.

- **SCAD is a weak external check.** Its scenario-disjoint sequence is not directly comparable to CuCD-ID, the joint ceiling is only 0.563 at 150 shots, the method ranking changes at 700 shots, and GRU/CNN are near chance. It is reported as a negative/boundary result, not as independent confirmation.

- **IID is not deployment.** Sliding-window aggregates and time/identifier features can cross the random train/test boundary. Cold-start temporal accuracy collapses; fair random-past temporal recovers much of the IID number but not all, and the no-leaky fair arm is lower still.

- **Leakage and drift are entangled.** The read-only audit in Section 7.4.1 shows dense temporal adjacency in the random split, but also class-incomplete time blocks and large within-class feature drift. The audit is a validity boundary, not a leakage-free estimate.

- **BUAC is stage-wise and architecture-dependent.** The gate does not simulate the downstream effect of rejection or the ground-assisted path; fair-protocol unsafe acceptance is near target only on MLP, while GRU/CNN exceed it. `r_fwd_flag` is a protocol proxy, not a physical risk sensor.

- **`full` is not the winner.** The combined EWC+replay+BiC configuration is below replay on CuCD-ID for all three architectures. Its tuning grid is smaller, and the lambda=0 control shows the selected EWC strength can be harmful.

- **Feature order is exploratory.** The permutation study uses one data split and two training seeds; it does not establish permutation invariance, including for the MLP.

- **Task-order modes are not independent.** On primary orders, `fixed_position` and `moves_with_task` are identical by construction because every primary order starts with T0.

- **Tuning transfer.** Hyperparameters are tuned on CuCD-ID only; STIN is cross-dataset transfer. Tuning seeds 0-2 overlap the confirmatory seeds, with validation-only selection (no test leakage, but not fully independent).

- **Inference is finite-data.** The 216-comparison family is pre-specified and Holm-corrected, but resamples the same finite dataset; it is not evidence about new operating conditions.
