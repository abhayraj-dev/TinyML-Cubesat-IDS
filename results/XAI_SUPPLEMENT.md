# XAI Supplement

_Generated automatically by `experiments/generate_summary_report.py` from the XAI CSVs. The main report uses correctly classified rows only; this supplement retains all-row and error-row detail. Sections 1, 3, 4, and 5 use the primary training-medoid background; Section 2 reports the second-background sensitivity._


## 1. Agreement and faithfulness by correctness subset


### All rows

| dataset   | arch   | method        | subset   |   n_runs |   n_samples_mean |   top5_ratio_mean |   avg_relative_mean |   spearman_mean |   ig_deletion_auc_mean |   shap_deletion_auc_mean |   ig_insertion_auc_mean |   shap_insertion_auc_mean |
|:----------|:-------|:--------------|:---------|---------:|-----------------:|------------------:|--------------------:|----------------:|-----------------------:|-------------------------:|------------------------:|--------------------------:|
| CuCD-ID   | GRU    | bic           | all      |        3 |               48 |            0.5444 |              0.0474 |          0.3748 |                 0.2634 |                   0.2817 |                  0.6805 |                    0.682  |
| CuCD-ID   | GRU    | icarl_ncm     | all      |        3 |               48 |            0.5347 |              0.0463 |          0.4007 |                 0.3124 |                   0.2977 |                  0.7182 |                    0.7235 |
| CuCD-ID   | GRU    | joint_scratch | all      |        3 |               48 |            0.5389 |              0.0408 |          0.4569 |                 0.2936 |                   0.3119 |                  0.8695 |                    0.8727 |
| CuCD-ID   | MLP    | bic           | all      |        3 |               48 |            0.7931 |              0.0181 |          0.7581 |                 0.2768 |                   0.2948 |                  0.8951 |                    0.8903 |
| CuCD-ID   | MLP    | icarl_ncm     | all      |        3 |               48 |            0.6056 |              0.0384 |          0.4579 |                 0.2697 |                   0.3219 |                  0.8465 |                    0.8138 |
| CuCD-ID   | MLP    | joint_scratch | all      |        3 |               48 |            0.8083 |              0.0164 |          0.8185 |                 0.279  |                   0.2854 |                  0.9257 |                    0.9166 |
| STIN      | GRU    | bic           | all      |        3 |               48 |            0.5833 |              0.0287 |          0.2818 |                 0.1163 |                   0.1122 |                  0.6502 |                    0.6132 |
| STIN      | GRU    | icarl_ncm     | all      |        3 |               48 |            0.5931 |              0.0276 |          0.2792 |                 0.1561 |                   0.1606 |                  0.6219 |                    0.6001 |
| STIN      | GRU    | joint_scratch | all      |        3 |               48 |            0.5847 |              0.0319 |          0.2841 |                 0.1356 |                   0.1375 |                  0.6918 |                    0.6207 |
| STIN      | MLP    | bic           | all      |        3 |               48 |            0.7097 |              0.0164 |          0.4506 |                 0.1248 |                   0.134  |                  0.6566 |                    0.6428 |
| STIN      | MLP    | icarl_ncm     | all      |        3 |               48 |            0.6167 |              0.0272 |          0.3599 |                 0.144  |                   0.1695 |                  0.4678 |                    0.453  |
| STIN      | MLP    | joint_scratch | all      |        3 |               48 |            0.6986 |              0.0195 |          0.4743 |                 0.1116 |                   0.1171 |                  0.6857 |                    0.6683 |

### Correct rows

| dataset   | arch   | method        | subset   |   n_runs |   n_samples_mean |   top5_ratio_mean |   avg_relative_mean |   spearman_mean |   ig_deletion_auc_mean |   shap_deletion_auc_mean |   ig_insertion_auc_mean |   shap_insertion_auc_mean |
|:----------|:-------|:--------------|:---------|---------:|-----------------:|------------------:|--------------------:|----------------:|-----------------------:|-------------------------:|------------------------:|--------------------------:|
| CuCD-ID   | GRU    | bic           | correct  |        3 |          45      |            0.5463 |              0.0477 |          0.3752 |                 0.2681 |                   0.2907 |                  0.6892 |                    0.6931 |
| CuCD-ID   | GRU    | icarl_ncm     | correct  |        3 |          41.6667 |            0.5327 |              0.0469 |          0.3864 |                 0.3072 |                   0.293  |                  0.747  |                    0.749  |
| CuCD-ID   | GRU    | joint_scratch | correct  |        3 |          47.6667 |            0.5384 |              0.0408 |          0.4561 |                 0.2949 |                   0.313  |                  0.8687 |                    0.8721 |
| CuCD-ID   | MLP    | bic           | correct  |        3 |          47.6667 |            0.7932 |              0.0181 |          0.7576 |                 0.2783 |                   0.2964 |                  0.8952 |                    0.8904 |
| CuCD-ID   | MLP    | icarl_ncm     | correct  |        3 |          45      |            0.5964 |              0.039  |          0.4637 |                 0.2838 |                   0.3388 |                  0.8443 |                    0.8096 |
| CuCD-ID   | MLP    | joint_scratch | correct  |        3 |          48      |            0.8083 |              0.0164 |          0.8185 |                 0.279  |                   0.2854 |                  0.9257 |                    0.9166 |
| STIN      | GRU    | bic           | correct  |        3 |          32      |            0.5828 |              0.0286 |          0.2836 |                 0.1354 |                   0.1296 |                  0.6649 |                    0.6318 |
| STIN      | GRU    | icarl_ncm     | correct  |        3 |          31      |            0.5878 |              0.0278 |          0.2807 |                 0.1818 |                   0.1882 |                  0.6675 |                    0.6438 |
| STIN      | GRU    | joint_scratch | correct  |        3 |          33.3333 |            0.5957 |              0.0314 |          0.2816 |                 0.1592 |                   0.1615 |                  0.718  |                    0.6602 |
| STIN      | MLP    | bic           | correct  |        3 |          32.3333 |            0.7175 |              0.0166 |          0.451  |                 0.1515 |                   0.1614 |                  0.695  |                    0.6817 |
| STIN      | MLP    | icarl_ncm     | correct  |        3 |          28.3333 |            0.626  |              0.0264 |          0.3834 |                 0.156  |                   0.1843 |                  0.4955 |                    0.478  |
| STIN      | MLP    | joint_scratch | correct  |        3 |          31.3333 |            0.6961 |              0.0205 |          0.4753 |                 0.1397 |                   0.1473 |                  0.7231 |                    0.7055 |

### Error rows

| dataset   | arch   | method        | subset   |   n_runs |   n_samples_mean | top5_ratio_mean   | avg_relative_mean   | spearman_mean   | ig_deletion_auc_mean   | shap_deletion_auc_mean   | ig_insertion_auc_mean   | shap_insertion_auc_mean   |
|:----------|:-------|:--------------|:---------|---------:|-----------------:|:------------------|:--------------------|:----------------|:-----------------------|:-------------------------|:------------------------|:--------------------------|
| CuCD-ID   | GRU    | bic           | error    |        3 |           3      | 0.4800            | 0.0405              | 0.4091          | 0.0916                 | 0.0799                   | 0.5133                  | 0.4705                    |
| CuCD-ID   | GRU    | icarl_ncm     | error    |        3 |           6.3333 | 0.5492            | 0.0431              | 0.4920          | 0.3482                 | 0.3300                   | 0.5325                  | 0.5555                    |
| CuCD-ID   | GRU    | joint_scratch | error    |        3 |           0.3333 |                   |                     |                 |                        |                          |                         |                           |
| CuCD-ID   | MLP    | bic           | error    |        3 |           0.3333 |                   |                     |                 |                        |                          |                         |                           |
| CuCD-ID   | MLP    | icarl_ncm     | error    |        3 |           3      | 0.7750            | 0.0305              | 0.3961          | 0.0602                 | 0.0738                   | 0.8886                  | 0.8879                    |
| CuCD-ID   | MLP    | joint_scratch | error    |        3 |           0      |                   |                     |                 |                        |                          |                         |                           |
| STIN      | GRU    | bic           | error    |        3 |          16      | 0.5822            | 0.0287              | 0.2778          | 0.0751                 | 0.0743                   | 0.6223                  | 0.5781                    |
| STIN      | GRU    | icarl_ncm     | error    |        3 |          17      | 0.6025            | 0.0272              | 0.2765          | 0.1108                 | 0.1112                   | 0.5414                  | 0.5234                    |
| STIN      | GRU    | joint_scratch | error    |        3 |          14.6667 | 0.5604            | 0.0330              | 0.2874          | 0.0831                 | 0.0841                   | 0.6303                  | 0.5313                    |
| STIN      | MLP    | bic           | error    |        3 |          15.6667 | 0.6950            | 0.0159              | 0.4501          | 0.0682                 | 0.0754                   | 0.5772                  | 0.5621                    |
| STIN      | MLP    | icarl_ncm     | error    |        3 |          19.6667 | 0.6042            | 0.0282              | 0.3304          | 0.1256                 | 0.1482                   | 0.4310                  | 0.4194                    |
| STIN      | MLP    | joint_scratch | error    |        3 |          16.6667 | 0.6992            | 0.0178              | 0.4721          | 0.0626                 | 0.0643                   | 0.6177                  | 0.5991                    |

Error-only cells with fewer than three errors are unstable (NaN standard deviations) and are not used for claims.


## 2. Background sensitivity

Primary = training medoid (one real training row nearest the global mean). Secondary = class-balanced mean of per-class training medoids. Values are secondary minus primary; small deltas indicate that the qualitative ranking is not an artefact of one background row.

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

Across 12 cells, mean |delta Top5Ratio| = 0.0343 (max 0.0986); mean |delta IG deletion AUC| = 0.0115; mean |delta SHAP deletion AUC| = 0.0140.


## 3. Completeness diagnostics

IG uses 512 steps and a training-medoid background. KernelSHAP uses 256 sampled coalitions; its efficiency constraint is enforced through the KKT system, so completeness is exact by construction. That is an implementation sanity check, not evidence of attribution accuracy.

| dataset   | arch   | method        |   ig_completeness_mean_rel_mean |   ig_completeness_max_abs_mean |   ig_completeness_frac_1pct_mean |   shap_completeness_mean_rel_mean |   shap_completeness_max_abs_mean |   shap_completeness_frac_1pct_mean |
|:----------|:-------|:--------------|--------------------------------:|-------------------------------:|---------------------------------:|----------------------------------:|---------------------------------:|-----------------------------------:|
| CuCD-ID   | GRU    | bic           |                          0.0013 |                         0.0142 |                           0.9722 |                                 0 |                                0 |                                  1 |
| CuCD-ID   | GRU    | icarl_ncm     |                          0      |                         0.0011 |                           1      |                                 0 |                                0 |                                  1 |
| CuCD-ID   | GRU    | joint_scratch |                          0      |                         0.0006 |                           1      |                                 0 |                                0 |                                  1 |
| CuCD-ID   | MLP    | bic           |                          0.0005 |                         0.0257 |                           1      |                                 0 |                                0 |                                  1 |
| CuCD-ID   | MLP    | icarl_ncm     |                          0.0015 |                         0.0313 |                           0.9931 |                                 0 |                                0 |                                  1 |
| CuCD-ID   | MLP    | joint_scratch |                          0.002  |                         0.0258 |                           0.9931 |                                 0 |                                0 |                                  1 |
| STIN      | GRU    | bic           |                          0.0044 |                         0.6003 |                           0.9375 |                                 0 |                                0 |                                  1 |
| STIN      | GRU    | icarl_ncm     |                          0.0001 |                         0.0322 |                           1      |                                 0 |                                0 |                                  1 |
| STIN      | GRU    | joint_scratch |                          0.0002 |                         0.0002 |                           0.9931 |                                 0 |                                0 |                                  1 |
| STIN      | MLP    | bic           |                          0.0007 |                         0.0338 |                           1      |                                 0 |                                0 |                                  1 |
| STIN      | MLP    | icarl_ncm     |                          0.0009 |                         0.0406 |                           0.9931 |                                 0 |                                0 |                                  1 |
| STIN      | MLP    | joint_scratch |                          0.0006 |                         0.0331 |                           1      |                                 0 |                                0 |                                  1 |

## 4. Feature-group attribution and deletion

Shares are mean absolute attribution fractions; deletion drop is target-probability drop under group replacement by the medoid. `insertion_gain` is measured against the all-baseline probability (`inserted_prob - baseline_prob`). `insertion_gap_vs_full` is the older `inserted_prob - base_prob` contrast; it is negative by construction whenever full input is more confident than a partial group restoration, so it is supplementary.

| dataset   | arch   | method        | group            |   n_features |   ig_share_mean |   shap_share_mean |   deletion_drop_mean |   baseline_prob_mean |   insertion_gain_mean |   insertion_gap_vs_full_mean |
|:----------|:-------|:--------------|:-----------------|-------------:|----------------:|------------------:|---------------------:|---------------------:|----------------------:|-----------------------------:|
| CuCD-ID   | GRU    | bic           | ids              |            3 |          0.08   |            0.0588 |               0.0323 |               0.1874 |                0.0003 |                      -0.685  |
| CuCD-ID   | GRU    | bic           | raw_telemetry    |            9 |          0.2827 |            0.3704 |               0.2175 |               0.1874 |                0.1186 |                      -0.5667 |
| CuCD-ID   | GRU    | bic           | time             |            1 |          0.1172 |            0.0867 |               0.1807 |               0.1874 |                0.058  |                      -0.6273 |
| CuCD-ID   | GRU    | bic           | window           |            9 |          0.5201 |            0.4842 |               0.4319 |               0.1874 |                0.2562 |                      -0.4291 |
| CuCD-ID   | GRU    | icarl_ncm     | ids              |            3 |          0.0712 |            0.0537 |               0.0202 |               0.2224 |                0.0042 |                      -0.6982 |
| CuCD-ID   | GRU    | icarl_ncm     | raw_telemetry    |            9 |          0.3007 |            0.3342 |               0.2986 |               0.2224 |                0.1093 |                      -0.5931 |
| CuCD-ID   | GRU    | icarl_ncm     | time             |            1 |          0.0784 |            0.0537 |               0.1269 |               0.2224 |               -0.0012 |                      -0.7036 |
| CuCD-ID   | GRU    | icarl_ncm     | window           |            9 |          0.5497 |            0.5584 |               0.5294 |               0.2224 |                0.3105 |                      -0.3919 |
| CuCD-ID   | GRU    | joint_scratch | ids              |            3 |          0.0971 |            0.0415 |               0.0014 |               0.1944 |                0.0301 |                      -0.7678 |
| CuCD-ID   | GRU    | joint_scratch | raw_telemetry    |            9 |          0.2264 |            0.247  |               0.1099 |               0.1944 |                0.0678 |                      -0.73   |
| CuCD-ID   | GRU    | joint_scratch | time             |            1 |          0.2268 |            0.2121 |               0.4151 |               0.1944 |                0.2788 |                      -0.519  |
| CuCD-ID   | GRU    | joint_scratch | window           |            9 |          0.4496 |            0.4994 |               0.3412 |               0.1944 |                0.2699 |                      -0.5279 |
| CuCD-ID   | MLP    | bic           | ids              |            3 |          0.0595 |            0.0509 |               0.0012 |               0.1945 |                0.054  |                      -0.729  |
| CuCD-ID   | MLP    | bic           | raw_telemetry    |            9 |          0.2238 |            0.2428 |               0.0352 |               0.1945 |                0.0502 |                      -0.7327 |
| CuCD-ID   | MLP    | bic           | time             |            1 |          0.2212 |            0.2323 |               0.3033 |               0.1945 |                0.2181 |                      -0.5649 |
| CuCD-ID   | MLP    | bic           | window           |            9 |          0.4955 |            0.474  |               0.3854 |               0.1945 |                0.3879 |                      -0.3951 |
| CuCD-ID   | MLP    | icarl_ncm     | ids              |            3 |          0.0707 |            0.0479 |               0.0029 |               0.1947 |                0.0256 |                      -0.7095 |
| CuCD-ID   | MLP    | icarl_ncm     | raw_telemetry    |            9 |          0.2246 |            0.2493 |               0.0416 |               0.1947 |                0.0384 |                      -0.6967 |
| CuCD-ID   | MLP    | icarl_ncm     | time             |            1 |          0.2134 |            0.2368 |               0.2939 |               0.1947 |                0.2528 |                      -0.4823 |
| CuCD-ID   | MLP    | icarl_ncm     | window           |            9 |          0.4912 |            0.466  |               0.3146 |               0.1947 |                0.3555 |                      -0.3795 |
| CuCD-ID   | MLP    | joint_scratch | ids              |            3 |          0.0626 |            0.0543 |               0.0009 |               0.2014 |                0.0248 |                      -0.7683 |
| CuCD-ID   | MLP    | joint_scratch | raw_telemetry    |            9 |          0.2543 |            0.2799 |               0.0275 |               0.2014 |                0.0541 |                      -0.7391 |
| CuCD-ID   | MLP    | joint_scratch | time             |            1 |          0.1715 |            0.1727 |               0.3439 |               0.2014 |                0.1855 |                      -0.6076 |
| CuCD-ID   | MLP    | joint_scratch | window           |            9 |          0.5115 |            0.4931 |               0.3378 |               0.2014 |                0.4121 |                      -0.3811 |
| STIN      | GRU    | bic           | flow_volume      |            6 |          0.4124 |            0.3589 |               0.4061 |               0.0817 |                0.0733 |                      -0.5377 |
| STIN      | GRU    | bic           | iat_timing       |            3 |          0.0238 |            0.0339 |               0.0014 |               0.0817 |               -0.0021 |                      -0.6131 |
| STIN      | GRU    | bic           | packet_length    |            4 |          0.4421 |            0.3384 |               0.4398 |               0.0817 |                0.1155 |                      -0.4955 |
| STIN      | GRU    | bic           | tcp_flags        |            8 |          0      |            0.1362 |               0      |               0.0817 |                0      |                      -0.6111 |
| STIN      | GRU    | bic           | tcp_window_block |            9 |          0.1217 |            0.1327 |               0.1297 |               0.0817 |                0.0376 |                      -0.5735 |
| STIN      | GRU    | icarl_ncm     | flow_volume      |            6 |          0.4478 |            0.3343 |               0.2855 |               0.1285 |                0.1285 |                      -0.3993 |
| STIN      | GRU    | icarl_ncm     | iat_timing       |            3 |          0.0151 |            0.0234 |               0.0016 |               0.1285 |               -0.0053 |                      -0.5331 |
| STIN      | GRU    | icarl_ncm     | packet_length    |            4 |          0.4437 |            0.4024 |               0.246  |               0.1285 |                0.0982 |                      -0.4296 |
| STIN      | GRU    | icarl_ncm     | tcp_flags        |            8 |          0      |            0.1219 |               0      |               0.1285 |                0      |                      -0.5278 |
| STIN      | GRU    | icarl_ncm     | tcp_window_block |            9 |          0.0935 |            0.1181 |               0.1517 |               0.1285 |                0.1205 |                      -0.4073 |
| STIN      | GRU    | joint_scratch | flow_volume      |            6 |          0.3438 |            0.3452 |               0.3653 |               0.1037 |                0.0345 |                      -0.5972 |
| STIN      | GRU    | joint_scratch | iat_timing       |            3 |          0.0233 |            0.0263 |               0.0058 |               0.1037 |                0.0017 |                      -0.63   |
| STIN      | GRU    | joint_scratch | packet_length    |            4 |          0.5284 |            0.3784 |               0.5103 |               0.1037 |                0.1482 |                      -0.4835 |
| STIN      | GRU    | joint_scratch | tcp_flags        |            8 |          0      |            0.1258 |               0      |               0.1037 |                0      |                      -0.6317 |
| STIN      | GRU    | joint_scratch | tcp_window_block |            9 |          0.1045 |            0.1243 |               0.1279 |               0.1037 |                0.0845 |                      -0.5472 |
| STIN      | MLP    | bic           | flow_volume      |            6 |          0.3198 |            0.2587 |               0.2589 |               0.0946 |                0.0443 |                      -0.5616 |
| STIN      | MLP    | bic           | iat_timing       |            3 |          0.019  |            0.0282 |               0.011  |               0.0946 |                0.0009 |                      -0.605  |
| STIN      | MLP    | bic           | packet_length    |            4 |          0.5764 |            0.5252 |               0.4641 |               0.0946 |                0.2357 |                      -0.3702 |
| STIN      | MLP    | bic           | tcp_flags        |            8 |          0      |            0.094  |               0      |               0.0946 |                0      |                      -0.6059 |
| STIN      | MLP    | bic           | tcp_window_block |            9 |          0.0848 |            0.0939 |               0.1004 |               0.0946 |                0.0754 |                      -0.5305 |
| STIN      | MLP    | icarl_ncm     | flow_volume      |            6 |          0.2892 |            0.2059 |               0.0473 |               0.1222 |                0.0727 |                      -0.2919 |
| STIN      | MLP    | icarl_ncm     | iat_timing       |            3 |          0.0127 |            0.0295 |               0.0002 |               0.1222 |                0.0004 |                      -0.3643 |
| STIN      | MLP    | icarl_ncm     | packet_length    |            4 |          0.6444 |            0.5318 |               0.1865 |               0.1222 |                0.2037 |                      -0.1609 |
| STIN      | MLP    | icarl_ncm     | tcp_flags        |            8 |          0      |            0.1621 |               0      |               0.1222 |                0      |                      -0.3647 |
| STIN      | MLP    | icarl_ncm     | tcp_window_block |            9 |          0.0537 |            0.0707 |               0.0945 |               0.1222 |                0.0444 |                      -0.3202 |
| STIN      | MLP    | joint_scratch | flow_volume      |            6 |          0.3156 |            0.2881 |               0.3328 |               0.0775 |                0.0655 |                      -0.5843 |
| STIN      | MLP    | joint_scratch | iat_timing       |            3 |          0.019  |            0.0315 |               0.0053 |               0.0775 |                0.0004 |                      -0.6493 |
| STIN      | MLP    | joint_scratch | packet_length    |            4 |          0.5864 |            0.4826 |               0.5189 |               0.0775 |                0.1961 |                      -0.4536 |
| STIN      | MLP    | joint_scratch | tcp_flags        |            8 |          0      |            0.1076 |               0      |               0.0775 |                0      |                      -0.6497 |
| STIN      | MLP    | joint_scratch | tcp_window_block |            9 |          0.079  |            0.0901 |               0.0969 |               0.0775 |                0.0449 |                      -0.6048 |

## 5. Top features by mean absolute attribution

Averaged over architectures for compactness; the full per-architecture tables are in `results/xai_feature_summary.csv` and `results/xai_feature_importance.csv`. Under the primary training-medoid background, the top feature is not universally `TimeRadians`: for CuCD-ID/GRU with BiC or iCaRL+NCM, `UniqueMessageIDsInWindow` ranks first, while MLP cells and GRU/joint_scratch rank `TimeRadians` first. Under the class-balanced sensitivity background the GRU ranking can shift to `TimeRadians` or `MemoryAnonMB`, so read the pattern as time/window aggregates rather than a stable single feature.


**CuCD-ID / bic / ig**

| feature                      |   mean_abs_attribution |
|:-----------------------------|-----------------------:|
| TimeRadians                  |                 0.1692 |
| UniqueMessageIDsInWindow     |                 0.1464 |
| SlidingWindowMeanIntervalSec |                 0.0847 |
| SlidingWindowMinIntervalSec  |                 0.0809 |
| MemoryPercpuMB               |                 0.0657 |

**CuCD-ID / bic / shap**

| feature                      |   mean_abs_attribution |
|:-----------------------------|-----------------------:|
| TimeRadians                  |                 0.1595 |
| UniqueMessageIDsInWindow     |                 0.1181 |
| SlidingWindowMeanIntervalSec |                 0.0811 |
| SlidingWindowMinIntervalSec  |                 0.0795 |
| MemoryPercpuMB               |                 0.0634 |

**CuCD-ID / icarl_ncm / ig**

| feature                      |   mean_abs_attribution |
|:-----------------------------|-----------------------:|
| UniqueMessageIDsInWindow     |                 0.1678 |
| TimeRadians                  |                 0.1459 |
| SlidingWindowMinIntervalSec  |                 0.0889 |
| MemoryPercpuMB               |                 0.0723 |
| SlidingWindowMeanIntervalSec |                 0.0645 |

**CuCD-ID / icarl_ncm / shap**

| feature                      |   mean_abs_attribution |
|:-----------------------------|-----------------------:|
| UniqueMessageIDsInWindow     |                 0.1454 |
| TimeRadians                  |                 0.1453 |
| SlidingWindowMinIntervalSec  |                 0.074  |
| MessageRateInWindow          |                 0.0665 |
| SlidingWindowMeanIntervalSec |                 0.0603 |

**CuCD-ID / joint_scratch / ig**

| feature                      |   mean_abs_attribution |
|:-----------------------------|-----------------------:|
| TimeRadians                  |                 0.1992 |
| SlidingWindowMeanIntervalSec |                 0.1061 |
| UniqueMessageIDsInWindow     |                 0.095  |
| SlidingWindowMinIntervalSec  |                 0.0724 |
| MessageCountInWindow         |                 0.0681 |

**CuCD-ID / joint_scratch / shap**

| feature                      |   mean_abs_attribution |
|:-----------------------------|-----------------------:|
| TimeRadians                  |                 0.1924 |
| SlidingWindowMeanIntervalSec |                 0.0977 |
| SlidingWindowMinIntervalSec  |                 0.0937 |
| UniqueMessageIDsInWindow     |                 0.0847 |
| MessageCountInWindow         |                 0.06   |

**STIN / bic / ig**

| feature     |   mean_abs_attribution |
|:------------|-----------------------:|
| pkt_len_min |                 0.2774 |
| l_fw_pkt    |                 0.2259 |
| pkt_len_max |                 0.219  |
| fl_byt_s    |                 0.1075 |
| fw_win_byt  |                 0.09   |

**STIN / bic / shap**

| feature     |   mean_abs_attribution |
|:------------|-----------------------:|
| pkt_len_min |                 0.2148 |
| l_fw_pkt    |                 0.1698 |
| pkt_len_max |                 0.1693 |
| fl_byt_s    |                 0.0833 |
| fw_win_byt  |                 0.0729 |

**STIN / icarl_ncm / ig**

| feature     |   mean_abs_attribution |
|:------------|-----------------------:|
| pkt_len_min |                 0.3074 |
| l_fw_pkt    |                 0.2647 |
| pkt_len_max |                 0.222  |
| fl_byt_s    |                 0.0794 |
| fw_win_byt  |                 0.0654 |

**STIN / icarl_ncm / shap**

| feature     |   mean_abs_attribution |
|:------------|-----------------------:|
| pkt_len_min |                 0.2365 |
| pkt_len_max |                 0.1676 |
| l_fw_pkt    |                 0.1396 |
| fl_byt_s    |                 0.0589 |
| fw_win_byt  |                 0.0522 |

**STIN / joint_scratch / ig**

| feature     |   mean_abs_attribution |
|:------------|-----------------------:|
| pkt_len_min |                 0.3135 |
| pkt_len_max |                 0.2248 |
| l_fw_pkt    |                 0.2235 |
| fl_byt_s    |                 0.0823 |
| fw_win_byt  |                 0.0802 |

**STIN / joint_scratch / shap**

| feature     |   mean_abs_attribution |
|:------------|-----------------------:|
| pkt_len_min |                 0.2123 |
| l_fw_pkt    |                 0.1684 |
| pkt_len_max |                 0.1425 |
| fl_byt_s    |                 0.0859 |
| fw_win_byt  |                 0.0604 |

## 6. Interpretation limits

- Deletion/insertion are model-behaviour diagnostics, not causal feature-use evidence.

- The primary medoid background is a single in-distribution reference; the class-balanced medoid-mean sensitivity check is reported in Section 2.

- KernelSHAP completeness is exact by construction; use agreement and faithfulness metrics as accuracy evidence.

- Random-split XAI results inherit the feature-provenance caveats quantified in Sections 5.3 and 7.10 of the main report.

