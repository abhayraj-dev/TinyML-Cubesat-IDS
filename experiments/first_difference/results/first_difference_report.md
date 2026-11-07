# First-differencing robustness check

Question: do the class-incremental conclusions survive when the resource-state session offsets identified by Singh et al. are removed by first-differencing?

Setup:

- Architecture: `MLP`
- Seeds: 0--14 (15 seeds)
- Methods: finetune, replay, ewc, full, lwf, bic, icarl, joint, joint_scratch
- Protocols: iid, temporal_purged_fair
- Rows before differencing: 22465
- Rows after differencing: resource_state=22460

Differenced feature sets:

- `raw` (9 columns): `MsgLength`, `MemoryAnonMB`, `MemoryFileMB`, `MemoryKernelstackMB`, `MemoryPageTableMB`, `MemorySocketMB`, `MemoryPercpuMB`, `MemoryShmemMB`, `MemorySlabUnreclaimableMB`
- `raw_window` (18 columns): `MsgLength`, `MemoryAnonMB`, `MemoryFileMB`, `MemoryKernelstackMB`, `MemoryPageTableMB`, `MemorySocketMB`, `MemoryPercpuMB`, `MemoryShmemMB`, `MemorySlabUnreclaimableMB`, `FlowLengthInWindow`, `SlidingWindowMeanIntervalSec`, `SlidingWindowMaxIntervalSec`, `SlidingWindowMinIntervalSec`, `MessageCountInWindow`, `UniqueMessageIDsInWindow`, `AverageMessageLengthInWindow`, `MessageRateInWindow`, `StdDevMessageLengthInWindow`
- `resource_state` (17 columns): `MemoryAnonMB`, `MemoryFileMB`, `MemoryKernelstackMB`, `MemoryPageTableMB`, `MemorySocketMB`, `MemoryPercpuMB`, `MemoryShmemMB`, `MemorySlabUnreclaimableMB`, `FlowLengthInWindow`, `SlidingWindowMeanIntervalSec`, `SlidingWindowMaxIntervalSec`, `SlidingWindowMinIntervalSec`, `MessageCountInWindow`, `UniqueMessageIDsInWindow`, `AverageMessageLengthInWindow`, `MessageRateInWindow`, `StdDevMessageLengthInWindow`
- `raw_no_prov` (9 columns): `MsgLength`, `MemoryAnonMB`, `MemoryFileMB`, `MemoryKernelstackMB`, `MemoryPageTableMB`, `MemorySocketMB`, `MemoryPercpuMB`, `MemoryShmemMB`, `MemorySlabUnreclaimableMB`
- `resource_state` also drops: `TimeRadians`, `MsgId`, `CmdCode`, `ApId`, `MsgLength`
- `raw_no_prov` also drops: `TimeRadians`, `MsgId`, `CmdCode`, `ApId`, `FlowLengthInWindow`, `SlidingWindowMeanIntervalSec`, `SlidingWindowMaxIntervalSec`, `SlidingWindowMinIntervalSec`, `MessageCountInWindow`, `UniqueMessageIDsInWindow`, `AverageMessageLengthInWindow`, `MessageRateInWindow`, `StdDevMessageLengthInWindow`

Baseline source: `results/temporal_purged_compare_MLP.csv`

## raw

### iid

| method        |   ACC_mean |    ACC_std |    FM_mean |    BWT_mean |
|:--------------|-----------:|-----------:|-----------:|------------:|
| joint_scratch |   0.992759 | 0.00249259 | 0.00223515 | -0.00216588 |
| joint         |   0.991196 | 0.00251422 | 0.00307023 | -0.0029748  |
| icarl         |   0.977645 | 0.00820196 | 0.0257056  | -0.025645   |
| bic           |   0.973578 | 0.0120666  | 0.0243847  | -0.0243154  |
| replay        |   0.966341 | 0.0249472  | 0.0400028  | -0.0399854  |
| full          |   0.859016 | 0.141359   | 0.181121   | -0.181087   |
| finetune      |   0.260037 | 0.0316118  | 0.986505   | -0.986505   |
| lwf           |   0.257752 | 0.0297636  | 0.989568   | -0.989568   |
| ewc           |   0.25215  | 0.00632339 | 0.997038   | -0.997038   |

Delta versus baseline:

| method        |   baseline_ACC |   diff_ACC |   delta_ACC |   ACC_wins |   n_seeds |
|:--------------|---------------:|-----------:|------------:|-----------:|----------:|
| lwf           |       0.258797 |   0.257752 | -0.00104526 |          1 |        15 |
| finetune      |       0.262549 |   0.260037 | -0.0025124  |          0 |        15 |
| joint_scratch |       0.996621 |   0.992759 | -0.00386184 |          0 |        15 |
| joint         |       0.995886 |   0.991196 | -0.0046903  |          0 |        15 |
| icarl         |       0.990342 |   0.977645 | -0.0126964  |          1 |        15 |
| bic           |       0.992309 |   0.973578 | -0.0187313  |          0 |        15 |
| ewc           |       0.274353 |   0.25215  | -0.0222035  |          2 |        15 |
| replay        |       0.989406 |   0.966341 | -0.0230652  |          1 |        15 |
| full          |       0.95306  |   0.859016 | -0.0940433  |          1 |        15 |

### temporal_purged_fair

| method        |   ACC_mean |    ACC_std |   FM_mean |   BWT_mean |
|:--------------|-----------:|-----------:|----------:|-----------:|
| joint_scratch |   0.915953 | 0.0288588  | 0.0273546 | -0.016273  |
| joint         |   0.914967 | 0.030537   | 0.0371369 | -0.0245044 |
| bic           |   0.856703 | 0.0738311  | 0.106428  | -0.101989  |
| icarl         |   0.832557 | 0.0741923  | 0.147188  | -0.142954  |
| replay        |   0.82219  | 0.0842942  | 0.165213  | -0.150796  |
| full          |   0.719528 | 0.111027   | 0.291679  | -0.287063  |
| ewc           |   0.268543 | 0.0706228  | 0.943438  | -0.943438  |
| lwf           |   0.251946 | 0.00753765 | 0.970579  | -0.970579  |
| finetune      |   0.251295 | 0.00344737 | 0.971225  | -0.971225  |

Delta versus baseline:

| method        |   baseline_ACC |   diff_ACC |    delta_ACC |   ACC_wins |   n_seeds |
|:--------------|---------------:|-----------:|-------------:|-----------:|----------:|
| replay        |       0.763953 |   0.82219  |  0.0582366   |          9 |        15 |
| joint         |       0.879926 |   0.914967 |  0.0350408   |         12 |        15 |
| bic           |       0.838201 |   0.856703 |  0.0185022   |          8 |        15 |
| joint_scratch |       0.902355 |   0.915953 |  0.0135986   |          9 |        15 |
| lwf           |       0.252216 |   0.251946 | -0.000269473 |          1 |        15 |
| full          |       0.724448 |   0.719528 | -0.00492042  |          8 |        15 |
| finetune      |       0.262745 |   0.251295 | -0.0114502   |          1 |        15 |
| ewc           |       0.285911 |   0.268543 | -0.017368    |          4 |        15 |
| icarl         |       0.869374 |   0.832557 | -0.0368174   |          5 |        15 |

## raw_window

### iid

| method        |   ACC_mean |     ACC_std |    FM_mean |   BWT_mean |
|:--------------|-----------:|------------:|-----------:|-----------:|
| joint_scratch |   0.95469  | 0.0120078   | 0.0104296  | -0.0100396 |
| joint         |   0.953935 | 0.00755911  | 0.00596444 | -0.0054899 |
| bic           |   0.833193 | 0.0374459   | 0.0727639  | -0.0725301 |
| icarl         |   0.737245 | 0.0752367   | 0.321203   | -0.321134  |
| replay        |   0.721672 | 0.0636011   | 0.34839    | -0.348155  |
| full          |   0.624054 | 0.0599308   | 0.469315   | -0.469141  |
| ewc           |   0.257279 | 0.0103948   | 0.987515   | -0.987515  |
| lwf           |   0.25032  | 0.0010919   | 0.997106   | -0.997106  |
| finetune      |   0.250046 | 8.97877e-05 | 0.997184   | -0.997184  |

Delta versus baseline:

| method        |   baseline_ACC |   diff_ACC |   delta_ACC |   ACC_wins |   n_seeds |
|:--------------|---------------:|-----------:|------------:|-----------:|----------:|
| lwf           |       0.258797 |   0.25032  | -0.00847742 |          3 |        15 |
| finetune      |       0.262549 |   0.250046 | -0.0125033  |          2 |        15 |
| ewc           |       0.274353 |   0.257279 | -0.0170744  |          8 |        15 |
| joint_scratch |       0.996621 |   0.95469  | -0.041931   |          0 |        15 |
| joint         |       0.995886 |   0.953935 | -0.0419511  |          0 |        15 |
| bic           |       0.992309 |   0.833193 | -0.159116   |          0 |        15 |
| icarl         |       0.990342 |   0.737245 | -0.253097   |          0 |        15 |
| replay        |       0.989406 |   0.721672 | -0.267734   |          0 |        15 |
| full          |       0.95306  |   0.624054 | -0.329006   |          0 |        15 |

### temporal_purged_fair

| method        |   ACC_mean |   ACC_std |   FM_mean |   BWT_mean |
|:--------------|-----------:|----------:|----------:|-----------:|
| joint         |   0.872904 | 0.0166252 | 0.0144632 |  0.0090369 |
| joint_scratch |   0.871349 | 0.0300548 | 0.0118037 |  0.014163  |
| bic           |   0.794103 | 0.0490665 | 0.120977  | -0.116241  |
| icarl         |   0.681624 | 0.0755502 | 0.330914  | -0.328346  |
| replay        |   0.63492  | 0.0712514 | 0.384007  | -0.362347  |
| full          |   0.501523 | 0.0659172 | 0.547559  | -0.539995  |
| ewc           |   0.267752 | 0.0273968 | 0.902255  | -0.902255  |
| lwf           |   0.256002 | 0.0173387 | 0.926483  | -0.926483  |
| finetune      |   0.250718 | 0.0026659 | 0.935408  | -0.935408  |

Delta versus baseline:

| method        |   baseline_ACC |   diff_ACC |   delta_ACC |   ACC_wins |   n_seeds |
|:--------------|---------------:|-----------:|------------:|-----------:|----------:|
| lwf           |       0.252216 |   0.256002 |  0.00378648 |          2 |        15 |
| joint         |       0.879926 |   0.872904 | -0.00702149 |          8 |        15 |
| finetune      |       0.262745 |   0.250718 | -0.0120276  |          2 |        15 |
| ewc           |       0.285911 |   0.267752 | -0.0181594  |          8 |        15 |
| joint_scratch |       0.902355 |   0.871349 | -0.0310059  |          6 |        15 |
| bic           |       0.838201 |   0.794103 | -0.0440981  |          5 |        15 |
| replay        |       0.763953 |   0.63492  | -0.129033   |          3 |        15 |
| icarl         |       0.869374 |   0.681624 | -0.18775    |          1 |        15 |
| full          |       0.724448 |   0.501523 | -0.222926   |          0 |        15 |

## resource_state

### iid

| method        |   ACC_mean |    ACC_std |   FM_mean |     BWT_mean |
|:--------------|-----------:|-----------:|----------:|-------------:|
| joint         |   0.756767 | 0.0137997  | 0.0075222 |  0.000777349 |
| joint_scratch |   0.749457 | 0.013097   | 0.0295566 | -0.0291084   |
| bic           |   0.690711 | 0.0340471  | 0.126114  | -0.116156    |
| replay        |   0.626389 | 0.0503025  | 0.418264  | -0.418264    |
| icarl         |   0.593092 | 0.0484157  | 0.464214  | -0.464214    |
| full          |   0.528099 | 0.0548104  | 0.550107  | -0.550107    |
| finetune      |   0.254037 | 0.00662583 | 0.985955  | -0.985955    |
| lwf           |   0.252454 | 0.00509302 | 0.98926   | -0.98926     |
| ewc           |   0.251122 | 0.00388328 | 0.98983   | -0.98983     |

Delta versus baseline:

| method        |   baseline_ACC |   diff_ACC |   delta_ACC |   ACC_wins |   n_seeds |
|:--------------|---------------:|-----------:|------------:|-----------:|----------:|
| lwf           |       0.258797 |   0.252454 | -0.0063436  |          3 |        15 |
| finetune      |       0.262549 |   0.254037 | -0.00851207 |          6 |        15 |
| ewc           |       0.274353 |   0.251122 | -0.0232307  |          3 |        15 |
| joint         |       0.995886 |   0.756767 | -0.239118   |          0 |        15 |
| joint_scratch |       0.996621 |   0.749457 | -0.247163   |          0 |        15 |
| bic           |       0.992309 |   0.690711 | -0.301598   |          0 |        15 |
| replay        |       0.989406 |   0.626389 | -0.363017   |          0 |        15 |
| icarl         |       0.990342 |   0.593092 | -0.39725    |          0 |        15 |
| full          |       0.95306  |   0.528099 | -0.42496    |          0 |        15 |

### temporal_purged_fair

| method        |   ACC_mean |     ACC_std |   FM_mean |   BWT_mean |
|:--------------|-----------:|------------:|----------:|-----------:|
| joint         |   0.709015 | 0.0197349   | 0.0299749 | -0.0190316 |
| joint_scratch |   0.691748 | 0.0356576   | 0.0701378 | -0.0634013 |
| bic           |   0.55124  | 0.0507776   | 0.268181  | -0.257444  |
| replay        |   0.547181 | 0.0470823   | 0.468729  | -0.468729  |
| full          |   0.483733 | 0.0513948   | 0.553108  | -0.553108  |
| icarl         |   0.474393 | 0.049068    | 0.571416  | -0.571416  |
| finetune      |   0.250365 | 0.00165802  | 0.973319  | -0.973319  |
| lwf           |   0.249965 | 0.000270901 | 0.975907  | -0.975907  |
| ewc           |   0.249033 | 0.0022501   | 0.969935  | -0.969935  |

Delta versus baseline:

| method        |   baseline_ACC |   diff_ACC |   delta_ACC |   ACC_wins |   n_seeds |
|:--------------|---------------:|-----------:|------------:|-----------:|----------:|
| lwf           |       0.252216 |   0.249965 | -0.00225081 |          2 |        15 |
| finetune      |       0.262745 |   0.250365 | -0.0123796  |          4 |        15 |
| ewc           |       0.285911 |   0.249033 | -0.0368782  |          1 |        15 |
| joint         |       0.879926 |   0.709015 | -0.170911   |          0 |        15 |
| joint_scratch |       0.902355 |   0.691748 | -0.210607   |          0 |        15 |
| replay        |       0.763953 |   0.547181 | -0.216772   |          1 |        15 |
| full          |       0.724448 |   0.483733 | -0.240715   |          1 |        15 |
| bic           |       0.838201 |   0.55124  | -0.286961   |          0 |        15 |
| icarl         |       0.869374 |   0.474393 | -0.394981   |          0 |        15 |

## raw_no_prov

### iid

| method        |   ACC_mean |     ACC_std |      FM_mean |    BWT_mean |
|:--------------|-----------:|------------:|-------------:|------------:|
| joint         |   0.677799 | 0.00708322  | -0.000310848 |  0.00483874 |
| joint_scratch |   0.67184  | 0.0114981   |  0.00848964  | -0.00758296 |
| bic           |   0.604223 | 0.0370491   |  0.254441    | -0.254441   |
| full          |   0.580837 | 0.0682253   |  0.480065    | -0.480065   |
| replay        |   0.552044 | 0.0457384   |  0.537098    | -0.537098   |
| icarl         |   0.509059 | 0.0311148   |  0.611976    | -0.611976   |
| ewc           |   0.250026 | 0.000101294 |  0.977347    | -0.977347   |
| lwf           |   0.250013 | 5.06471e-05 |  0.977412    | -0.977412   |
| finetune      |   0.250007 | 2.53235e-05 |  0.977021    | -0.977021   |

Delta versus baseline:

| method        |   baseline_ACC |   diff_ACC |   delta_ACC |   ACC_wins |   n_seeds |
|:--------------|---------------:|-----------:|------------:|-----------:|----------:|
| lwf           |       0.258797 |   0.250013 | -0.00878431 |          0 |        15 |
| finetune      |       0.262549 |   0.250007 | -0.0125425  |          1 |        15 |
| ewc           |       0.274353 |   0.250026 | -0.0243269  |          0 |        15 |
| joint         |       0.995886 |   0.677799 | -0.318087   |          0 |        15 |
| joint_scratch |       0.996621 |   0.67184  | -0.324781   |          0 |        15 |
| full          |       0.95306  |   0.580837 | -0.372223   |          0 |        15 |
| bic           |       0.992309 |   0.604223 | -0.388086   |          0 |        15 |
| replay        |       0.989406 |   0.552044 | -0.437362   |          0 |        15 |
| icarl         |       0.990342 |   0.509059 | -0.481282   |          0 |        15 |

### temporal_purged_fair

| method        |   ACC_mean |     ACC_std |   FM_mean |   BWT_mean |
|:--------------|-----------:|------------:|----------:|-----------:|
| joint_scratch |   0.643319 | 0.0134081   | 0.0154044 | -0.010571  |
| joint         |   0.617011 | 0.0308629   | 0.0175785 | -0.0146422 |
| bic           |   0.546583 | 0.0549442   | 0.20544   | -0.202163  |
| replay        |   0.503729 | 0.0495924   | 0.521253  | -0.521253  |
| icarl         |   0.449729 | 0.0613371   | 0.632256  | -0.632256  |
| full          |   0.449288 | 0.0580904   | 0.58487   | -0.58487   |
| ewc           |   0.25001  | 3.79928e-05 | 0.970891  | -0.970891  |
| finetune      |   0.25     | 0           | 0.972834  | -0.972834  |
| lwf           |   0.25     | 0           | 0.970857  | -0.970857  |

Delta versus baseline:

| method        |   baseline_ACC |   diff_ACC |   delta_ACC |   ACC_wins |   n_seeds |
|:--------------|---------------:|-----------:|------------:|-----------:|----------:|
| lwf           |       0.252216 |   0.25     | -0.00221569 |          0 |        15 |
| finetune      |       0.262745 |   0.25     | -0.0127451  |          0 |        15 |
| ewc           |       0.285911 |   0.25001  | -0.0359014  |          1 |        15 |
| joint_scratch |       0.902355 |   0.643319 | -0.259035   |          0 |        15 |
| replay        |       0.763953 |   0.503729 | -0.260224   |          0 |        15 |
| joint         |       0.879926 |   0.617011 | -0.262914   |          0 |        15 |
| full          |       0.724448 |   0.449288 | -0.275161   |          0 |        15 |
| bic           |       0.838201 |   0.546583 | -0.291619   |          0 |        15 |
| icarl         |       0.869374 |   0.449729 | -0.419645   |          0 |        15 |

## Reading the result

- A positive `delta_ACC` means first-differencing improved accuracy.
- The key check is not whether every number moves, but whether the qualitative ranking changes: memory-free methods still collapse, exemplar methods still retain most accuracy, and the best exemplar method remains consistent with the original conclusion.
- If the fair-temporal advantage disappears after differencing, the original result is partly a session-offset artifact and should not be used as the headline claim.
