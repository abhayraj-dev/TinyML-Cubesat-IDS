# Legacy (superseded) result files — retained for provenance only

These files were produced by earlier pipeline versions and are **not** used
by any current figure, table, or paper claim. Do not cite them.

| Legacy file | Superseded by | Why superseded |
|---|---|---|
| `temporal_compare.csv` | `../temporal_purged_compare.csv` | Unpurged temporal split; headline protocol is now IID vs forward-temporal with 0.02 rad TimeRadians purge |
| `stats_wilcoxon.csv` | `../stats_permutation.csv` | Normal-approximation Wilcoxon; headline stats are now exact paired permutation + Holm + Cohen's d |
| `ablation_task_order.csv` | `../task_order_disentangled.csv` | 3-permutation confounded design (budgets move with task); headline is the 12-permutation disentangled study |

All live artifacts are produced by `python3 run_all.py` from the current tree.
