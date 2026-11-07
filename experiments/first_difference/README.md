# First-differencing robustness check

This folder isolates the session-confounding control suggested by Singh,
Pandey and Kumar, *CubeSat Intrusion Detection Under Session Confounding*
(IEEE Access, 2026). The question is whether the project's class-incremental
conclusions survive after constant session-level resource offsets are removed
from CuCD-ID by first-differencing.

## Run

From the repository root:

```bash
python3 experiments/first_difference/run_first_difference.py
python3 experiments/first_difference/analyze_first_difference.py
```

Defaults: MLP, 15 seeds, 9 methods, `iid` and `temporal_purged_fair`, and four
representations:

- `raw`: `MsgLength` plus the eight `Memory*` resource-state columns.
- `raw_window`: `raw` plus the nine 20-second sliding-window columns.
- `resource_state`: the 17 retained resource-state features in Singh et al.
  (eight memory gauges plus nine window aggregates), with `MsgLength`, time and
  packet identifiers dropped. This is the closest analogue of their control.
- `raw_no_prov`: `raw`, then time, packet identifiers and window aggregates are
  dropped. This is stricter than Singh et al. because it removes the
  differenced window signal too; treat it as a lower-bound control.

The first row of each class is dropped because it has no predecessor.

## Outputs

All outputs are under `experiments/first_difference/results/`:

- `first_difference_runs.csv` — per-seed results.
- `first_difference_summary.csv` — mean/std by differencing set, protocol and method.
- `first_difference_comparison.csv` — paired deltas against
  `results/temporal_purged_compare_MLP.csv`.
- `first_difference_tests.csv` — exact paired sign-flip tests with Holm
  correction within each differencing-set x protocol family.
- `first_difference_report.md` — full accuracy tables.
- `first_difference_takeaways.md` — short interpretation and recommendation.
- `first_difference_setup.json` — protocol metadata.

## Headline result

With `raw` differencing, memory-free methods stay at the 0.25 level and
replay/BiC/iCaRL average 0.973 under IID and 0.837 under the fair temporal
protocol. The qualitative conclusion survives, but time, packet identifiers and
window aggregates remain, so this is only an offset check.

`raw_window` also differences the window aggregates and is a harsher stress
test. Absolute accuracy drops, but memory-free methods still collapse and
exemplar methods still dominate.

`resource_state` is the closest analogue of Singh et al.'s control. After
differencing the 17 resource-state features and dropping `MsgLength`, time and
identifiers, memory-free methods remain at 0.25, while replay/BiC/iCaRL fall to
0.637 IID and 0.524 fair-temporal. Joint/joint_scratch average 0.753 IID and
0.700 fair-temporal. Every exemplar and joint drop is significant after Holm
correction, but exemplars remain about twice chance and well above the
memory-free baselines. A large part of the original 0.837 fair-temporal
accuracy depends on resource-state offsets and provenance fields; the method
gap does not vanish.

`raw_no_prov` is stricter: it also removes the differenced window signal. It
leaves replay/BiC/iCaRL at 0.555 IID and 0.500 fair-temporal. The exact and
stricter controls therefore bracket the fair-temporal online-exemplar result
at roughly 0.47--0.55. Dropping the window signal lowers the bound only
slightly; it does not create the residual skill.

Recommendation: keep this as a secondary robustness analysis. Say that the
conclusions persist after the offset-only control, and that under both combined
controls absolute accuracy falls significantly while exemplar methods remain
above chance. Do not say the confound is removed or that the conclusions are
invariant to it. Do not replace the headline results with this analysis.

The exact and stricter controls are reported in the manuscript's RQ3 and in
the first-differencing table.
