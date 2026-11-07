# First-differencing: takeaways

First-differencing removes constant offsets in selected feature groups before the same task sequences are rebuilt. It is a robustness check, not a replacement for the main protocol, and it does not by itself remove time, identifier or window shortcuts.

## What each representation controls

- `raw` differences `MsgLength` and the eight memory gauges. Time, packet identifiers and window aggregates remain, so this is the weakest control.
- `raw_window` also differences the nine window aggregates but keeps time and packet identifiers. It is a harsher stress test, not a stronger causal control.
- `resource_state` differences the 17 retained resource-state features in Singh et al. (eight memory gauges and nine window aggregates) and drops `MsgLength`, time and packet identifiers. This is the closest analogue of their control.
- `raw_no_prov` differences the raw memory/length features and then drops time, all packet identifiers and all window aggregates. This is stricter than Singh et al. because it removes the differenced window signal entirely; treat it as a lower-bound control, not a replication.

## Headline

### `raw`

- **iid**: memory-free mean ACC 0.257; replay/BiC/iCaRL mean 0.973; best online exemplar `icarl` (0.978). Baseline best exemplar `bic` (0.992).
- **temporal_purged_fair**: memory-free mean ACC 0.257; replay/BiC/iCaRL mean 0.837; best online exemplar `bic` (0.857). Baseline best exemplar `icarl` (0.869).

### `raw_window`

- **iid**: memory-free mean ACC 0.253; replay/BiC/iCaRL mean 0.764; best online exemplar `bic` (0.833). Baseline best exemplar `bic` (0.992).
- **temporal_purged_fair**: memory-free mean ACC 0.258; replay/BiC/iCaRL mean 0.704; best online exemplar `bic` (0.794). Baseline best exemplar `icarl` (0.869).

### `resource_state`

- **iid**: memory-free mean ACC 0.253; replay/BiC/iCaRL mean 0.637; best online exemplar `bic` (0.691). Baseline best exemplar `bic` (0.992).
- **temporal_purged_fair**: memory-free mean ACC 0.250; replay/BiC/iCaRL mean 0.524; best online exemplar `bic` (0.551). Baseline best exemplar `icarl` (0.869).

### `raw_no_prov`

- **iid**: memory-free mean ACC 0.250; replay/BiC/iCaRL mean 0.555; best online exemplar `bic` (0.604). Baseline best exemplar `bic` (0.992).
- **temporal_purged_fair**: memory-free mean ACC 0.250; replay/BiC/iCaRL mean 0.500; best online exemplar `bic` (0.547). Baseline best exemplar `icarl` (0.869).

## Interpretation

1. **`raw` removes the memory/length offsets but not the provenance shortcuts.** Memory-free methods remain at the 0.25 level while online exemplars average 0.973 IID and 0.837 fair-temporal. That is a useful offset-only check, but it cannot answer the full session-confounding critique because time, packet identifiers and window aggregates are still present.
2. **`resource_state` is the closest control to Singh et al.** Memory-free methods remain at 0.25, online exemplars fall to 0.637 IID and 0.524 fair-temporal, and joint/joint_scratch average 0.753 IID and 0.700 fair-temporal. Every exemplar and joint/scratch drop is significant after Holm correction. A large part of the 0.837 fair-temporal accuracy therefore depends on resource-state offsets and provenance fields, but the method gap does not vanish: exemplars stay about twice chance and well above memory-free baselines.
3. **`raw_no_prov` is a stricter lower bound.** It removes the differenced window signal as well, leaving online exemplars at 0.555 IID and 0.500 fair-temporal, and joint/joint_scratch at 0.675 IID and 0.630 fair-temporal. The exact and stricter controls bracket the result at roughly 0.45--0.55 fair-temporal for the online exemplars. Dropping the window signal therefore does not create the residual skill; it lowers the bound only slightly.
4. **The exact exemplar ranking is not stable.** Under baseline fair temporal evaluation iCaRL is best; after `raw` differencing BiC is best, and after `resource_state` BiC is still best (0.551, `bic`), as it is after `raw_no_prov` (0.547, `bic`). Treat this as a difference in ordering: the paired tests compare each method with its own baseline, not BiC against iCaRL, and the ordering changes are not pairwise evidence.
5. **`raw_window` is a stress test, not a better control.** Removing offsets from the window aggregates significantly lowers replay, iCaRL and the replay+EWC hybrid. Window features may carry attack signal, so do not read that drop as confound removal.
6. **Random-split inflation shrinks but does not disappear.** The `resource_state` gap is 0.637 IID against 0.524 fair-temporal, much smaller than the `raw` gap. The paper should not claim that the random-split advantage is entirely independent of provenance fields.

## Recommendation

Use `raw` as the offset-only check, `raw_window` as a stress test, `resource_state` as the closest control to Singh et al., and `raw_no_prov` as the stricter lower bound. Say that the conclusions persist after differencing memory/length offsets, and, in both combined controls, that absolute accuracy drops significantly while exemplar methods remain above chance and above memory-free baselines. Do not say the confound is removed or that the conclusions are invariant to it. Keep this as a secondary robustness analysis, not a new methodological contribution.

## Paired tests

`first_difference_tests.csv` reports exact paired sign-flip tests for each method's ACC change relative to the original run, with Holm correction within each `diff_set x protocol` family of nine methods.
