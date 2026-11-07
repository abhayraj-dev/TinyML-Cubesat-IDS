# TinyML CubeSat Class-Incremental IDS

## What this project is

This is a reproducible research codebase for **onboard class-incremental
intrusion detection on CubeSat telemetry**. It asks a practical mission
question: can a small IDS learn new attack classes after launch without
retraining from scratch, forgetting earlier attacks, or exceeding the
spacecraft's memory, compute, and uplink budgets?

The project is implemented from scratch in NumPy. It contains the full
experimental pipeline, frozen hyperparameters, seed-level results, statistical
tests, XAI tables, and the final v6.7 report.

## What it does

- Builds class-incremental tasks where every new task introduces attack
  classes that never appeared before, using a single growing classifier head.
- Simulates the onboard constraint: task 0 gets a larger pre-launch corpus,
  later tasks get only a small 150-sample/class uplink budget.
- Compares fine-tuning, EWC, LwF, experience replay, Replay+EWC, BiC-lite,
  iCaRL-lite, and joint-training references across MLP, GRU, and 1-D CNN
  backbones.
- Measures accuracy, backward transfer, forgetting, calibration, per-class F1,
  persistent memory, compute, and uplink payload.
- Audits leakage and validity with temporal splits, feature ablations,
  first-differencing controls, task-order tests, buffer-size sweeps, and
  budget-aware calibration (BUAC).
- Uses CuCD-ID as the primary simulated CubeSat telemetry benchmark, STIN as a
  cross-domain stress test, and SCAD as an external boundary validation.

**All CuCD-ID telemetry is simulated** (NOS3/cFS software-in-the-loop digital
twin), not captured from an orbiting spacecraft. No result here is a
deployment or flight-certification claim.

## Release snapshot

| Item | Location |
|---|---|
| Final report | [`v6_7_project_report_final.md`](v6_7_project_report_final.md) |
| Results summary | [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md) |
| XAI supplement | [`results/XAI_SUPPLEMENT.md`](results/XAI_SUPPLEMENT.md) |
| Changelog | [`CHANGELOG.md`](CHANGELOG.md) |
| Full experiment log | [`DATALOG.md`](DATALOG.md) |
| Main result matrix | `results/main_results_summary.csv` |
| First-differencing controls | `experiments/first_difference/results/` |

The manuscript, planning notes, and other private project documents are
intentionally **not part of this repository**.

## Headline result

On the primary **CuCD-ID / MLP** protocol (4 tasks, 5 classes, 15 seeds),
experience replay, BiC-lite, and iCaRL-lite retain near-joint accuracy, while
fine-tuning and EWC suffer catastrophic forgetting:

| Method | Accuracy | BWT | Forgetting | Persistent state |
|---|---:|---:|---:|---:|
| Fine-tune | 0.263 ± 0.037 | −0.983 | 0.983 | 1.3 KB |
| EWC | 0.274 ± 0.056 | −0.967 | 0.967 | 6.6 KB |
| Replay | **0.989 ± 0.007** | −0.011 | 0.011 | 13.3 KB |
| Replay+EWC | 0.953 ± 0.065 | −0.060 | 0.060 | 12.6 KB |
| BiC-lite | **0.992 ± 0.003** | −0.006 | 0.006 | 13.3 KB |
| iCaRL-lite | **0.990 ± 0.005** | −0.011 | 0.011 | 13.3 KB |
| Joint (scratch) | 0.997 ± 0.001 | −0.002 | 0.002 | 1.3 KB |

On **STIN / MLP** (3 tasks, 9 attack classes), the same ordering holds:
BiC-lite 0.687 accuracy, iCaRL-lite 0.658, replay 0.648, versus fine-tune
0.288 and EWC 0.269. SCAD is a separate external validation and is reported
as a boundary/negative result, not as another headline win.

All primary and STIN gains are significant after Holm correction across the
full confirmatory matrix. Paired tests and effect sizes are in
`results/stats_*.csv`; seed-level data are in the `*_runs.csv` files.

## First-differencing control (v6.7)

The original fair-temporal result used raw telemetry and could exploit
constant recording/session offsets. v6.7 adds a within-recording
first-differencing control and removes provenance fields:

| Strategy | Fair raw | Fair 17-feat. | Fair no-prov. | IID 17-feat. | IID no-prov. |
|---|---:|---:|---:|---:|---:|
| Fine-tune | 0.251 | 0.250 | 0.250 | 0.254 | 0.250 |
| Replay | 0.822 | 0.547 | 0.504 | 0.626 | 0.552 |
| Replay+EWC | 0.720 | 0.484 | 0.449 | 0.528 | 0.581 |
| BiC-lite | 0.857 | 0.551 | 0.547 | 0.691 | 0.604 |
| iCaRL-lite | 0.833 | 0.474 | 0.450 | 0.593 | 0.509 |
| Joint (scratch) | 0.916 | 0.692 | 0.643 | 0.749 | 0.672 |

The 17-feature set keeps only resource-state features (eight memory gauges
and nine window aggregates). The no-provenance set is stricter: differenced
raw values only, with time, identifiers, and window aggregates removed.

**Interpretation:** exemplar methods retain roughly half of their original
fair-temporal accuracy, at **0.45–0.55**, while memory-free methods stay at
0.250. Every exemplar/joint drop is significant after Holm correction
(`p < 0.001`). This is still not a leakage-free estimate: first-differencing
removes constant offsets, not slowly varying session signatures, and the
control adapts rather than replicates the Singh et al. calibration.

### What is and is not claimed

**Supported**

- Experience replay is the main viable mechanism for this simulated
  CubeSat CIL setting.
- BiC-lite and iCaRL-lite improve calibration/forgetting at a few KB of
  persistent state, with BiC-lite the best accuracy/rejection trade-off.
- The IID, temporal, and first-differenced orderings are distinct estimands;
  the first two are not interchangeable.
- BUAC's probability gate beats fixed thresholds; with BiC-lite the cost-aware
  gate gives 0.989 probability-of-correct decisions, ~98% expected-cost
  reduction, and 92% safe-rejection accuracy.

**Not supported**

- Flight readiness, real-orbit validation, or deployment guarantees.
- A leakage-free accuracy estimate from the IID split.
- The claim that the temporal/first-differenced confound is fully removed.
- A strict upper bound from `joint_scratch`; it is an offline reference.

## Repository layout

```
src/            NumPy models, training loop, CMMU, metrics, XAI
experiments/    Experiment runners and summary generator
  first_difference/   v6.7 differencing controls
results/        CSV/JSON result matrices, stats, XAI tables
figures/        Generated figures
data/           Dataset download instructions and label schema
tests/          Gradient, metric, split, and protocol tests
analysis/       Exploratory analysis
run_all.py      Full pipeline driver
```

Key source files: `src/models.py` (MLP/GRU/CNN1D with hand-written
backward passes), `src/train.py` (CIL loop and growing head),
`src/data_utils.py` (task construction and splits), `src/cmmu.py`
(calibration/memory accounting), `src/xai.py` (explanation tables).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ is required. The pipeline uses only NumPy, pandas, matplotlib,
and tabulate; there is no PyTorch/TensorFlow/sklearn dependency.

## Dataset download

The raw CSVs are deliberately not committed. Download them from the official
sources and place them at the expected paths:

| Dataset | Download | Destination |
|---|---|---|
| CuCD-ID | [Mendeley Data v3](https://data.mendeley.com/datasets/7n2d42pm3n/3) | `data/CuCD-ID/noised_dataset.csv` |
| STIN | [kun9717/STIN-data-set](https://github.com/kun9717/STIN-data-set) | `data/STIN/SAT20.csv`, `data/STIN/TER20.csv` |
| SCAD | Original external source; not redistributed | See `DATALOG.md` |

Exact files, SHA-256 checksums, citations, and upstream license notes are in
[`data/README.md`](data/README.md). The committed results were produced with
the files whose checksums are listed there.

## Reproduce

The full pipeline is ordered and resumable:

```bash
python3 run_all.py
```

Individual stages:

```bash
python3 experiments/run_validation_tune.py        # freeze hyperparameters
python3 experiments/run_main.py                   # primary CuCD-ID matrix
python3 experiments/run_purged_temporal.py        # fair temporal protocol
python3 experiments/run_stats.py                  # paired tests + Holm
python3 experiments/run_buac.py                   # budget-aware calibration
python3 experiments/run_xai.py                    # explanation tables
python3 experiments/run_scad.py --archs MLP --tag mlp_primary
python3 experiments/generate_summary_report.py    # regenerate reports
```

First-differencing controls:

```bash
python3 experiments/first_difference/run_first_difference.py
python3 experiments/first_difference/analyze_first_difference.py
```

Do not tune on test results. Hyperparameters are tuned on validation data and
frozen in `results/best_hparams_per_method_arch.json` before confirmatory
runs; downstream code calls `load_frozen_hparams_strict()` and refuses to
silently fall back.

## Tests

The tests are standalone scripts, not `unittest` cases:

```bash
for f in tests/test_*.py; do echo "== $f"; python3 "$f" || exit 1; done
```

Tests cover finite-difference gradients (including the growing head), BUAC
properties, split/calibration leakage guards, feature groups, NCM, SCAD
splits, temporal sampling, task order, statistics, and XAI invariants.

## Data and provenance

- **CuCD-ID** — Fayyaz et al., *Data in Brief* (2026),
  doi:10.1016/j.dib.2026.112598; data doi:10.17632/7n2d42pm3n.3.
  Simulated NOS3/cFS CubeSat telemetry. The dataset is CC BY 4.0; download
  `noised_dataset.csv` to `data/CuCD-ID/` as described in [`data/README.md`](data/README.md).
- **STIN** — Li et al., *IEEE Access* (2020),
  doi:10.1109/ACCESS.2020.3041641. Used as a cross-domain stress test. The
  upstream license does not clearly permit redistribution, so `SAT20.csv` and
  `TER20.csv` are downloaded separately rather than committed. The provided
  files have no benign class.
- **SCAD** — external 4-class CIL validation only; the raw dataset is not
  redistributed.

CuCD-ID contains a circular time field, CCSDS packet identifiers, and
20-second sliding-window aggregates. The IID split can share overlapping
windows across train/test; it is a benchmark estimand, not a deployment
estimate. The fair temporal protocol, cold-start support control, feature
ablation, and v6.7 first-differencing controls are separate probes with
different limitations.

## Uplink budget

Task 0 is trained on 700 samples/class; later tasks use 150 samples/class.
Reported uplink bytes include adaptation samples plus the onboard
calibration/val-balanced Fisher payload, using packed float32/int8 estimates.
CSV text formatting, quantization metadata, and protocol framing are excluded.
See Section 4 of `results/RESULTS_SUMMARY.md` for the per-task ledger.


## License

The original code and repository materials are released under the MIT License
(see [`LICENSE`](LICENSE)). Third-party datasets are **not** covered by that
license and remain subject to their upstream terms; see
[`data/README.md`](data/README.md) for attribution and download details.
