# Dataset download and placement

The raw dataset CSVs are **not committed** to this repository. Download them
from their official sources and place them at the paths below before running
the pipeline.

## CuCD-ID (primary benchmark)

- **Dataset:** CubeSat Cybersecurity Dataset for Intrusion Detection (CuCD-ID)
- **Source:** https://data.mendeley.com/datasets/7n2d42pm3n/3
- **File to download:** `noised_dataset.csv` (version 3, 22,465 rows)
- **Place at:** `data/CuCD-ID/noised_dataset.csv`
- **Included here:** `data/CuCD-ID/label_schema.json`
- **SHA-256:** `656fc2f23544469d8cbdca631747debc8dae7164621a2cd6740a5928cfae3c68`
- **License:** CC BY 4.0
- **Citation:** Fayyaz et al., “CubeSat cybersecurity dataset for intrusion
  detection (CuCD-ID): Labelled NOS3/cFS telemetry (raw + augmented) with
  COSMOS reproduction scripts,” *Data in Brief* 65 (2026) 112598.
  https://doi.org/10.1016/j.dib.2026.112598

All CuCD-ID telemetry is simulated with NASA NOS3/cFS; it was not captured
from an orbiting spacecraft.

## STIN (cross-domain stress test)

- **Dataset:** SAT20 + TER20 flow-based satellite/terrestrial network security data
- **Source:** https://github.com/kun9717/STIN-data-set
- **Files to download:**
  - `SAT20.csv`
  - `TER20.csv`
- **Place at:**
  - `data/STIN/SAT20.csv`
  - `data/STIN/TER20.csv`
- **Direct links:**
  - https://raw.githubusercontent.com/kun9717/STIN-data-set/main/SAT20.csv
  - https://raw.githubusercontent.com/kun9717/STIN-data-set/main/TER20.csv
- **SHA-256:**
  - `SAT20.csv`: `cedb4245c0983019ca3462f5ac46a209b556aa0c9b5baac0c7d5445f67141391`
  - `TER20.csv`: `6b865c234fd772ba6ff1b1d8bf6253125b7b7a22c54a9d332f0f7572404f8910`
- **Upstream terms:** copyright National Engineering Lab for Next Generation
  Internet Interconnection Devices, BJTU. The upstream `LICENSE.txt` requests
  citation but does not explicitly grant redistribution, so the raw CSVs are
  not redistributed here.
- **Citation:** Li et al., “Distributed Network Intrusion Detection System in
  Satellite-Terrestrial Integrated Networks using Federated Learning,”
  *IEEE Access* (2020). https://doi.org/10.1109/ACCESS.2020.3041641

The provided STIN files contain no benign/normal class; they are used only as
a larger-scale, higher-class-count cross-domain stress test.

## SCAD (external boundary validation)

SCAD is not redistributed. Its experiment is optional and its raw data path is
documented in `DATALOG.md`; obtain SCAD from its original source before running
`experiments/run_scad.py`.

## Verify downloads

```bash
sha256sum data/CuCD-ID/noised_dataset.csv \
          data/STIN/SAT20.csv \
          data/STIN/TER20.csv
```

The checksums above match the files used for the committed results.
