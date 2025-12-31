# Universal memory index M(t) + EWS + drought-onset diagnostics (CHIRPS monthly)

This repository implements the computational pipeline used in the manuscript:
- CHIRPS monthly precipitation → regional aggregation
- seasonal z-score anomalies z(t)
- drought definition via threshold z_thr
- "real onset" events: first dry month after a non-dry month
- baseline onset-rate mu(t)
- triggered component + universal memory index M(t)
- EWS (variance, AC(1)) on rolling windows
- lead–lag diagnostics, plus Mediterranean-only block-bootstrap uncertainty and minimal OOS tests

## Quickstart

### 1) Create environment
```bash
conda env create -f environment.yml
conda activate universal-memory-ews
