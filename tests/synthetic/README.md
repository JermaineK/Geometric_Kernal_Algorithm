# Synthetic Suite for GKA

This folder provides a self-contained synthetic validation harness for GKA. It is the
first gate before running any real weather, JJ, OAM, EM, or plasma data.

**Synthetic suite is the gatekeeper: do not trust real-data results until synthetic suite passes.**

## Purpose

The suite checks GKA behavior as a diagnostic pipeline:
- knee detection correctness
- scaling slope recovery
- forbidden-middle isolation behavior
- parity/null integrity under pairing destruction
- optional impedance alignment consistency
- optional stability classification consistency

## Run one synthetic dataset

```bash
python tests/synthetic/generate_synthetic.py \
  --config tests/synthetic/configs/testB_hybrid_knee.yaml \
  --outdir tests/synthetic/outputs/testB_hybrid_knee \
  --seed 123
```

## Run full Monte Carlo suite

```bash
python tests/synthetic/run_synthetic_suite.py \
  --configs tests/synthetic/configs/*.yaml \
  --runs 50 \
  --outroot tests/synthetic/outputs \
  --seed 12345
```

Add `--plots` to produce quick histograms for `gamma` and knee locations.

## Tests

- `testA_no_knee`: clean power law, checks false-knee rejection and gamma recovery.
- `testB_hybrid_knee`: injected coherence cutoff, checks knee localization and flattening.
- `testC_forbidden_middle`: injected unstable middle regime, checks band isolation behavior.
- `testD_parity_null`: label swaps + pair breaking, checks parity collapse and lock decay.
- `testE_impedance_align` (optional): checks `omega_k*L/(2*pi*c)` ratio centering near unity.
- `testF_eigen_stability` (optional): checks stability class consistency with injected gamma.

## Outputs

Per test and per run:
- `outputs/<test_name>/run_0001/data.csv`
- `outputs/<test_name>/run_0001/meta.json`
- `outputs/<test_name>/run_0001/results.json`
- `outputs/<test_name>/run_0001/gka_output/*`

Per test aggregate:
- `outputs/<test_name>/results.json`
- `outputs/<test_name>/summary.md`
- `outputs/<test_name>/data.csv` (copy of first run)
- `outputs/<test_name>/meta.json` (copy of first run)

Suite aggregate:
- `outputs/suite_results.json`
- `outputs/suite_summary.md`

## Failure interpretation

A test fails when one or more configured thresholds in
`tests/synthetic/expected/expectations.yaml` are violated.

Typical implications:
- false knees in Test A: knee detector is over-triggering noise/trend curvature
- missed knees in Test B: transition logic is weak or confidence calibration is off
- Test C failure: forbidden-middle behavior is being averaged away
- Test D failure: parity metrics are not robust to broken mirror pairing
