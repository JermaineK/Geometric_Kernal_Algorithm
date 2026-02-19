# Stress Suite for GKA Transfer Readiness

This suite extends the synthetic harness with adversarial failure modes that are common in real weather and field data.

## Purpose

The stress suite verifies that GKA remains stable under:
- nonstationary drift
- temporal warping
- heavy-tailed noise and burst outliers
- missingness and duplicate timestamps
- multi-regime overlap
- parity confounds

Synthetic pass is necessary but not sufficient. Stress pass is the transfer-readiness gate.

## Run one stress test

```bash
python tests/stress/run_stress_suite.py --configs tests/stress/configs/stressB_hybrid_heavytail.yaml --runs 50
```

## Run full stress suite

```bash
python tests/stress/run_stress_suite.py --configs tests/stress/configs/*.yaml --runs 200
```

This command now also writes `robustness_report.json` by default, including:
- parameter robustness sweep (noise, L samples, slope, xi, tau_s)
- blind synthetic regime classification (default `n=100`)
- invariant stability map over `(gamma, xi/L, tau_s, impedance_ratio)`
- calibrated threshold recommendation at `tests/stress/outputs/calibration.json`

## Outputs

Outputs mirror the synthetic suite format under `tests/stress/outputs/`:
- `suite_results.json`
- `suite_summary.md`
- `robustness_report.json`
- `robustness_gate.json`
- per-test `results.json` and `summary.md`
- per-run `data.csv`, `meta.json`, `gka_output/*`

Calibration entries include:
- `gamma_true`, `gamma_hat`, `gamma_abs_err`, `gamma_abs_pct_err`
- `omega_k_true`, `omega_k_hat`, `omega_k_abs_err`
- `tau_s_true`, `tau_s_hat`, `tau_s_abs_err`
- `S_mu_k`, `W_mu`, `band_hit_rate`

Additional stress configs:
- `stressG_heavytail_1f_no_knee`: 1/f heavy-tail drift without knee.
- `stressH_fake_knee_logistic`: logistic curvature control for fake knees.
- `stressI_sign_flip_parity`: stochastic parity sign flips per scale.
- `stressJ_correlated_non_spiral`: `omega_k ~ 1/L` alignment without coherent stability.
- `stressK_sparse_spectral_peaks`: sparse random peaks unrelated to predicted bands.
- `stressL_low_contrast_real_knee`: real but low-contrast knees (recall stressor).
- `stressM_multiknee`: dual-knee mixture with broadened transition.
- `stressN_timewarp_knee`: true knee under strong temporal warping/missingness.
- `stressO_borderline_forbidden`: borderline logistic transition guard against over-triggering forbidden-middle.

## Interpretation

If stress tests fail while synthetic baseline passes, the failure is usually transfer fragility:
- overfitting to idealized knees
- parity leakage under confounds
- unstable band labeling under drift/noise

Treat stress failures as blockers for real-data claims.
