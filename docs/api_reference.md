# API Reference

## CLI

- `gka prepare --domain <name> --in <raw> --out <dataset-dir>`
- `gka validate <dataset-dir> [--allow-missing] [--json]`
- `gka run <dataset-dir> --domain <name> --out <results-dir> [--config path] [--null N] [--dump-intermediates]`
- `gka report --in <results-dir> [--out <report.html>] [--out-json <report.json>] [--out-md <report.md>]`
- `gka diagnose --data <dataset-dir> [--domain <name>] [--config <path>]`
- `gka calibrate --suite stress --runs 200 [--calibration-out <calibration.json>]`
- `gka score --parameter-runs <parameter_runs.json> --calibration <calibration.json> [--out <score_report.json>]`
- `gka audit <run-dir> [--json]`

## Core adapter protocol

`gka.domains.base.DomainAdapter` defines:
- `load(dataset_path)`
- `mirror_map(bundle)`
- `observable(bundle)`
- `size_proxy(bundle)`
- `frequency_proxy(bundle)`
- `impedance_proxy(bundle)`

## Pipeline outputs

`results.parquet` includes per-scale and global diagnostics:
- `L`, `eta`, `E_plus`, `E_minus`
- `L_k`, forbidden band bounds
- `gamma`, `Delta_hat`, drift diagnostics
- stability class and inequality status
- impedance ratio and pass/fail
- coherence metrics `A`, `F`, `P_lock`
- operational diagnostics: `omega_k_hat`, `tau_s_hat`, `S_at_mu_k`, `W_mu`, `W_L`,
  `R_align`, `M_Z`, `band_hit_rate`, `band_class_hat`, `eigen_band`, `stability_margin`,
  `forbidden_middle_width`, `forbidden_middle_center`, `forbidden_middle_reason_codes`
