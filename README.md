# GKA (Geometric Kernel Algorithm) - modular diagnostics for knees, parity, and scaling

This repository implements GKA-II: a physics-consistent diagnostic pipeline for
finding regime boundaries ("knees"), mirror-odd response structure, post-knee
parity locking, and BORGT-style size-law scaling.

The toolkit is domain-agnostic. Domain adapters define:
- how to load data
- what the observable `X` is
- what mirror operation `M` is
- what size proxy `L` is
- how to compute impedance-alignment inputs (optional)

## Install

```bash
pip install -e .
```

## Quickstart

1. Prepare a dataset into canonical layout:

```bash
gka prepare --domain weather --in RAW_PATH --out dataset/
```

2. Validate schema:

```bash
gka validate dataset/
```

3. Run pipeline:

```bash
gka run dataset/ --domain weather --config config.yaml --out results/
```

Optional debug artifacts:

```bash
gka run dataset/ --domain weather --config config.yaml --out results/ --dump-intermediates
```

4. Create report:

```bash
gka report --in results/ --out results/report.html
```

## Canonical dataset format

A dataset folder must contain:
- `dataset.yaml` (required)
- `samples.parquet` (required)

`samples.parquet` must include at minimum:
- `case_id`: group identifier
- `t`: time index
- `L`: size proxy (float)
- `hand`: `"L"` or `"R"`
- `O`: scalar observable OR `O_path` to arrays

Pairing rule: each `(case_id, t, L)` must contain both hands.

## How to prepare your dataset (general recipe)

1. Choose your unit of analysis (`case_id`):
   storm id, device id, sweep id, experiment run id, etc.
2. Choose a size proxy `L`:
   geometric outer radius, characteristic wavelength, device size, or patch size.
3. Define the mirror involution `M`:
   spatial reflection (images/fields), label swap (L/R device pairing), or index reversal.
   It must satisfy `M(M(x))=x`.
4. Ensure strict pairing:
   for every `case_id` and `L` at time `t`, provide one `hand="L"` row and one
   `hand="R"` row.
5. Provide the observable:
   scalar `O` (recommended) or `O_path` pointing to array data per record.

Then run `gka validate dataset/` until it passes.

## Outputs

`gka run` writes:
- `results.parquet`: invariants per record/group
- `run_metadata.json`: full reproducibility metadata
- `config_resolved.yaml`: effective config used after defaults and overrides
- optional null distributions under `results/nulls/`

Key invariants include:
- `eta` (mirror-odd contrast)
- `gamma`, `Delta_hat` (BORGT scaling)
- `L_k`, `omega_k` (knee)
- stability class (`stable`/`marginal`/`forbidden`)
- impedance alignment ratio (if enabled)
- `P_lock` (post-knee parity locking)

## Commands

- `gka prepare`: convert raw data into canonical dataset layout.
- `gka validate`: run schema and pairing validation with structured findings.
- `gka run`: execute the full deterministic pipeline.
- `gka report`: produce a single HTML report with key plots and tables.
- `gka diagnose`: run data diagnostics and emit compact operational metrics.
- `gka calibrate --suite synthetic --runs 200`: run synthetic calibration and suggest threshold settings.
- `gka calibrate --suite stress --runs 200`: run adversarial stress calibration and suggest thresholds.
- `gka audit <run_dir>`: explain knee/parity/stability decisions from one run.

## Development

```bash
pip install -e .[dev]
pytest
```
