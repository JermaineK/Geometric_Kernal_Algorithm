# weather_min adapter

Minimal real-data bridge for GKA domain transfer checks.

## Input
- `xarray.Dataset` on disk (NetCDF/Zarr)
- preferred variables: `u10`, `v10`
- fallback: `vorticity` or first numeric variable

## Output
`gka_run.json` containing:
- `Omega_hat`
- `knee_L_hat`, `knee_confidence`
- `gamma_hat`, `Delta_b`
- `tau_s_hat`, `S_at_mu_k`, `W_mu`, `band_label`

## Usage

```bash
python adapters/weather_min/run_weather_min.py \
  --in path/to/tile.nc \
  --out adapters/weather_min/gka_run.json \
  --seed 42
```

This is intentionally small and diagnostic-first. It validates end-to-end plumbing on a single region/time slice before scaling to full production weather pipelines.
