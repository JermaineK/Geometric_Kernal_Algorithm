# Weather Dataset Contract (ERA5-style)

This document defines the minimum preprocessing contract for weather ingestion.

## Supported input

- `xarray.Dataset` or netCDF/zarr readable by xarray.
- Required variables:
  - `u10` (zonal near-surface wind)
  - `v10` (meridional near-surface wind)
- Required dimensions:
  - `time`
  - `latitude`
  - `longitude`

## Adapter

Use `gka.adapters.weather_era5.WeatherERA5Adapter`.

Responsibilities:

- unit-preserving loading from xarray
- finite-value masking
- spatial rolling-window smoothing (configurable window)
- per-time normalization to suppress slow amplitude drift
- mirror definition as longitude reflection

## Produced standardized payload

- `X`: 1D odd-channel proxy from normalized wind field gradient
- `mirror_op`: `{ "type": "spatial_reflection", "axis": "longitude" }`
- `coords`:
  - `time_index`
  - `latitude`
  - `longitude`
- `meta`:
  - source path
  - variable names
  - smoothing window
  - frame count

## Notes

- Keep this adapter layer domain-specific; do not place weather heuristics into
  `src/gka/ops/` or `src/gka/stats/`.
- If your dataset uses different names/dimensions, remap them via adapter args
  instead of editing core diagnostics.
