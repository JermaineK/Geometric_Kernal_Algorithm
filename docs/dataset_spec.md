# Dataset Specification

## Canonical folder layout

```text
dataset/
|- dataset.yaml
|- samples.parquet
|- arrays/
|  |- X.npy (optional)
|  |- X_plus.npy (optional cache)
|  `- X_minus.npy (optional cache)
`- assets/
   |- raw/ (optional)
   `- notes.md (optional)
```

## `dataset.yaml` required fields

```yaml
schema_version: 1
domain: weather | josephson | oam | em_resonator | plasma | custom
id: "coral_sea_v1"
description: "Short description"
units:
  time: "hours since 1970-01-01"
  L: "km"
  omega: "rad/s"
mirror:
  type: "spatial_reflection|label_swap|index_reverse|custom"
  details: {}
columns:
  time: "t"
  size: "L"
  handedness: "hand"
  group: "case_id"
  observable: ["O"]
analysis:
  knee:
    method: segmented
    rho: 1.5
  scaling:
    method: wls
    min_points: 4
    exclude_forbidden: true
  stability:
    b: 2.0
  impedance:
    enabled: true
    tolerance: 0.1
```

## `samples.parquet` minimum columns

- `case_id`: int or string
- `t`: timestamp or numeric index
- `L`: float size proxy
- `hand`: `L` or `R`
- `O`: scalar observable (or `O_path` as file pointer)

## Pairing rule

For every `(case_id, t, L)` key there must be one `hand=L` row and one `hand=R` row.
Validation fails when completeness is below `99%` unless `--allow-missing` is passed.
