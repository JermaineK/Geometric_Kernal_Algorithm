# Weather Minipilot

This is a small end-to-end pilot for validating real-data plumbing before full-scale runs.

## 1) Prepare a tiny paired dataset

```bash
python examples/weather_minipilot/prepare_minipilot.py \
  --input data/raw/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \
  --out dataset/weather_minipilot \
  --max-pairs 600
```

## 2) Validate

```bash
gka validate dataset/weather_minipilot
```

## 3) Run diagnostics

```bash
gka run dataset/weather_minipilot --domain weather --config examples/weather_minipilot/config.yaml --out results/weather_minipilot --dump-intermediates
```

## 4) Build report bundle

```bash
gka report --in results/weather_minipilot --out results/weather_minipilot/report.html
```

This pilot is expected to be conservative:

- no crashes
- explainable knee accept/reject path (`gka audit`)
- parity null behavior can be checked via stress controls before scaling up
