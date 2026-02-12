import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def build_dataset(out_dir: Path) -> None:
    rng = np.random.default_rng(7)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(exist_ok=True)
    (out_dir / "assets").mkdir(exist_ok=True)

    rows = []
    L_values = np.array([30, 40, 55, 70, 90, 120, 160, 210, 280, 360], dtype=float)
    times = pd.date_range("2025-01-01", periods=8, freq="6h")
    for case in range(3):
        for t in times:
            for L in L_values:
                base = 0.25 + 0.9 * (L / L_values.max()) ** 0.7
                parity = 0.15 * np.tanh((L - 100.0) / 25.0)
                noise = rng.normal(0.0, 0.03)
                rows.append({"case_id": f"storm_{case}", "t": t, "L": L, "hand": "L", "O": base + parity + noise})
                rows.append({"case_id": f"storm_{case}", "t": t, "L": L, "hand": "R", "O": base - parity + noise})

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "samples.parquet", index=False)

    dataset_yaml = {
        "schema_version": 1,
        "domain": "weather",
        "id": "weather_demo_v1",
        "description": "Synthetic weather parity dataset",
        "units": {"time": "UTC", "L": "km", "omega": "rad/s"},
        "mirror": {"type": "label_swap", "details": {"left": "L", "right": "R"}},
        "columns": {
            "time": "t",
            "size": "L",
            "handedness": "hand",
            "group": "case_id",
            "observable": ["O"],
        },
        "analysis": {
            "knee": {"method": "segmented", "rho": 1.5},
            "scaling": {"method": "wls", "min_points": 4, "exclude_forbidden": True},
            "stability": {"b": 2.0},
            "impedance": {"enabled": True, "tolerance": 0.1},
        },
    }
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_dataset(Path(args.out))
