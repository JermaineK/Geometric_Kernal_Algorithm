import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def build_dataset(out_dir: Path) -> None:
    rng = np.random.default_rng(11)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(exist_ok=True)
    (out_dir / "assets").mkdir(exist_ok=True)

    rows = []
    L_values = np.array([0.12, 0.15, 0.19, 0.24, 0.3, 0.38, 0.47, 0.58, 0.72, 0.9])
    omega = np.linspace(1.0e9, 4.0e9, 6)
    for device in range(4):
        for t in range(12):
            for i, L in enumerate(L_values):
                base = 0.5 + 0.2 * i
                parity = 0.05 * np.tanh((L - 0.28) / 0.07)
                noise = rng.normal(0, 0.01)
                cm = 2.1e8 + rng.normal(0, 2.0e6)
                rows.append({"case_id": f"res_{device}", "t": t, "L": L, "hand": "L", "O": base + parity + noise, "omega": omega[i], "cm": cm})
                rows.append({"case_id": f"res_{device}", "t": t, "L": L, "hand": "R", "O": base - parity + noise, "omega": omega[i], "cm": cm})

    pd.DataFrame(rows).to_parquet(out_dir / "samples.parquet", index=False)

    spec = {
        "schema_version": 1,
        "domain": "em_resonator",
        "id": "em_demo_v1",
        "description": "Synthetic EM resonator parity dataset",
        "units": {"time": "index", "L": "m", "omega": "rad/s"},
        "mirror": {"type": "label_swap", "details": {"left": "L", "right": "R"}},
        "columns": {
            "time": "t",
            "size": "L",
            "handedness": "hand",
            "group": "case_id",
            "observable": ["O"],
        },
        "analysis": {
            "knee": {"method": "cpt", "rho": 1.4},
            "scaling": {"method": "wls", "min_points": 4, "exclude_forbidden": True},
            "stability": {"b": 2.0},
            "impedance": {"enabled": True, "tolerance": 0.08},
        },
    }
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_dataset(Path(args.out))
