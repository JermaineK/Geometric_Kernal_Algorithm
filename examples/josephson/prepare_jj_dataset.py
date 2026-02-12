import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def build_dataset(out_dir: Path) -> None:
    rng = np.random.default_rng(21)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(exist_ok=True)
    (out_dir / "assets").mkdir(exist_ok=True)

    rows = []
    L_values = np.array([0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.4, 4.4, 5.8, 7.5], dtype=float)
    for device in range(4):
        for sweep in range(20):
            t = sweep
            for L in L_values:
                base = 0.8 * (L ** 0.4)
                parity = 0.2 * np.tanh((L - 2.5) / 0.8)
                noise = rng.normal(0.0, 0.02)
                rows.append({"case_id": f"jj_{device}", "t": t, "L": L, "hand": "L", "O": base + parity + noise})
                rows.append({"case_id": f"jj_{device}", "t": t, "L": L, "hand": "R", "O": base - parity + noise})

    pd.DataFrame(rows).to_parquet(out_dir / "samples.parquet", index=False)
    spec = {
        "schema_version": 1,
        "domain": "josephson",
        "id": "jj_demo_v1",
        "description": "Synthetic Josephson sweep dataset",
        "units": {"time": "index", "L": "um", "omega": "rad/s"},
        "mirror": {"type": "label_swap", "details": {"left": "L", "right": "R"}},
        "columns": {
            "time": "t",
            "size": "L",
            "handedness": "hand",
            "group": "case_id",
            "observable": ["O"],
        },
        "analysis": {
            "knee": {"method": "segmented", "rho": 1.6},
            "scaling": {"method": "wls", "min_points": 4, "exclude_forbidden": True},
            "stability": {"b": 2.0},
            "impedance": {"enabled": True, "tolerance": 0.12},
        },
    }
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_dataset(Path(args.out))
