import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def build_dataset(out_dir: Path) -> None:
    rng = np.random.default_rng(5)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = out_dir / "arrays"
    arrays_dir.mkdir(exist_ok=True)
    (out_dir / "assets").mkdir(exist_ok=True)

    rows = []
    L_values = np.array([12, 16, 20, 24, 32, 40, 48, 64, 80, 96], dtype=float)
    for case in range(5):
        for idx, L in enumerate(L_values):
            t = idx
            left = rng.normal(loc=0.2 + 0.01 * L, scale=0.05, size=(24, 24))
            right = np.fliplr(left) + rng.normal(0.0, 0.01, size=(24, 24))
            left_path = arrays_dir / f"case{case}_L{int(L)}_left.npy"
            right_path = arrays_dir / f"case{case}_L{int(L)}_right.npy"
            np.save(left_path, left)
            np.save(right_path, right)
            rows.append({"case_id": f"oam_{case}", "t": t, "L": L, "hand": "L", "O_path": str(left_path.relative_to(out_dir))})
            rows.append({"case_id": f"oam_{case}", "t": t, "L": L, "hand": "R", "O_path": str(right_path.relative_to(out_dir))})

    pd.DataFrame(rows).to_parquet(out_dir / "samples.parquet", index=False)
    spec = {
        "schema_version": 1,
        "domain": "oam",
        "id": "oam_demo_v1",
        "description": "Synthetic OAM image parity dataset",
        "units": {"time": "index", "L": "pixels", "omega": "rad/s"},
        "mirror": {"type": "spatial_reflection", "details": {"axis": "x"}},
        "columns": {
            "time": "t",
            "size": "L",
            "handedness": "hand",
            "group": "case_id",
            "observable": ["O_path"],
        },
        "analysis": {
            "knee": {"method": "log_curvature", "rho": 1.5},
            "scaling": {"method": "theil_sen", "min_points": 4, "exclude_forbidden": True},
            "stability": {"b": 2.0},
            "impedance": {"enabled": False, "tolerance": 0.1},
        },
    }
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_dataset(Path(args.out))
