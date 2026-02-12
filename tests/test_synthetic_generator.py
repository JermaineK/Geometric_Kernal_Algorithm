from pathlib import Path
import sys

import pandas as pd

SYN_PATH = Path(__file__).resolve().parent / "synthetic"
if str(SYN_PATH) not in sys.path:
    sys.path.insert(0, str(SYN_PATH))

from generate_synthetic import generate_dataset, load_config, write_generated_dataset


def test_generate_synthetic_outputs(tmp_path: Path):
    cfg_path = Path("tests/synthetic/configs/testB_hybrid_knee.yaml")
    cfg = load_config(cfg_path)

    generated = generate_dataset(cfg, seed=123, run_index=0)
    assert not generated.data.empty
    required_cols = {"case_id", "L", "omega", "O_L", "O_R", "domain", "test_name"}
    assert required_cols.issubset(set(generated.data.columns))
    assert (generated.data["O_L"] > 0).all()
    assert (generated.data["O_R"] > 0).all()

    out = tmp_path / "synthetic_case"
    write_generated_dataset(out, generated)
    assert (out / "data.csv").exists()
    assert (out / "meta.json").exists()

    df = pd.read_csv(out / "data.csv")
    assert len(df) == len(generated.data)
