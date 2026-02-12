from pathlib import Path
import sys

SYN_PATH = Path(__file__).resolve().parent / "synthetic"
if str(SYN_PATH) not in sys.path:
    sys.path.insert(0, str(SYN_PATH))

from run_synthetic_suite import SuiteArgs, run_suite


def test_synthetic_suite_smoke(tmp_path: Path):
    args = SuiteArgs(
        config_paths=[
            Path("tests/synthetic/configs/testB_hybrid_knee.yaml").resolve(),
            Path("tests/synthetic/configs/testD_parity_null.yaml").resolve(),
        ],
        runs=3,
        outroot=tmp_path / "outputs",
        expectations=Path("tests/synthetic/expected/expectations.yaml").resolve(),
        seed=42,
        use_cli=False,
        plots=False,
    )
    payload = run_suite(args)

    assert "testB_hybrid_knee" in payload["tests"]
    assert "testD_parity_null" in payload["tests"]
    assert "calibration" in payload["tests"]["testB_hybrid_knee"]
    assert (tmp_path / "outputs" / "suite_results.json").exists()
    assert (tmp_path / "outputs" / "testB_hybrid_knee" / "results.json").exists()
    assert (tmp_path / "outputs" / "testD_parity_null" / "results.json").exists()
