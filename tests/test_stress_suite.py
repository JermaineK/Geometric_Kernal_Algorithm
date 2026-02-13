from pathlib import Path
import sys

SYN_PATH = Path(__file__).resolve().parent / "synthetic"
if str(SYN_PATH) not in sys.path:
    sys.path.insert(0, str(SYN_PATH))
STRESS_PATH = Path(__file__).resolve().parent / "stress"
if str(STRESS_PATH) not in sys.path:
    sys.path.insert(0, str(STRESS_PATH))

from run_synthetic_suite import SuiteArgs, run_suite
from run_stress_suite import build_robustness_report


def test_stress_suite_smoke(tmp_path: Path):
    args = SuiteArgs(
        config_paths=[
            Path("tests/stress/configs/stressB_hybrid_heavytail.yaml").resolve(),
            Path("tests/stress/configs/stressF_screening_control.yaml").resolve(),
        ],
        runs=2,
        outroot=tmp_path / "stress_outputs",
        expectations=Path("tests/stress/expected/expectations.yaml").resolve(),
        seed=7,
        use_cli=False,
        plots=False,
    )
    payload = run_suite(args)

    assert "stressB_hybrid_heavytail" in payload["tests"]
    assert "stressF_screening_control" in payload["tests"]
    cal = payload["tests"]["stressB_hybrid_heavytail"]["runs"][0]["calibration"]
    assert "tau_s_hat" in cal
    assert "S_mu_k" in cal
    assert "band_hit_rate" in cal
    assert (tmp_path / "stress_outputs" / "suite_results.json").exists()


def test_stress_robustness_report_smoke(tmp_path: Path):
    report = build_robustness_report(
        outroot=tmp_path / "stress_diag",
        expectations_path=Path("tests/stress/expected/expectations.yaml").resolve(),
        use_cli=False,
        seed=11,
        robustness_samples=2,
        blind_n=3,
    )
    assert "parameter_robustness" in report
    assert "blind_test" in report
    assert "invariant_stability_map" in report
