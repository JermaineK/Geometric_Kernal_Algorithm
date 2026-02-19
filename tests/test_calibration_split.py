from pathlib import Path

import pandas as pd

from gka.calibration.fit import fit_calibration_from_parameter_runs, write_calibration
from gka.calibration.score import score_parameter_runs


def test_calibration_fit_and_score(tmp_path: Path):
    runs = pd.DataFrame(
        {
            "has_knee_true": [True, True, False, False, True, False, True, False],
            "knee_p": [0.8, 0.7, 0.2, 0.1, 0.65, 0.3, 0.9, 0.25],
            "knee_strength": [1.1, 0.7, -0.2, -0.3, 0.8, 0.1, 1.2, -0.1],
            "knee_delta_bic": [12, 9, 2, 1, 10, 3, 14, 0],
            "middle_score": [0.2, 0.3, 0.65, 0.75, 0.4, 0.6, 0.2, 0.7],
        }
    )
    runs_path = tmp_path / "parameter_runs.json"
    runs.to_json(runs_path, orient="records", indent=2)

    calibration = fit_calibration_from_parameter_runs(runs_path, target_fp_max=0.5, objective_beta=1.0)
    cal_path = tmp_path / "calibration.json"
    write_calibration(calibration, cal_path)

    score = score_parameter_runs(parameter_runs_path=runs_path, calibration_path=cal_path)
    assert score["schema_version"] == 1
    assert "metrics" in score
    assert "false_positive_rate" in score["metrics"]
    assert "false_negative_rate" in score["metrics"]
