import pandas as pd

from gka.calibrate.fit_thresholds import fit_thresholds_from_robustness


def test_threshold_calibration_returns_keys():
    runs = pd.DataFrame(
        {
            "has_knee_true": [True, True, False, False, True, False],
            "knee_p": [0.8, 0.6, 0.2, 0.1, 0.7, 0.3],
            "knee_strength": [1.2, 0.9, 0.1, -0.2, 1.0, 0.0],
            "knee_delta_bic": [12, 9, 1, 0, 10, 2],
            "middle_score": [0.3, 0.4, 0.6, 0.7, 0.35, 0.55],
        }
    )
    out = fit_thresholds_from_robustness(runs, target_fp_max=0.5, objective_beta=1.0)
    assert "knee_p_min" in out
    assert "knee_strength_min" in out
    assert "knee_delta_bic_min" in out
    assert "middle_score_band" in out
