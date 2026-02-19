from gka.metrics.forbidden_middle import compute_forbidden_middle_score


def test_forbidden_middle_reason_codes():
    flagged = compute_forbidden_middle_score(
        knee_probability=0.5,
        knee_ci=(30.0, 75.0),
        L_bounds=(10.0, 120.0),
        knee_strength=0.2,
        delta_gamma=0.4,
        slope_drift=0.18,
        ridge_strength=0.4,
        bootstrap_cov=0.42,
        threshold=0.6,
    )
    assert flagged.label == "forbidden_middle"
    assert "slope_drift_high" in flagged.reason_codes
    assert any(code.startswith("posterior_") for code in flagged.reason_codes)
    assert flagged.width is not None and flagged.width > 0.0

    resolved = compute_forbidden_middle_score(
        knee_probability=0.95,
        knee_ci=(48.0, 52.0),
        L_bounds=(10.0, 120.0),
        knee_strength=1.2,
        delta_gamma=0.8,
        slope_drift=0.02,
        ridge_strength=2.5,
        bootstrap_cov=0.1,
        threshold=0.6,
    )
    assert resolved.label == "resolved"
    assert resolved.width == 0.0
