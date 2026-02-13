import numpy as np

from gka.ops.knee import detect_knee


def test_knee_returns_posterior_fields():
    L = np.geomspace(8, 220, 26)
    eta = 0.06 * (L / 20.0) ** 1.5 * np.exp(-((L / 48.0) ** 2.0))
    out = detect_knee(
        eta_series=eta,
        L_series=L,
        method="segmented",
        min_points=6,
        bic_delta_min=3.0,
        knee_p_min=0.2,
        knee_strength_min=-0.5,
        bootstrap_n=64,
        rng=np.random.default_rng(7),
    )
    assert 0.0 <= out.knee_p <= 1.0
    assert out.knee_ci[0] is None or out.knee_ci[1] is None or out.knee_ci[0] <= out.knee_ci[1]
    assert out.middle_label in {"resolved", "forbidden_middle"}
