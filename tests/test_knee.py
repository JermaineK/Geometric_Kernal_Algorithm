import numpy as np

from gka.ops.knee import detect_knee


def test_knee_segmented_detects_transition():
    L = np.array([8, 10, 13, 16, 20, 26, 34, 44, 58, 76, 100, 132, 174, 230], dtype=float)
    eta = np.array([0.02, 0.03, 0.045, 0.065, 0.09, 0.12, 0.16, 0.18, 0.19, 0.195, 0.2, 0.202, 0.203, 0.204], dtype=float)
    out = detect_knee(
        eta,
        L,
        method="segmented",
        rho=1.5,
        min_points=6,
        bootstrap_n=128,
        bic_delta_min=1.0,
        edge_buffer_frac=0.05,
    )
    assert out.has_knee
    assert out.L_k is not None
    assert out.knee_window[0] is not None and out.knee_window[1] is not None
    assert out.knee_window[0] <= out.L_k <= out.knee_window[1]
    assert out.forbidden_band[0] is not None and out.forbidden_band[1] is not None
    assert out.forbidden_band[0] < out.forbidden_band[1]


def test_knee_model_selection_rejects_clean_power_law():
    L = np.geomspace(8, 200, 18)
    eta = 0.03 * (L / 20.0) ** 1.2
    out = detect_knee(
        eta,
        L,
        method="segmented",
        rho=1.5,
        min_points=6,
        bic_delta_min=10.0,
        bootstrap_n=64,
    )
    assert not out.has_knee
    assert out.L_k is None
    assert out.confidence == 0.0


def test_knee_unknown_method_raises():
    L = np.arange(10, dtype=float) + 1.0
    eta = np.linspace(0.1, 0.9, 10)
    try:
        detect_knee(eta, L, method="bad")
        assert False
    except ValueError as exc:
        assert "Unsupported" in str(exc)
