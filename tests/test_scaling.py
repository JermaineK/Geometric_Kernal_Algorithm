import numpy as np

from gka.ops.scaling import fit_scaling


def test_scaling_wls_basic():
    L = np.array([8, 12, 18, 27, 40, 60], dtype=float)
    eta = 0.02 * L ** 0.8
    out = fit_scaling(
        eta=eta,
        L=L,
        exclude_band=None,
        weights=None,
        method="wls",
        min_sizes=4,
        bootstrap_n=200,
        rng=np.random.default_rng(0),
    )
    assert 0.5 < out.gamma < 1.1
    assert out.Delta_hat == 2 - out.gamma
    assert out.ci[0] < out.ci[1]


def test_scaling_unknown_method_raises():
    L = np.array([1, 2, 3, 4, 5], dtype=float)
    eta = np.array([1, 2, 3, 4, 5], dtype=float)
    try:
        fit_scaling(eta, L, None, None, method="bad", min_sizes=4)
        assert False
    except ValueError as exc:
        assert "Unsupported" in str(exc)
