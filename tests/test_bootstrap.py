import numpy as np

from gka.stats.bootstrap import block_bootstrap_series, bootstrap_slope_ci


def test_bootstrap_slope_ci_orders_bounds():
    x = np.linspace(1.0, 10.0, 30)
    y = 1.7 * x + 0.2
    lo, hi = bootstrap_slope_ci(x, y, method="wls", n=200, rng=np.random.default_rng(0))
    assert lo < hi
    assert lo < 1.7 < hi


def test_block_bootstrap_series_shape():
    arr = np.arange(40, dtype=float)
    out = block_bootstrap_series(arr, n=8, block_size=5, rng=np.random.default_rng(0))
    assert out.shape == (8, arr.size)
