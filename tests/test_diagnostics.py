import numpy as np

from gka.ops.diagnostics import (
    alignment_residual,
    band_hit_rate,
    compute_stability_diagnostics,
    predict_bands,
)


def test_predict_bands_and_hit_rate():
    pred = predict_bands(L=10.0, c_m=2.0, n_max=3)
    obs = pred * np.array([1.01, 0.99, 1.04])
    hit = band_hit_rate(observed_peaks=obs, predicted_peaks=pred, rel_tol=0.05)
    assert hit == 1.0


def test_stability_diagnostics_outputs():
    diag = compute_stability_diagnostics(
        gamma=1.0,
        b=2.0,
        mu_k_hat=3.0,
        tau_s_hat=0.2,
        eps_log=0.15,
        L_values=np.geomspace(5, 100, 20),
        c_m_hat=2.0,
    )
    assert diag.S_at_mu_k is not None
    assert diag.W_mu is not None
    assert diag.S_curve_mu.size > 0
    assert diag.band_class_hat in {"coherent", "forbidden_middle", "incoherent"}


def test_alignment_residual_none_on_missing():
    r, mz = alignment_residual(None, 2.0, 1.0)
    assert r is None
    assert mz is None
