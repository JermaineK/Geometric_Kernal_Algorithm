from gka.metrics.eigen_stability import estimate_eigen_stability


def test_eigen_stability_classification():
    stable = estimate_eigen_stability(gamma=0.8, b=2.0, margin_eps=0.05)
    marginal = estimate_eigen_stability(gamma=0.0, b=2.0, margin_eps=0.05)
    unstable = estimate_eigen_stability(gamma=-0.7, b=2.0, margin_eps=0.05)

    assert stable.eigen_band == "stable"
    assert marginal.eigen_band == "marginal"
    assert unstable.eigen_band == "unstable"
