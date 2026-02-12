from gka.ops.stability import classify_stability


def test_stability_classification_in_gamma_space():
    assert classify_stability(gamma=1.0, drift=0.3, b=2.0, marginal_eps=0.1).stability_class == "stable"
    assert classify_stability(gamma=0.02, drift=0.3, b=2.0, marginal_eps=0.1).stability_class == "marginal"
    assert classify_stability(gamma=-0.3, drift=0.0, b=2.0, marginal_eps=0.1).stability_class == "forbidden"
