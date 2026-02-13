import numpy as np

from gka.classify.blind_model import InvariantBlindClassifier


def test_blind_model_smoke():
    rng = np.random.default_rng(0)
    x0 = rng.normal(loc=-1.0, scale=0.3, size=(40, 4))
    x1 = rng.normal(loc=1.0, scale=0.3, size=(40, 4))
    X = np.vstack([x0, x1])
    y = np.array(["a"] * 40 + ["b"] * 40)

    model = InvariantBlindClassifier(random_state=3, max_iter=300)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    assert metrics.accuracy >= 0.9
