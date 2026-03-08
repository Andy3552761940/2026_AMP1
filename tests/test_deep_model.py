import numpy as np

from hemo_pred.deep_model import ESMDeepClassifier


def test_deep_model_fit_predict_proba():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)

    model = ESMDeepClassifier(
        hidden_dim=64,
        batch_size=16,
        max_epochs=3,
        patience=2,
        device="cpu",
        random_state=7,
    )
    model.fit(x, y)
    probs = model.predict_proba(x)

    assert probs.shape == (64, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
