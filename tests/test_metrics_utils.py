import numpy as np

from hemo_pred.metrics_utils import compute_binary_metrics, find_best_threshold


def test_find_best_threshold_returns_valid_range():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])

    thr, score = find_best_threshold(y, p, metric="mcc")

    assert 0.05 <= thr <= 0.95
    assert -1.0 <= score <= 1.0


def test_compute_binary_metrics_contains_threshold():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.6, 0.7])

    metrics = compute_binary_metrics(y, p, thr=0.5)

    assert set(["accuracy", "auroc", "f1", "mcc", "threshold"]).issubset(metrics.keys())
    assert metrics["threshold"] == 0.5
