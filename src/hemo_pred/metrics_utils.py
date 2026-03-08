from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auroc": float(roc_auc_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "threshold": float(thr),
    }


def find_best_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    metric: str = "mcc",
    search_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    if search_grid is None:
        search_grid = np.linspace(0.05, 0.95, 181)

    if metric not in {"mcc", "f1", "accuracy"}:
        raise ValueError("metric must be one of {'mcc', 'f1', 'accuracy'}")

    best_thr, best_score = 0.5, -1e9
    for thr in search_grid:
        pred = (prob >= thr).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(y_true, pred)
        elif metric == "f1":
            score = f1_score(y_true, pred)
        else:
            score = accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, float(best_score)
