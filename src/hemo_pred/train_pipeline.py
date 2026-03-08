from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embedding import ESMEmbedder
from .features import build_handcrafted_matrix
from .metrics_utils import compute_binary_metrics, find_best_threshold


def train_with_cv(
    df,
    seq_col: str,
    label_col: str,
    out_dir: str,
    folds: int = 5,
    repeats: int = 2,
    seed: int = 42,
    device: str = "cpu",
    threshold_metric: str = "mcc",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_h = build_handcrafted_matrix(df, seq_col=seq_col)
    y = df[label_col].values.astype(int)
    seqs = df[seq_col].astype(str).tolist()

    embedder = ESMEmbedder(device=device)
    X_e = embedder.encode(seqs)

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)

    oof_lgb = np.zeros(len(df), dtype=float)
    oof_esm = np.zeros(len(df), dtype=float)
    oof_counts = np.zeros(len(df), dtype=float)

    for tr, va in rskf.split(X_h, y):
        lgb = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
        )
        lgb.fit(X_h[tr], y[tr])
        oof_lgb[va] += lgb.predict_proba(X_h[va])[:, 1]

        esm_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.2)),
        ])
        esm_lr.fit(X_e[tr], y[tr])
        oof_esm[va] += esm_lr.predict_proba(X_e[va])[:, 1]
        oof_counts[va] += 1.0

    oof_counts[oof_counts == 0] = 1.0
    oof_lgb = oof_lgb / oof_counts
    oof_esm = oof_esm / oof_counts

    X_meta = np.vstack([oof_lgb, oof_esm]).T
    meta_base = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8)
    calibrator = CalibratedClassifierCV(estimator=meta_base, method="sigmoid", cv=5)
    calibrator.fit(X_meta, y)
    oof_stack = calibrator.predict_proba(X_meta)[:, 1]

    best_thr, best_thr_score = find_best_threshold(y, oof_stack, metric=threshold_metric)

    cv_metrics = {
        "handcrafted_lgbm": compute_binary_metrics(y, oof_lgb, thr=0.5),
        "esm_lr": compute_binary_metrics(y, oof_esm, thr=0.5),
        "stacking": compute_binary_metrics(y, oof_stack, thr=best_thr),
        "best_threshold": {"metric": threshold_metric, "threshold": best_thr, "score": best_thr_score},
    }

    final_lgb = LGBMClassifier(
        n_estimators=900,
        learning_rate=0.02,
        num_leaves=24,
        min_child_samples=20,
        reg_lambda=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=seed,
    )
    final_lgb.fit(X_h, y)

    final_esm_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.2)),
    ])
    final_esm_lr.fit(X_e, y)

    joblib.dump(final_lgb, out / "branch_handcrafted_lgbm.joblib")
    joblib.dump(final_esm_lr, out / "branch_esm_lr.joblib")
    final_meta = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8),
        method="sigmoid",
        cv=5,
    )
    final_meta.fit(np.vstack([final_lgb.predict_proba(X_h)[:, 1], final_esm_lr.predict_proba(X_e)[:, 1]]).T, y)

    joblib.dump(final_meta, out / "stacking_model.joblib")
    with open(out / "cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cv_metrics, f, indent=2, ensure_ascii=False)
    with open(out / "decision_config.json", "w", encoding="utf-8") as f:
        json.dump({"recommended_threshold": best_thr}, f, indent=2, ensure_ascii=False)

    return cv_metrics
