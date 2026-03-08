from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .embedding import ESMEmbedder
from .features import build_handcrafted_matrix


def predict_proba(df: pd.DataFrame, model_dir: str, seq_col: str = "sequence", device: str = "cpu") -> np.ndarray:
    model_dir = Path(model_dir)
    meta = joblib.load(model_dir / "stacking_model.joblib")

    feature_order = ["handcrafted_lgbm", "esm_lr"]
    cfg_path = model_dir / "stack_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        feature_order = cfg.get("meta_feature_order", feature_order)

    X_h = build_handcrafted_matrix(df, seq_col=seq_col)
    embedder = ESMEmbedder(device=device)
    X_e = embedder.encode(df[seq_col].astype(str).tolist())

    branch_probs = {}
    if (model_dir / "branch_handcrafted_lgbm.joblib").exists():
        lgb = joblib.load(model_dir / "branch_handcrafted_lgbm.joblib")
        branch_probs["handcrafted_lgbm"] = lgb.predict_proba(X_h)[:, 1]

    if (model_dir / "branch_esm_lr.joblib").exists():
        esm_lr = joblib.load(model_dir / "branch_esm_lr.joblib")
        branch_probs["esm_lr"] = esm_lr.predict_proba(X_e)[:, 1]

    if (model_dir / "branch_esm_dnn.joblib").exists():
        esm_dnn = joblib.load(model_dir / "branch_esm_dnn.joblib")
        branch_probs["esm_dnn"] = esm_dnn.predict_proba(X_e)[:, 1]

    missing = [name for name in feature_order if name not in branch_probs]
    if missing:
        raise FileNotFoundError(f"Missing branch outputs required by stacking model: {missing}")

    X_meta = np.vstack([branch_probs[name] for name in feature_order]).T
    pm = meta.predict_proba(X_meta)[:, 1]
    return pm
