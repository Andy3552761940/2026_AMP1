from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from hemo_pred.infer import predict_proba

from .celery_app import celery_app


MODEL_DIR = os.getenv("MODEL_DIR", "outputs/exp1")
SEQ_COL = os.getenv("SEQ_COL", "sequence")
PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.5"))


@celery_app.task(name="hemo.predict_from_csv")
def predict_from_csv(input_csv: str, output_csv: str, seq_col: str = SEQ_COL) -> dict:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if seq_col not in df.columns:
        raise ValueError(f"CSV 缺少序列列: {seq_col}")

    proba = predict_proba(df=df, model_dir=MODEL_DIR, seq_col=seq_col, device="cpu")
    result_df = df.copy()
    result_df["hemolysis_probability"] = proba
    result_df["hemolysis_label"] = (proba >= PRED_THRESHOLD).astype(int)
    result_df.to_csv(output_path, index=False)

    return {
        "rows": int(len(result_df)),
        "output_csv": str(output_path),
        "threshold": PRED_THRESHOLD,
    }
