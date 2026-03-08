#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

sys.path.append(os.path.abspath("src"))

from hemo_pred.data import load_dataset
from hemo_pred.infer import predict_proba


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    df = load_dataset(args.test_csv, args.seq_col, args.label_col)
    y = df[args.label_col].values.astype(int)
    p = predict_proba(df, args.model_dir, seq_col=args.seq_col, device=args.device)
    thr = args.thr
    if thr is None:
        cfg_file = Path(args.model_dir) / "decision_config.json"
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                thr = float(json.load(f).get("recommended_threshold", 0.5))
        else:
            thr = 0.5

    pred = (p >= thr).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "threshold": float(thr),
    }
    print(metrics)


if __name__ == "__main__":
    main()
