#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from hemo_pred.data import load_dataset
from hemo_pred.infer import predict_proba

app = FastAPI(title="Hemolysis Predictor API", version="1.0.0")


def _resolve_model_dir(model_dir: str | None) -> str:
    if model_dir and model_dir.strip():
        return model_dir.strip()
    env_model_dir = os.getenv("HEMO_MODEL_DIR", "").strip()
    if env_model_dir:
        return env_model_dir
    raise HTTPException(status_code=400, detail="model_dir 未提供，且环境变量 HEMO_MODEL_DIR 未配置")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    seq_col: str = Form("sequence"),
    thr: float = Form(0.5),
    device: str = Form("cpu"),
    model_dir: str | None = Form(None),
) -> Response:
    if not file.filename:
        raise HTTPException(status_code=400, detail="上传文件缺少文件名")

    selected_model_dir = _resolve_model_dir(model_dir)
    if not Path(selected_model_dir).exists():
        raise HTTPException(status_code=400, detail=f"模型目录不存在: {selected_model_dir}")

    with tempfile.TemporaryDirectory(prefix="hemo_job_") as td:
        workdir = Path(td)
        input_csv = workdir / "input.csv"
        output_csv = workdir / "output.csv"

        content = await file.read()
        input_csv.write_bytes(content)

        try:
            df = load_dataset(str(input_csv), seq_col=seq_col, label_col="__dummy__")
            probs = predict_proba(df, selected_model_dir, seq_col=seq_col, device=device)
            out = df.copy()
            out["p_hemolysis"] = probs
            out["pred_label"] = (probs >= thr).astype(int)
            out.to_csv(output_csv, index=False)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

        output_name = f"{Path(file.filename).stem}_predictions.csv"
        return Response(
            content=output_csv.read_bytes(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
