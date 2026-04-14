from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from celery.result import AsyncResult
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .celery_app import celery_app
from .tasks import predict_from_csv


UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "runtime/uploads"))
RESULT_DIR = Path(os.getenv("RESULT_DIR", "runtime/results"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Hemolysis Remote Predictor", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/predict")
async def submit_csv(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="请上传 CSV 文件")

    job_id = str(uuid.uuid4())
    input_csv = UPLOAD_DIR / f"{job_id}.csv"
    output_csv = RESULT_DIR / f"{job_id}_result.csv"

    with input_csv.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = predict_from_csv.delay(str(input_csv), str(output_csv))
    return {"job_id": job_id, "task_id": task.id}


@app.get("/v1/tasks/{task_id}")
def task_status(task_id: str) -> dict:
    task = AsyncResult(task_id, app=celery_app)
    payload = {
        "task_id": task_id,
        "state": task.state,
    }

    if task.state == "FAILURE":
        payload["error"] = str(task.result)
    elif task.state == "SUCCESS":
        payload["result"] = task.result

    return payload


@app.get("/v1/results/{job_id}")
def get_result(job_id: str):
    path = RESULT_DIR / f"{job_id}_result.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="结果不存在或任务尚未完成")
    return FileResponse(path=path, filename=path.name, media_type="text/csv")
