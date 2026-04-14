# Hemolysis-Pred：本地上传 CSV + 远端异步预测 + 结果回传（FastAPI + Celery + PySide6）

本项目支持你提出的完整流程：
1. 用户在**本地桌面客户端（PySide6）**上传肽序列 CSV；
2. 文件通过网络发送到**远端 FastAPI 服务**；
3. 服务端将任务投递给 **Celery Worker** 异步执行预测；
4. 预测完成后，客户端轮询任务状态并下载结果 CSV。

---

## 一、项目结构（按你要求拆分）

```text
2026_AMP1/
├── server/
│   └── app/
│       ├── __init__.py
│       ├── main.py           # FastAPI 接口代码
│       ├── celery_app.py     # Celery 实例与配置
│       └── tasks.py          # Celery 异步任务（读 CSV、跑模型、写结果）
├── client/
│   └── pyside6_client.py     # PySide6 本地客户端
├── docker/
│   ├── Dockerfile.api        # API 服务镜像
│   ├── Dockerfile.worker     # Celery Worker 镜像
│   └── docker-compose.yml    # Redis + API + Worker 编排
├── src/hemo_pred/            # 现有预测核心模块（被 server 调用）
├── outputs/exp1/             # 训练好的模型目录（运行时挂载）
├── requirements.txt
└── README.md
```

---

## 二、服务端说明

### 1) FastAPI 接口（`server/app/main.py`）
提供 4 个接口：

- `GET /health`
  - 健康检查。

- `POST /v1/predict`
  - 入参：`multipart/form-data` 上传 `file`（CSV）。
  - 行为：保存上传文件、创建 `job_id`、投递 Celery 任务。
  - 出参：`{job_id, task_id}`。

- `GET /v1/tasks/{task_id}`
  - 查询 Celery 任务状态（`PENDING/STARTED/SUCCESS/FAILURE`）。

- `GET /v1/results/{job_id}`
  - 任务成功后下载结果 CSV。

### 2) Celery 异步任务（`server/app/tasks.py`）
任务名：`hemo.predict_from_csv`

执行步骤：
1. 读取上传的 CSV；
2. 检查序列列（默认 `sequence`）；
3. 调用 `hemo_pred.infer.predict_proba(...)` 进行预测；
4. 写回结果列：
   - `hemolysis_probability`
   - `hemolysis_label`（默认阈值 `0.5`）
5. 导出结果 CSV 到共享目录，供 API 下载。

### 3) Celery 配置（`server/app/celery_app.py`）
- 默认 Broker/Backend：`redis://redis:6379/0`
- JSON 序列化
- UTC 时区

---

## 三、本地客户端说明（PySide6）

客户端文件：`client/pyside6_client.py`

功能：
- 输入服务端地址（默认 `http://127.0.0.1:8000`）
- 选择本地 CSV
- 上传后自动轮询任务状态（每 3 秒）
- 成功后弹出“另存为”下载预测结果 CSV

---

## 四、Docker 部署文件说明

### `docker/Dockerfile.api`
- 构建 FastAPI 服务镜像
- 启动命令：`uvicorn server.app.main:app --host 0.0.0.0 --port 8000`

### `docker/Dockerfile.worker`
- 构建 Celery Worker 镜像
- 启动命令：`celery -A server.app.celery_app:celery_app worker --loglevel=INFO`

### `docker/docker-compose.yml`
一次启动 3 个服务：
- `redis`
- `api`
- `worker`

并挂载：
- `../outputs/exp1` → `/models/exp1`（只读，模型目录）
- 共享卷 `runtime_data`（上传文件和结果文件共享）

---

## 五、完整运行流程（详细）

> 以下流程按“先训练模型，再部署远端推理服务，再启动本地客户端”执行。

### Step 0. 准备数据格式
CSV 至少包含一列：
- `sequence`：肽序列字符串

示例：

```csv
sequence
KWKLFKKIGAVLKVL
GIGKFLHSAKKFGKAFVGEIMNS
```

---

### Step 1. 训练并导出模型（一次性）
如果你还没有 `outputs/exp1`，先训练：

```bash
python scripts/train.py \
  --train_csv data/processed/train.csv \
  --out_dir outputs/exp1
```

训练后确保以下文件存在（至少）：
- `outputs/exp1/stacking_model.joblib`
- `outputs/exp1/branch_handcrafted_lgbm.joblib`
- `outputs/exp1/branch_esm_lr.joblib`

---

### Step 2. 启动远端服务（Docker）
在项目根目录执行：

```bash
docker compose -f docker/docker-compose.yml up --build
```

成功后：
- API 地址：`http://<服务器IP>:8000`
- 健康检查：`GET /health`

---

### Step 3. 本地启动 PySide6 客户端
在本地机器执行：

```bash
python client/pyside6_client.py
```

客户端操作顺序：
1. 在“服务端地址”输入：`http://<服务器IP>:8000`
2. 点击“选择CSV”
3. 点击“上传并预测”
4. 等待状态显示 `SUCCESS`
5. 保存下载的结果 CSV

---

### Step 4. 接口级联调（可选，用 curl）

#### 4.1 提交预测任务
```bash
curl -X POST "http://127.0.0.1:8000/v1/predict" \
  -F "file=@data/processed/test.csv"
```
返回示例：
```json
{"job_id":"...","task_id":"..."}
```

#### 4.2 查询状态
```bash
curl "http://127.0.0.1:8000/v1/tasks/<task_id>"
```

#### 4.3 下载结果
```bash
curl -L "http://127.0.0.1:8000/v1/results/<job_id>" -o result.csv
```

---

## 六、关键环境变量

- `REDIS_URL`：Redis 地址（默认 `redis://redis:6379/0`）
- `MODEL_DIR`：模型目录（默认 `outputs/exp1`，容器中建议 `/models/exp1`）
- `UPLOAD_DIR`：上传临时目录（默认 `runtime/uploads`）
- `RESULT_DIR`：结果目录（默认 `runtime/results`）
- `SEQ_COL`：序列列名（默认 `sequence`）
- `PRED_THRESHOLD`：分类阈值（默认 `0.5`）

---

## 七、常见问题

1. **任务一直 PENDING**
   - 检查 worker 是否启动；
   - 检查 `REDIS_URL` 在 api 与 worker 是否一致。

2. **下载结果 404**
   - 任务未完成；
   - 或 api 与 worker 没有共享同一个结果目录/卷。

3. **提示缺少模型文件**
   - 检查 `MODEL_DIR` 是否包含 `stacking_model.joblib` 及分支模型文件。

4. **CSV 列名不是 sequence**
   - 设置环境变量 `SEQ_COL=<你的列名>`。

---

## 八、开发环境安装

```bash
pip install -r requirements.txt
```

包含：模型推理依赖 + FastAPI/Celery/Redis 客户端 + PySide6。
