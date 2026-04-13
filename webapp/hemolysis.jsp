<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>溶血性预测系统</title>
    <style>
        :root {
            --bg: #f5f7fb;
            --card: #ffffff;
            --primary: #2952cc;
            --primary-hover: #1f3fa1;
            --danger: #bf1b1b;
            --text: #1d2433;
            --muted: #5f6b85;
            --border: #dbe2ef;
            --ok-bg: #edf8f1;
            --ok-text: #177245;
            --bad-bg: #fff1f1;
            --bad-text: #9d1c1c;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
        }

        .container {
            max-width: 980px;
            margin: 36px auto;
            padding: 0 20px;
        }

        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 6px 20px rgb(41 82 204 / 7%);
        }

        h1 {
            margin: 0 0 8px;
            font-size: 28px;
        }

        .subtitle {
            margin: 0 0 20px;
            color: var(--muted);
            line-height: 1.6;
        }

        .form-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            margin-bottom: 14px;
        }

        .tip {
            color: var(--muted);
            font-size: 14px;
            margin: 0 0 8px;
        }

        input[type="file"] {
            flex: 1;
            min-width: 260px;
            border: 1px dashed var(--border);
            border-radius: 8px;
            padding: 10px;
            background: #fbfcff;
        }

        button {
            border: none;
            border-radius: 8px;
            padding: 11px 18px;
            cursor: pointer;
            font-weight: 600;
            transition: 0.2s ease;
        }

        .primary {
            background: var(--primary);
            color: #fff;
        }

        .primary:hover {
            background: var(--primary-hover);
        }

        .secondary {
            background: #eef2ff;
            color: var(--primary);
        }

        .secondary:hover {
            background: #dce6ff;
        }

        .error {
            color: var(--danger);
            margin: 10px 0 0;
            min-height: 20px;
        }

        .hidden {
            display: none;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 18px;
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }

        thead {
            background: #f0f4ff;
        }

        th, td {
            border-bottom: 1px solid var(--border);
            padding: 10px 12px;
            text-align: left;
            vertical-align: top;
            font-size: 14px;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }

        .tag-ok {
            background: var(--ok-bg);
            color: var(--ok-text);
        }

        .tag-bad {
            background: var(--bad-bg);
            color: var(--bad-text);
        }

        .summary {
            margin-top: 14px;
            color: var(--muted);
        }
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>氨基酸序列溶血性预测</h1>
        <p class="subtitle">
            上传包含序列信息的 CSV 文件（建议包含列：<code>id</code>、<code>sequence</code>），
            页面会调用后端预测接口并展示每条序列是否具有溶血性。
        </p>

        <form id="predictForm" enctype="multipart/form-data">
            <p class="tip">支持格式：.csv；最大大小可在后端配置中限制。</p>
            <div class="form-row">
                <input id="csvFile" type="file" accept=".csv,text/csv" name="file" required />
                <button type="submit" class="primary" id="submitBtn">开始预测</button>
                <button type="button" class="secondary" id="resetBtn">清空结果</button>
            </div>
            <div class="error" id="errorMsg"></div>
        </form>

        <div id="resultPanel" class="hidden">
            <p class="summary" id="summaryText"></p>
            <table>
                <thead>
                <tr>
                    <th style="width: 70px;">序号</th>
                    <th style="width: 180px;">样本 ID</th>
                    <th>序列</th>
                    <th style="width: 140px;">预测结果</th>
                    <th style="width: 120px;">置信度</th>
                </tr>
                </thead>
                <tbody id="resultBody"></tbody>
            </table>
        </div>
    </div>
</div>

<script>
    const form = document.getElementById("predictForm");
    const fileInput = document.getElementById("csvFile");
    const errorMsg = document.getElementById("errorMsg");
    const resultPanel = document.getElementById("resultPanel");
    const resultBody = document.getElementById("resultBody");
    const summaryText = document.getElementById("summaryText");
    const submitBtn = document.getElementById("submitBtn");
    const resetBtn = document.getElementById("resetBtn");

    const endpoint = "<%= request.getContextPath() %>/api/hemolysis/predict";

    function setError(message) {
        errorMsg.textContent = message || "";
    }

    function escapeHtml(raw) {
        return String(raw)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function renderResults(rows) {
        if (!Array.isArray(rows) || rows.length === 0) {
            resultPanel.classList.add("hidden");
            resultBody.innerHTML = "";
            summaryText.textContent = "";
            return;
        }

        let hemolyticCount = 0;
        const html = rows.map((row, index) => {
            const predicted = String(row.prediction || "").toLowerCase();
            const hemolytic = predicted === "hemolytic" || predicted === "yes" || predicted === "1";
            if (hemolytic) {
                hemolyticCount += 1;
            }

            const confidence = row.confidence == null ? "-" : `${(Number(row.confidence) * 100).toFixed(2)}%`;

            return `
                <tr>
                    <td>${index + 1}</td>
                    <td>${escapeHtml(row.id || `sample_${index + 1}`)}</td>
                    <td><code>${escapeHtml(row.sequence || "")}</code></td>
                    <td>
                        <span class="tag ${hemolytic ? "tag-bad" : "tag-ok"}">
                            ${hemolytic ? "具有溶血性" : "不具有溶血性"}
                        </span>
                    </td>
                    <td>${escapeHtml(confidence)}</td>
                </tr>
            `;
        }).join("");

        resultBody.innerHTML = html;
        summaryText.textContent = `共 ${rows.length} 条序列；预测具有溶血性的序列 ${hemolyticCount} 条。`;
        resultPanel.classList.remove("hidden");
    }

    async function submitPrediction(event) {
        event.preventDefault();
        setError("");

        const file = fileInput.files[0];
        if (!file) {
            setError("请先选择 CSV 文件。");
            return;
        }

        if (!file.name.toLowerCase().endsWith(".csv")) {
            setError("仅支持 .csv 文件。");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        submitBtn.disabled = true;
        submitBtn.textContent = "预测中...";

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`请求失败（HTTP ${response.status}）。`);
            }

            const data = await response.json();
            renderResults(data.results || []);

            if (!Array.isArray(data.results)) {
                setError("返回格式不符合预期：缺少 results 数组。");
            }
        } catch (error) {
            renderResults([]);
            setError(error.message || "预测失败，请检查后端服务日志。");
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "开始预测";
        }
    }

    function resetPage() {
        form.reset();
        setError("");
        renderResults([]);
    }

    form.addEventListener("submit", submitPrediction);
    resetBtn.addEventListener("click", resetPage);
</script>
</body>
</html>
