from __future__ import annotations

import sys
from pathlib import Path

import requests
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hemolysis Remote Client")

        self.server_input = QLineEdit("http://127.0.0.1:8000")
        self.csv_input = QLineEdit()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.select_btn = QPushButton("选择CSV")
        self.upload_btn = QPushButton("上传并预测")

        self.select_btn.clicked.connect(self.select_file)
        self.upload_btn.clicked.connect(self.submit_job)

        self.task_id: str | None = None
        self.job_id: str | None = None

        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.poll_status)

        root = QWidget()
        layout = QVBoxLayout(root)

        layout.addWidget(QLabel("服务端地址"))
        layout.addWidget(self.server_input)

        row = QHBoxLayout()
        row.addWidget(self.csv_input)
        row.addWidget(self.select_btn)
        layout.addLayout(row)

        layout.addWidget(self.upload_btn)
        layout.addWidget(self.log_view)
        self.setCentralWidget(root)

    def log(self, msg: str) -> None:
        self.log_view.append(msg)

    def select_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择CSV", str(Path.cwd()), "CSV Files (*.csv)")
        if filename:
            self.csv_input.setText(filename)

    def submit_job(self) -> None:
        csv_path = Path(self.csv_input.text().strip())
        if not csv_path.exists():
            QMessageBox.warning(self, "错误", "请选择有效 CSV 文件")
            return

        server = self.server_input.text().strip().rstrip("/")
        try:
            with csv_path.open("rb") as f:
                resp = requests.post(f"{server}/v1/predict", files={"file": (csv_path.name, f, "text/csv")}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self.task_id = data["task_id"]
            self.job_id = data["job_id"]
            self.log(f"任务已提交: task_id={self.task_id}, job_id={self.job_id}")
            self.timer.start()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "上传失败", str(exc))

    def poll_status(self) -> None:
        if not self.task_id:
            return

        server = self.server_input.text().strip().rstrip("/")
        try:
            resp = requests.get(f"{server}/v1/tasks/{self.task_id}", timeout=20)
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state", "UNKNOWN")
            self.log(f"任务状态: {state}")

            if state == "SUCCESS":
                self.timer.stop()
                self.download_result()
            elif state == "FAILURE":
                self.timer.stop()
                QMessageBox.critical(self, "任务失败", data.get("error", "未知错误"))
        except Exception as exc:  # noqa: BLE001
            self.log(f"轮询失败: {exc}")

    def download_result(self) -> None:
        if not self.job_id:
            return
        server = self.server_input.text().strip().rstrip("/")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存预测结果",
            str(Path.cwd() / f"{self.job_id}_result.csv"),
            "CSV Files (*.csv)",
        )
        if not save_path:
            self.log("用户取消下载")
            return

        try:
            resp = requests.get(f"{server}/v1/results/{self.job_id}", timeout=60)
            resp.raise_for_status()
            Path(save_path).write_bytes(resp.content)
            QMessageBox.information(self, "完成", f"结果已保存: {save_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "下载失败", str(exc))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(780, 520)
    w.show()
    sys.exit(app.exec())
