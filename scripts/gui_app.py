#!/usr/bin/env python3
from __future__ import annotations


import os

import threading
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class HemoPredictorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hemolysis Predictor Pro")
        self.root.geometry("1200x760")
        self.root.minsize(980, 620)

        self.api_url = tk.StringVar(value="http://127.0.0.1:8000/predict")
        self.model_dir = tk.StringVar()
        self.input_csv = tk.StringVar()
        self.output_csv = tk.StringVar()
        self.seq_col = tk.StringVar(value="sequence")
        self.device = tk.StringVar(value="cpu")
        self.threshold = tk.DoubleVar(value=0.5)

        self.status_var = tk.StringVar(value="请选择CSV并配置云端API后开始预测")

        self.status_var = tk.StringVar(value="请先点“测试API连接”，通过后再开始预测")


        self.df: pd.DataFrame | None = None
        self.pred_df: pd.DataFrame | None = None

        self._setup_style()
        self._build_layout()

    def _setup_style(self) -> None:
        self.root.configure(bg="#0f172a")
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("Card.TFrame", background="#111827")
        style.configure("Title.TLabel", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 20, "bold"))
        style.configure("Hint.TLabel", background="#0f172a", foreground="#94a3b8", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#111827", foreground="#e2e8f0", font=("Segoe UI", 11, "bold"))
        style.configure("Body.TLabel", background="#111827", foreground="#cbd5e1", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=7)
        style.map("Accent.TButton", background=[("active", "#2563eb"), ("!disabled", "#1d4ed8")], foreground=[("!disabled", "#ffffff")])

        style.configure("Treeview", background="#0b1220", foreground="#dbeafe", fieldbackground="#0b1220", rowheight=24)
        style.configure("Treeview.Heading", background="#1e293b", foreground="#e2e8f0", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        outer.pack(fill="both", expand=True, padx=12, pady=12)

        header = ttk.Frame(outer, style="Card.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Hemolysis Predictor Pro", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="本地选CSV并上传到云端推理 → 下载结果CSV → 本地预览与导出",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 12))

        config = ttk.Frame(outer, style="Card.TFrame", padding=12)
        config.pack(fill="x", pady=(0, 8))

        self._entry_row(config, "云端API", self.api_url, 0)

        self._file_row(config, "服务器模型目录(可选)", self.model_dir, self.pick_model_dir, 1)
        self._file_row(config, "输入CSV", self.input_csv, self.pick_input_csv, 2)
        self._file_row(config, "输出CSV", self.output_csv, self.pick_output_csv, 3)

        options = ttk.Frame(config, style="Card.TFrame")
        options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 4))

        ttk.Label(config, text="示例: http://<服务器IP>:8000/predict", style="Hint.TLabel").grid(
            row=1, column=1, columnspan=2, sticky="w", padx=(0, 8)
        )
        self._entry_row(config, "服务器模型目录(可选)", self.model_dir, 2)
        self._file_row(config, "输入CSV", self.input_csv, self.pick_input_csv, 3)
        self._file_row(config, "输出CSV", self.output_csv, self.pick_output_csv, 4)

        options = ttk.Frame(config, style="Card.TFrame")
        options.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 4))

        options.columnconfigure(6, weight=1)

        ttk.Label(options, text="序列列名", style="Body.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.seq_combo = ttk.Combobox(options, textvariable=self.seq_col, values=["sequence"], state="readonly", width=20)
        self.seq_combo.grid(row=0, column=1, sticky="w", padx=(0, 18))

        ttk.Label(options, text="云端设备", style="Body.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Combobox(options, textvariable=self.device, values=["cpu", "cuda"], state="readonly", width=8).grid(
            row=0, column=3, sticky="w", padx=(0, 18)
        )

        ttk.Label(options, text="判定阈值", style="Body.TLabel").grid(row=0, column=4, sticky="w", padx=(0, 8))
        ttk.Scale(options, from_=0.1, to=0.9, variable=self.threshold, orient="horizontal", length=180).grid(
            row=0, column=5, sticky="w"
        )
        self.threshold_label = ttk.Label(options, text="0.50", style="Body.TLabel")
        self.threshold_label.grid(row=0, column=6, sticky="w", padx=(10, 0))
        self.threshold.trace_add("write", self._on_threshold_change)

        actions = ttk.Frame(outer, style="Card.TFrame")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Button(actions, text="加载CSV列信息", command=self.load_csv_columns).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="测试API连接", command=self.test_api_connection).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="开始预测", style="Accent.TButton", command=self.start_predict).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="导出当前结果", command=self.export_current).pack(side="left")

        status_frame = ttk.Frame(outer, style="Card.TFrame", padding=(0, 4))
        status_frame.pack(fill="x")
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 4))
        ttk.Label(status_frame, textvariable=self.status_var, style="Hint.TLabel").pack(anchor="w")

        metrics = ttk.Frame(outer, style="Card.TFrame", padding=8)
        metrics.pack(fill="x", pady=(6, 8))
        self.metric_total = ttk.Label(metrics, text="样本数: -", style="CardTitle.TLabel")
        self.metric_total.pack(side="left", padx=(0, 18))
        self.metric_pos = ttk.Label(metrics, text="预测溶血(1): -", style="CardTitle.TLabel")
        self.metric_pos.pack(side="left", padx=(0, 18))
        self.metric_neg = ttk.Label(metrics, text="预测非溶血(0): -", style="CardTitle.TLabel")
        self.metric_neg.pack(side="left")

        table_card = ttk.Frame(outer, style="Card.TFrame", padding=8)
        table_card.pack(fill="both", expand=True)
        ttk.Label(table_card, text="预测结果预览（前500条）", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 6))

        cols = ["idx", "sequence", "p_hemolysis", "pred_label"]
        self.tree = ttk.Treeview(table_card, columns=cols, show="headings")
        for c, w in [("idx", 70), ("sequence", 600), ("p_hemolysis", 140), ("pred_label", 120)]:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")

    def _entry_row(self, parent: ttk.Frame, title: str, var: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=title, style="Body.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var, width=100).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=4)
        parent.columnconfigure(1, weight=1)

    def _file_row(self, parent: ttk.Frame, title: str, var: tk.StringVar, command, row: int) -> None:
        ttk.Label(parent, text=title, style="Body.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var, width=100).grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=4)
        ttk.Button(parent, text="浏览", command=command).grid(row=row, column=2, pady=4)
        parent.columnconfigure(1, weight=1)

    def _on_threshold_change(self, *_args) -> None:
        self.threshold_label.configure(text=f"{self.threshold.get():.2f}")


    def pick_model_dir(self) -> None:
        p = filedialog.askdirectory(title="服务器上的模型目录（可选）")
        if p:
            self.model_dir.set(p)
<<<<<<< HEAD

=======
>>>>>>> c59664dc7ae3f32f4b24f5a0e877edfc7b819ff6
    def _health_url(self) -> str:
        api_url = self.api_url.get().strip()
        parsed = urlparse(api_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("API地址格式错误，请使用 http://host:port/predict")
        return f"{parsed.scheme}://{parsed.netloc}/health"

    def test_api_connection(self) -> None:
        try:
            health_url = self._health_url()
            resp = requests.get(health_url, timeout=10)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            self.status_var.set(f"API连接正常: {health_url}")
            messagebox.showinfo("连接成功", f"已连通 {health_url}")
        except Exception as exc:
            self.status_var.set("API连接失败，请检查地址/端口/防火墙")
            messagebox.showerror("连接失败", str(exc))
<<<<<<< HEAD

=======
>>>>>>> c59664dc7ae3f32f4b24f5a0e877edfc7b819ff6

    def pick_input_csv(self) -> None:
        p = filedialog.askopenfilename(title="选择待预测CSV", filetypes=[("CSV", "*.csv")])
        if p:
            self.input_csv.set(p)
            if not self.output_csv.get():
                self.output_csv.set(str(Path(p).with_name(Path(p).stem + "_predictions.csv")))

    def pick_output_csv(self) -> None:
        p = filedialog.asksaveasfilename(title="选择输出CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.output_csv.set(p)

    def load_csv_columns(self) -> None:
        if not self.input_csv.get():
            messagebox.showwarning("提示", "请先选择输入CSV")
            return
        df = pd.read_csv(self.input_csv.get(), nrows=5)
        cols = list(df.columns)
        self.seq_combo.configure(values=cols)
        if "sequence" in cols:
            self.seq_col.set("sequence")
        elif cols:
            self.seq_col.set(cols[0])
        self.status_var.set(f"已加载列名: {', '.join(cols)}")

    def start_predict(self) -> None:
        if not self.api_url.get().strip() or not self.input_csv.get() or not self.output_csv.get():
            messagebox.showerror("缺少参数", "请完整配置云端API、输入CSV和输出CSV")
            return
        self.progress.start(10)
        self.status_var.set("上传CSV到云端并等待推理...")
        threading.Thread(target=self._run_prediction, daemon=True).start()

    def _run_prediction(self) -> None:
        try:
            seq_col = self.seq_col.get().strip()
            with open(self.input_csv.get(), "rb") as infile:
                files = {
                    "file": (Path(self.input_csv.get()).name, infile, "text/csv"),
                }
                data = {
                    "seq_col": seq_col,
                    "thr": f"{self.threshold.get():.6f}",
                    "device": self.device.get(),
                }
                if self.model_dir.get().strip():
                    data["model_dir"] = self.model_dir.get().strip()

                resp = requests.post(self.api_url.get().strip(), files=files, data=data, timeout=3600)

            if resp.status_code >= 400:
                raise RuntimeError(f"云端推理失败({resp.status_code}): {resp.text}")

            with open(self.output_csv.get(), "wb") as out:
                out.write(resp.content)

            out_df = pd.read_csv(self.output_csv.get())
            if seq_col not in out_df.columns:
                raise ValueError(f"结果CSV中找不到序列列: {seq_col}")

            self.pred_df = out_df
            self.df = out_df
            self.root.after(0, self._on_predict_done)
        except Exception as e:
            self.root.after(0, lambda: self._on_predict_error(str(e)))

    def _on_predict_done(self) -> None:
        self.progress.stop()
        assert self.pred_df is not None
        self.refresh_table(self.pred_df)
        total = len(self.pred_df)
        pos = int((self.pred_df["pred_label"] == 1).sum()) if "pred_label" in self.pred_df.columns else 0
        neg = total - pos

        self.metric_total.configure(text=f"样本数: {total}")
        self.metric_pos.configure(text=f"预测溶血(1): {pos}")
        self.metric_neg.configure(text=f"预测非溶血(0): {neg}")
        self.status_var.set(f"预测完成，结果已保存到: {self.output_csv.get()}")
        messagebox.showinfo("完成", "云端预测完成，结果已下载并导出CSV。")

    def _on_predict_error(self, msg: str) -> None:
        self.progress.stop()
        self.status_var.set("预测失败，请检查API和参数配置")
        messagebox.showerror("预测失败", msg)

    def refresh_table(self, df: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        seq_col = self.seq_col.get()
        for idx, row in df.head(500).iterrows():
            seq_val = str(row.get(seq_col, ""))
            p_val = row.get("p_hemolysis", float("nan"))
            label_val = row.get("pred_label", "")
            p_text = f"{float(p_val):.4f}" if pd.notna(p_val) else ""
            self.tree.insert("", "end", values=(idx, seq_val[:120], p_text, label_val))

    def export_current(self) -> None:
        if self.pred_df is None:
            messagebox.showwarning("提示", "当前没有可导出的预测结果")
            return
        target = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="导出结果")
        if not target:
            return
        self.pred_df.to_csv(target, index=False)
        self.status_var.set(f"已导出: {target}")


def main() -> None:
    root = tk.Tk()
    HemoPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
