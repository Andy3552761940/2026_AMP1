#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Hemolysis Predictor GUI Windows exe with PyInstaller")
    ap.add_argument("--name", default="HemolysisPredictorPro")
    ap.add_argument("--onefile", type=int, default=1, choices=[0, 1], help="1=single exe, 0=directory mode")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    gui_script = repo_root / "scripts" / "gui_app.py"
    dist_dir = repo_root / "dist"
    build_dir = repo_root / "build"

    if build_dir.exists():
        shutil.rmtree(build_dir)

    cmd = [
        "pyinstaller",
        str(gui_script),
        "--noconfirm",
        "--clean",
        "--name",
        args.name,
        "--paths",
        str(repo_root / "src"),
    ]
    if args.onefile == 1:
        cmd.append("--onefile")
    cmd.append("--windowed")

    run(cmd)

    print("\nBuild finished.")
    print(f"Executable output folder: {dist_dir}")


if __name__ == "__main__":
    main()
