@echo off
setlocal

REM Build GUI executable for Windows.
REM Usage:
REM   scripts\build_exe.bat
REM   scripts\build_exe.bat --name HemoGUI --onefile 1

python scripts\build_exe.py %*
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo Build success.
endlocal
