@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Benchmark: every GPU (forced one at a time) vs CPU on the 1.3.0 pixel pipeline. About 3-6 minutes.
echo.
"%PY%" tools\bench_devices.py .publish\v131\logs\bench-devices.json
echo.
echo Result saved to .publish\v131\logs\bench-devices.json
pause
