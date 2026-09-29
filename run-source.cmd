@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment. Python 3.12 x64 is recommended.
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.12 x64 from python.org, then run this file again.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -c "import numpy, cv2, PIL, rawpy, PySide6, tifffile, onnxruntime" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
