param([string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $PythonPath) { $PythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
$pythonPath = $PythonPath
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv with Python 3.12 and install the app dependencies first.'
}
& $pythonPath -c "import onnxruntime as ort; assert 'DmlExecutionProvider' in ort.get_available_providers(), 'DirectML provider missing'; import PySide6, rawpy, cv2, PyInstaller"
if ($LASTEXITCODE -ne 0) {
    throw 'Install requirements.txt, uninstall onnxruntime, then install requirements-directml.txt and pyinstaller.'
}
& $pythonPath tools/check_gpu.py
if ($LASTEXITCODE -ne 0) { throw 'DirectML GPU inference did not pass its hardware check.' }
& $pythonPath -m PyInstaller --noconfirm LumenARW.spec
if ($LASTEXITCODE -ne 0) { throw 'DirectML portable build failed.' }
Write-Output 'Local DirectML portable build: dist\LumenARW\LumenARW.exe'
