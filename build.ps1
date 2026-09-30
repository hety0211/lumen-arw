$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Run run-source.cmd first to create .venv, or build with your own Python environment.'
}
& $pythonPath -m pip install --no-cache-dir -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $pythonPath -m PyInstaller --noconfirm LumenRAW.spec
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
Write-Output 'CPU portable build: dist\LumenRAW\LumenRAW.exe'
