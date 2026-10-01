@echo off
setlocal
cd /d "%~dp0"
echo LUMEN RAW release build: tests, DirectML check, portable ZIP and installer.
echo Logs: .publish\v141\logs  (this takes about 15-25 minutes)
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_release.ps1" %*
set CODE=%ERRORLEVEL%
echo.
if %CODE%==0 (echo Build finished successfully.) else (echo Build FAILED - see .publish\v141\logs\build.log)
pause
exit /b %CODE%
