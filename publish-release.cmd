@echo off
setlocal
cd /d "%~dp0"
echo Publish LUMEN RAW to GitHub: commit + tag + push, then create the Release with the packages.
echo If you are not signed in, a one-time code appears below; open https://github.com/login/device and enter it.
echo Log: .publish\v131\logs\publish.log
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\publish_release.ps1" %*
set CODE=%ERRORLEVEL%
echo.
if %CODE%==0 (echo Published successfully.) else (echo Publish FAILED - see .publish\v131\logs\publish.log)
pause
exit /b %CODE%
