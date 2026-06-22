@echo off
cd /d "%~dp0"

set "REPORT=index.html"

if exist "%REPORT%" (
    start "" "%CD%\%REPORT%"
    exit /b 0
)

echo Cannot find report:
echo %CD%\%REPORT%
echo.
echo Please run the backtest first.
pause
