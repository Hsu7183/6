@echo off
cd /d "%~dp0"

set "REPORT=report_outputs\four_layer_matrix\four_layer_threshold_report.html"

if exist "%REPORT%" (
    start "" "%CD%\%REPORT%"
    exit /b 0
)

echo Cannot find report:
echo %CD%\%REPORT%
echo.
echo Please run the backtest first.
pause
