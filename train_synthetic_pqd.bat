@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.train --config configs\synthetic_pqd.yaml %*
if errorlevel 1 (
    echo.
    echo Signal training failed. See the error above.
) else (
    echo.
    echo Signal training completed.
)
pause
