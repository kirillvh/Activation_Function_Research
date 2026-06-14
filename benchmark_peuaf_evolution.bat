@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.peuaf_search_benchmark --config configs\benchmark_peuaf_evolution_confirmation.yaml %*
if errorlevel 1 (
    echo.
    echo PEUAF evolution benchmark failed. See the error above.
) else (
    echo.
    echo PEUAF evolution benchmark completed.
)
pause
