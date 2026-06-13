@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.benchmark --config configs\benchmark_activations.yaml %*
if errorlevel 1 (
    echo.
    echo Benchmark failed. See the error above.
) else (
    echo.
    echo Benchmark completed.
)
pause
