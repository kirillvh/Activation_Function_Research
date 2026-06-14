@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.benchmark --config configs\benchmark_periodic_gelu_cifar10_long.yaml %*
if errorlevel 1 (
    echo.
    echo Long periodic GELU benchmark failed. See the error above.
) else (
    echo.
    echo Long periodic GELU benchmark completed.
)
pause
