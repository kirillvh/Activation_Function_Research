@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.benchmark --config configs\benchmark_sine_triangle_cifar10.yaml %*
if errorlevel 1 (
    echo.
    echo Sine-triangle benchmark failed. See the error above.
) else (
    echo.
    echo Sine-triangle benchmark completed.
)
pause
