@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.benchmark --config configs\benchmark_cifar10_activations.yaml %*
if errorlevel 1 (
    echo.
    echo CIFAR-10 benchmark failed. See the error above.
) else (
    echo.
    echo CIFAR-10 benchmark completed.
)
pause

