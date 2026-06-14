@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.cifar_peuaf_benchmark --config configs\benchmark_peuaf_cifar10_confirmation.yaml %*
if errorlevel 1 (
    echo.
    echo CIFAR-10 PEUAF benchmark failed. See the error above.
) else (
    echo.
    echo CIFAR-10 PEUAF benchmark completed.
)
pause
