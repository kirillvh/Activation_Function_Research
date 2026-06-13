@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.multistart --config configs\peuaf_cifar10_successive_halving.yaml %*
if errorlevel 1 (
    echo.
    echo PEUAF multi-start training failed. See the error above.
) else (
    echo.
    echo PEUAF multi-start training completed.
)
pause
