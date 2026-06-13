@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.train --config configs\cifar10.yaml %*
if errorlevel 1 (
    echo.
    echo CIFAR-10 training failed. See the error above.
) else (
    echo.
    echo CIFAR-10 training completed.
)
pause

