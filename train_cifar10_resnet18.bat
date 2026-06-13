@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.train --config configs\cifar10_resnet18_research.yaml %*
if errorlevel 1 (
    echo.
    echo CIFAR-10 ResNet-18 research training failed. See the error above.
) else (
    echo.
    echo CIFAR-10 ResNet-18 research training completed.
)
pause
