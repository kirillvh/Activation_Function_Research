@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.plot_activations --activations peuaf sine_triangle gelu gelu_sine_triangle --w 1.0 --blend 0.5 %*
if errorlevel 1 (
    echo.
    echo Activation plotting failed. See the error above.
) else (
    echo.
    echo Activation plots completed.
)
pause
