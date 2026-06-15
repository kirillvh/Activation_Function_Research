@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.audio_benchmark --config configs\benchmark_audio_activations.yaml %*
if errorlevel 1 (
    echo.
    echo Audio activation benchmark failed. See the error above.
) else (
    echo.
    echo Audio activation benchmark completed.
)
pause
