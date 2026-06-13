@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m activation_benchmark.download_datasets %*
if errorlevel 1 (
    echo.
    echo Dataset download failed. See the error above.
) else (
    echo.
    echo Dataset download and extraction completed.
)
pause
