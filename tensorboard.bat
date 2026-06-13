@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
echo TensorBoard will be available at http://localhost:6006
"%PYTHON%" -m tensorboard.main --logdir runs --port 6006
pause
