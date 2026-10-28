@echo off
rem Starts the Spatial Measure app on Windows (double-click, or run from a terminal).
rem The first run creates .venv in this folder and installs requirements.txt; later runs start straight away.
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Creating the Python environment in .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist "%PY%" (
        echo Python 3.10 or newer is needed: https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
        pause
        exit /b 1
    )
)
"%PY%" -c "import cv2, PIL, ultralytics" 2>nul || (
    echo Installing the requirements, this takes a few minutes the first time ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt || (pause & exit /b 1)
)
"%PY%" app.py %*
if errorlevel 1 pause
