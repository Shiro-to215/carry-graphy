@echo off
setlocal
rem Launcher for integrated_app.py (Calibration / Homography GUI)
chcp 65001 >nul
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYW=%VENV_DIR%\Scripts\pythonw.exe"

rem --- 1. Create the virtual environment on first run ---
if not exist "%VENV_PY%" (
    echo [setup] Creating virtual environment in %VENV_DIR% ...
    where python >nul 2>nul
    if errorlevel 1 (
        echo [error] Python was not found. Install Python 3.10+ and add it to PATH.
        goto :fail
    )
    python -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
    echo [setup] Installing dependencies ...
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :fail
)

rem --- 2. Verify dependencies (install if missing) ---
"%VENV_PY%" -c "import tkinter, cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo [setup] Missing packages detected. Installing requirements ...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :fail
    "%VENV_PY%" -c "import tkinter, cv2, numpy"
    if errorlevel 1 goto :fail
)

rem --- 3. Launch the GUI without a console window ---
if exist "%VENV_PYW%" (
    start "" "%VENV_PYW%" "%~dp0integrated_app.py"
) else (
    start "" "%VENV_PY%" "%~dp0integrated_app.py"
)
exit /b 0

:fail
echo.
echo [error] Failed to launch the GUI. See the messages above.
pause
exit /b 1
