@echo off
setlocal
rem Launcher for integrated_app.py (Calibration / Homography GUI)
chcp 65001 >nul
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYW=%VENV_DIR%\Scripts\pythonw.exe"
set "STAMP_FILE=%VENV_DIR%\.requirements.stamp"

rem --- 1. Check and create the virtual environment ---
set "NEED_VENV_CREATE=0"
if not exist "%VENV_PY%" (
    set "NEED_VENV_CREATE=1"
) else (
    "%VENV_PY%" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo [warning] Existing virtual environment is broken or Python base path changed.
        echo [setup] Re-creating virtual environment ...
        rmdir /s /q "%VENV_DIR%" >nul 2>nul
        set "NEED_VENV_CREATE=1"
    )
)

if "%NEED_VENV_CREATE%"=="1" (
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
    copy /y "requirements.txt" "%STAMP_FILE%" >nul 2>nul
)

rem --- 2. Verify dependencies only when requirements changed or stamp is missing ---
fc /b "requirements.txt" "%STAMP_FILE%" >nul 2>nul
if errorlevel 1 (
    "%VENV_PY%" -c "import tkinter, cv2, numpy, customtkinter, tkinterdnd2" >nul 2>nul
    if errorlevel 1 (
        echo [setup] Missing packages detected. Installing requirements ...
        "%VENV_PY%" -m pip install -r requirements.txt
        if errorlevel 1 goto :fail
        "%VENV_PY%" -c "import tkinter, cv2, numpy, customtkinter, tkinterdnd2"
        if errorlevel 1 goto :fail
    )
    copy /y "requirements.txt" "%STAMP_FILE%" >nul 2>nul
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
