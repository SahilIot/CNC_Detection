@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Python virtual environment not found:
    echo "%PYTHON%"
    echo Run the setup commands in README.md first.
    pause
    exit /b 1
)

start "CNC Backend" /min "%ComSpec%" /k ""%ROOT%start_backend.bat""
timeout /t 2 /nobreak >nul
start "CNC Dashboard" /min "%ComSpec%" /k ""%ROOT%start_dashboard.bat""
