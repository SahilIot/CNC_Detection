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

cd /d "%ROOT%cnc_backend" || (
    echo ERROR: Backend folder not found: "%ROOT%cnc_backend"
    pause
    exit /b 1
)

"%PYTHON%" -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000
echo Backend stopped.
pause
