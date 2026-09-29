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

cd /d "%ROOT%cnc_frontend" || (
    echo ERROR: Dashboard folder not found: "%ROOT%cnc_frontend"
    pause
    exit /b 1
)

"%PYTHON%" -m uvicorn web.app:app --host 0.0.0.0 --port 8000
echo Dashboard stopped.
pause
