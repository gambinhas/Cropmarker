@echo off
setlocal

cd /d "%~dp0\.."

REM Create venv if missing (stored under webapp/.venv)
if not exist "webapp\.venv\Scripts\python.exe" (
  python -m venv "webapp\.venv"
)

"webapp\.venv\Scripts\python.exe" -m pip install -r "webapp\requirements.txt"
if errorlevel 1 (
  echo.
  echo Failed to install requirements.
  pause
  exit /b 1
)

REM Start server
set "CROPMARKER_SECRET_KEY=change-me"
echo.
echo Cropmarker Web will be available at: http://localhost:8000
echo.
"webapp\.venv\Scripts\python.exe" -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000

echo.
echo Server stopped (exit code: %ERRORLEVEL%).
pause
