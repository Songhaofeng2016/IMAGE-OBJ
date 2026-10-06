@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install Python 3 and try again.
  pause
  exit /b 1
)
python server.py
pause
