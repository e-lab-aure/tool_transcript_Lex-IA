@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo [ERREUR] Lance install.bat d'abord.
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" tray.py
