@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERREUR] Lance install.bat d'abord.
    exit /b 1
)
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
echo Ouverture sur http://localhost:5000 apres le chargement de Whisper.
".venv\Scripts\python.exe" app.py
exit /b %errorlevel%
