@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERREUR] Lance install.bat d'abord.
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade "yt-dlp[default,deno]" || exit /b 1
".venv\Scripts\python.exe" -m pip check || exit /b 1
".venv\Scripts\python.exe" -m yt_dlp --version
