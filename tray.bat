@echo off
if not exist .venv (
    echo ERREUR : Le virtualenv n'existe pas. Lance install.bat d'abord.
    pause
    exit /b 1
)

REM pythonw evite la console : le pilotage se fait par l'icone de notification.
start "" .venv\Scripts\pythonw.exe tray.py
