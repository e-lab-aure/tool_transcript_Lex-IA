@echo off
if not exist .venv (
    echo ERREUR : Le virtualenv n'existe pas. Lance install.bat d'abord.
    pause
    exit /b 1
)

echo Demarrage du serveur sur http://localhost:5000
echo Appuie sur Ctrl+C pour arreter.
echo.
.venv\Scripts\python.exe app.py
pause
