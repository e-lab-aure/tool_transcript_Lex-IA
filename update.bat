@echo off
echo Mise a jour de yt-dlp et deno...
.venv\Scripts\python.exe -m pip install --upgrade "yt-dlp[default,deno]"
echo.
echo yt-dlp et deno mis a jour.
pause
