@echo off
setlocal
cd /d "%~dp0"

where ffmpeg.exe >nul 2>&1 || (
    echo [ERREUR] FFmpeg est introuvable dans le PATH.
    exit /b 1
)
where git.exe >nul 2>&1 || (
    echo [ERREUR] Git est introuvable dans le PATH.
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv || (
        echo [ERREUR] Installe Python 3.12 x64 avec le lanceur py.exe.
        exit /b 1
    )
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -c "import struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 else 1)" || (
    echo [ERREUR] Le virtualenv doit utiliser Python 3.12 x64. Renomme .venv avant de relancer.
    exit /b 1
)

echo [1/3] Installation de PyTorch CUDA 12.8...
"%PY%" -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" || goto :failed

echo [2/3] Installation des dependances...
"%PY%" -m pip install -r requirements.txt -c constraints-windows.txt || goto :failed

echo [3/3] Verification...
"%PY%" -m pip check || goto :failed
"%PY%" -c "import app, torch, waitress; assert torch.cuda.is_available(), 'CUDA non disponible'; print(torch.cuda.get_device_name(0))" || goto :failed

echo Installation terminee. Lance tray.bat ou run.bat.
exit /b 0

:failed
echo [ERREUR] Installation ou verification incomplete. Consulte la sortie ci-dessus.
exit /b 1
