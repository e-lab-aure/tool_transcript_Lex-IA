@echo off
setlocal

echo ============================================
echo  Whisper Transcriber - Installation
echo  CUDA 12.8 / PyTorch 2.8 / NeMo trunk
echo ============================================
echo.

REM -- Verifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python introuvable dans le PATH.
    echo Installe Python 3.11 ou 3.12 depuis https://www.python.org
    pause
    exit /b 1
)

echo [1/5] Creation du virtualenv...
python -m venv .venv
if errorlevel 1 (
    echo [ERREUR] Creation du virtualenv echouee.
    pause
    exit /b 1
)

echo [2/5] Mise a jour de pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [3/5] Installation de PyTorch 2.8 cu128...
.venv\Scripts\pip.exe install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [ERREUR] Installation de PyTorch echouee.
    pause
    exit /b 1
)

echo [4/5] Installation des dependances de base...
.venv\Scripts\pip.exe install flask>=3.0.0 faster-whisper>=1.0.0 yt-dlp
if errorlevel 1 (
    echo [ERREUR] Installation des dependances echouee.
    pause
    exit /b 1
)

echo [5/5] Installation de NeMo trunk (Canary-Qwen-2.5B)...
echo      ^(peut prendre 10-15 min, ~5 Go^)
.venv\Scripts\pip.exe install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
if errorlevel 1 (
    echo [ERREUR] Installation de NeMo echouee.
    echo Verifie que git est installe et dans ton PATH.
    pause
    exit /b 1
)

echo.
echo [TEST] Verification CUDA...
.venv\Scripts\python.exe -c "import torch; print('  PyTorch :', torch.__version__); print('  CUDA dispo :', torch.cuda.is_available()); print('  GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NON DETECTE')"

echo.
echo ============================================
echo  Installation terminee !
echo  Lance run.bat pour demarrer le serveur.
echo ============================================
pause
