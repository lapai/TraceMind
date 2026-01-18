@echo off
SETLOCAL EnableDelayedExpansion

echo [TraceMind] Starting environment setup...

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

:: 2. Create and Enter Server Directory
cd /d %~dp0
if not exist "server\data" mkdir "server\data"

:: 3. Create Virtual Environment
if not exist ".venv" (
    echo [TraceMind] Creating virtual environment...
    python -m venv .venv
)

:: 4. Activate Venv and Install Requirements
echo [TraceMind] Installing dependencies using TUNA mirror...
call .venv\Scripts\activate
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r server\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 5. Launch Server
echo [TraceMind] Server is starting at http://localhost:8000
echo [TraceMind] Press Ctrl+C to stop.
python server\main.py

pause
