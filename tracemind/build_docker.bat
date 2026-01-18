@echo off
echo [TraceMind] Starting Docker image build...

:: 1. Check if Docker is running
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not running! Please install Docker Desktop first.
    pause
    exit /b 1
)

:: 2. Build the image
echo [TraceMind] Building Docker image 'tracemind:v1'...
docker build -t tracemind:v1 ./server

if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed!
    pause
    exit /b 1
)

:: 3. Save to tar
echo [TraceMind] Saving image to 'tracemind.tar'...
docker save -o tracemind.tar tracemind:v1

if %errorlevel% neq 0 (
    echo [ERROR] Failed to save image to tar!
    pause
    exit /b 1
)

echo.
echo ======================================================
echo [SUCCESS] tracemind.tar has been created successfully!
echo You can now upload this file to your NAS.
echo ======================================================
pause
