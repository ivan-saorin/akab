@echo off
REM Setup script for AKAB development environment (Windows)

echo AKAB Development Setup (Windows)
echo ================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found!
    exit /b 1
)

REM Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Upgrade pip
echo Upgrading pip...
venv\Scripts\pip.exe install --upgrade pip

REM Install substrate
echo Installing Substrate dependency...
if exist ..\substrate (
    venv\Scripts\pip.exe install -e ..\substrate
) else (
    echo Error: Substrate not found at ..\substrate
    exit /b 1
)

REM Install AKAB
echo Installing AKAB...
venv\Scripts\pip.exe install -e .

REM Install dev dependencies
echo Installing dev dependencies...
venv\Scripts\pip.exe install -e .[dev]

REM Install provider dependencies
echo Installing provider dependencies...
venv\Scripts\pip.exe install -e .[providers]

REM Create .env from example
if exist .env.example (
    if not exist .env (
        echo Creating .env file...
        copy .env.example .env
        echo Please edit .env and add your API keys!
    )
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Activate environment: venv\Scripts\activate.bat
echo 2. Edit .env and add your API keys
echo 3. Run: python -m akab
