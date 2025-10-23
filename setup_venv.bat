@echo off
REM setup_venv.bat
REM Virtual environment setup script for Riemann-J Cognitive Architecture (Windows)

echo ==================================
echo Riemann-J Virtual Environment Setup
echo ==================================
echo.

REM Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python not found in PATH.
    echo Please install Python 3.10+ and add it to PATH.
    exit /b 1
)

REM Check Python version
python --version | findstr /R "3\.1[0-9]\." >nul
if %errorlevel% neq 0 (
    echo Error: Python 3.10+ is required.
    python --version
    exit /b 1
)

echo Found Python:
python --version
echo.

REM Check if venv exists
if exist "venv\" (
    echo Virtual environment 'venv' already exists.
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing venv...
        rmdir /s /q venv
    ) else (
        echo Using existing venv. Run 'venv\Scripts\activate' to activate.
        exit /b 0
    )
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    exit /b 1
)

echo Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo.
echo Installing dependencies...
echo.

echo Installing core dependencies from requirements.txt...
pip install -r requirements.txt

REM Ask about dev dependencies
echo.
set /p INSTALL_DEV="Install development dependencies (pytest, black, etc.)? (Y/n): "
if /i not "%INSTALL_DEV%"=="n" (
    echo Installing development dependencies...
    pip install -r requirements-dev.txt
)

REM Install package in editable mode
echo.
echo Installing riemann-j package in editable mode...
pip install -e .

echo.
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate
echo.
echo To run the application:
echo   run.bat
echo.
echo To run tests:
echo   test.bat
echo.
echo To deactivate the virtual environment, run:
echo   deactivate
echo.

pause
