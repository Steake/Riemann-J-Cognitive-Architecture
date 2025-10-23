@echo off
REM run.bat
REM Run the Riemann-J Cognitive Architecture TUI (Windows)

echo ==================================
echo Riemann-J Cognitive Architecture
echo ==================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Error: Virtual environment not found.
    echo Run 'setup_venv.bat' first to set up the environment.
    exit /b 1
)

REM Activate virtual environment
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if package is installed
python -c "import riemann_j" >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: riemann-j package not found in environment.
    echo Run 'setup_venv.bat' to install the package.
    exit /b 1
)

echo Starting Riemann-J TUI...
echo.
echo Commands:
echo   - Type messages to interact with the agent
echo   - /switch ^<username^> - Switch user context
echo   - /exit - Exit the application
echo.
echo Press Ctrl+C to exit
echo.
echo ==================================
echo.

REM Run the application
python -m riemann_j

echo.
echo Application closed.
