@echo off
REM test.bat
REM Run tests for Riemann-J Cognitive Architecture (Windows)

setlocal enabledelayedexpansion

echo ==================================
echo Riemann-J Test Suite
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

REM Check if pytest is installed
where pytest >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: pytest not found.
    echo Install dev dependencies: pip install -r requirements-dev.txt
    exit /b 1
)

REM Parse command line arguments
set TEST_TYPE=all
set COVERAGE=false
set VERBOSE=false

:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="--unit" (
    set TEST_TYPE=unit
    shift
    goto parse_args
)
if /i "%~1"=="--bdd" (
    set TEST_TYPE=bdd
    shift
    goto parse_args
)
if /i "%~1"=="--integration" (
    set TEST_TYPE=integration
    shift
    goto parse_args
)
if /i "%~1"=="--coverage" (
    set COVERAGE=true
    shift
    goto parse_args
)
if /i "%~1"=="-v" (
    set VERBOSE=true
    shift
    goto parse_args
)
if /i "%~1"=="--verbose" (
    set VERBOSE=true
    shift
    goto parse_args
)
if /i "%~1"=="--help" (
    echo Usage: test.bat [OPTIONS]
    echo.
    echo Options:
    echo   --unit          Run only unit tests
    echo   --bdd           Run only BDD tests
    echo   --integration   Run only integration tests
    echo   --coverage      Generate coverage report
    echo   -v, --verbose   Verbose output
    echo   --help          Show this help message
    echo.
    echo Examples:
    echo   test.bat                    # Run all tests
    echo   test.bat --unit --coverage  # Run unit tests with coverage
    echo   test.bat --bdd -v           # Run BDD tests verbosely
    exit /b 0
)
echo Unknown option: %~1
echo Run 'test.bat --help' for usage information.
exit /b 1

:end_parse_args

REM Build pytest command
set PYTEST_CMD=pytest

REM Add verbosity
if "%VERBOSE%"=="true" (
    set PYTEST_CMD=!PYTEST_CMD! -v
)

REM Add coverage
if "%COVERAGE%"=="true" (
    set PYTEST_CMD=!PYTEST_CMD! --cov=src/riemann_j --cov-report=term-missing --cov-report=html
)

REM Add test path based on type
if "%TEST_TYPE%"=="unit" (
    echo Running unit tests...
    set PYTEST_CMD=!PYTEST_CMD! tests/unit/
) else if "%TEST_TYPE%"=="bdd" (
    echo Running BDD tests...
    set PYTEST_CMD=!PYTEST_CMD! tests/bdd/
) else if "%TEST_TYPE%"=="integration" (
    echo Running integration tests...
    set PYTEST_CMD=!PYTEST_CMD! tests/integration/
) else (
    echo Running all tests...
    set PYTEST_CMD=!PYTEST_CMD! tests/
)

echo.

REM Run tests
!PYTEST_CMD!

set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
    echo ==================================
    echo All tests passed!
    echo ==================================
    
    if "%COVERAGE%"=="true" (
        echo.
        echo Coverage report generated in htmlcov\index.html
    )
) else (
    echo ==================================
    echo Some tests failed (exit code: %EXIT_CODE%^)
    echo ==================================
)

echo.
exit /b %EXIT_CODE%
