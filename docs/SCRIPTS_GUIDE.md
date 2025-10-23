# Script Usage Guide

This guide explains how to use the setup, run, and test scripts for the Riemann-J Cognitive Architecture.

## Prerequisites

- **Python 3.10+** is required
- **Git** (for cloning the repository)
- **Bash** (Linux/macOS) or **Command Prompt** (Windows)

## Quick Start

### Linux/macOS

```bash
# 1. Set up virtual environment and install dependencies
./setup_venv.sh

# 2. Activate virtual environment (if not already active)
source venv/bin/activate

# 3. Run the application
./run.sh

# 4. Run tests
./test.sh
```

### Windows

```batch
# 1. Set up virtual environment and install dependencies
setup_venv.bat

# 2. Activate virtual environment (if not already active)
venv\Scripts\activate

# 3. Run the application
run.bat

# 4. Run tests
test.bat
```

---

## Script Details

### 1. `setup_venv.sh` / `setup_venv.bat`

**Purpose**: Creates a virtual environment, installs all dependencies, and sets up the project.

**What it does**:
1. Detects Python 3.10+ installation
2. Creates a virtual environment in `./venv/`
3. Upgrades pip, setuptools, and wheel
4. Installs core dependencies from `requirements.txt`
5. Optionally installs development dependencies from `requirements-dev.txt`
6. Installs the `riemann-j` package in editable mode

**Usage**:

```bash
# Linux/macOS
./setup_venv.sh

# Windows
setup_venv.bat
```

**Options**:
- If a virtual environment already exists, you'll be prompted to recreate it or use the existing one
- When prompted, choose whether to install development dependencies (pytest, black, etc.)

**First-time setup output**:
```
==================================
Riemann-J Virtual Environment Setup
==================================

✓ Found Python: Python 3.10.x
📦 Creating virtual environment...
✓ Virtual environment created

🔌 Activating virtual environment...
⬆️  Upgrading pip...
📥 Installing dependencies...

Installing core dependencies from requirements.txt...
[dependency installation progress...]

Install development dependencies? (Y/n): y
Installing development dependencies...
[dev dependency installation progress...]

📦 Installing riemann-j package in editable mode...
[package installation...]

==================================
✅ Setup Complete!
==================================

To activate the virtual environment, run:
  source venv/bin/activate
```

---

### 2. `run.sh` / `run.bat`

**Purpose**: Runs the Riemann-J TUI application.

**What it does**:
1. Checks if virtual environment exists
2. Activates the virtual environment if not already active
3. Verifies the `riemann-j` package is installed
4. Launches the application using `python -m riemann_j`

**Usage**:

```bash
# Linux/macOS
./run.sh

# Windows
run.bat
```

**Output**:
```
==================================
Riemann-J Cognitive Architecture
==================================

🔌 Activating virtual environment...
🚀 Starting Riemann-J TUI...

Commands:
  - Type messages to interact with the agent
  - /switch <username> - Switch user context
  - /exit - Exit the application

Press Ctrl+C to exit

==================================

Loading Causal Language Model and tokenizer...
Model loaded on device: cpu

[TUI interface launches here]
```

**Interactive Commands**:
- **Regular input**: Chat with the cognitive agent
- **`/switch alice`**: Switch to user context "alice"
- **`/exit`**: Exit the application
- **`Ctrl+C`**: Force quit

---

### 3. `test.sh` / `test.bat`

**Purpose**: Runs the test suite with various options.

**What it does**:
1. Checks if virtual environment exists
2. Activates the virtual environment if not already active
3. Verifies pytest is installed
4. Runs tests based on provided options
5. Generates coverage reports if requested

**Usage**:

```bash
# Linux/macOS
./test.sh [OPTIONS]

# Windows
test.bat [OPTIONS]
```

**Options**:

| Option | Description |
|--------|-------------|
| `--unit` | Run only unit tests (tests/unit/) |
| `--bdd` | Run only BDD tests (tests/bdd/) |
| `--integration` | Run only integration tests (tests/integration/) |
| `--coverage` | Generate code coverage report |
| `-v`, `--verbose` | Verbose test output |
| `--help` | Show help message |

**Examples**:

```bash
# Run all tests
./test.sh

# Run only unit tests
./test.sh --unit

# Run BDD tests with verbose output
./test.sh --bdd -v

# Run all tests with coverage report
./test.sh --coverage

# Run unit tests with coverage, verbose
./test.sh --unit --coverage -v
```

**Output (successful run)**:
```
==================================
Riemann-J Test Suite
==================================

🔌 Activating virtual environment...
🧪 Running all tests...

============================= test session starts ==============================
collected 23 items

tests/bdd/step_defs/test_pn_driver_steps.py ....                         [ 17%]
tests/unit/test_architecture.py ........                                 [ 52%]
tests/unit/test_config.py .....                                          [ 73%]
tests/unit/test_pn_driver.py ........                                    [100%]

================================ tests coverage ================================
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
src/riemann_j/__init__.py               6      0   100%
src/riemann_j/config.py                13      0   100%
src/riemann_j/pn_driver.py             35     11    69%   39-61
src/riemann_j/architecture.py         108     50    54%   41-42, ...
-----------------------------------------------------------------
TOTAL                                 276    161    42%

============================= 23 passed in 10.02s ==============================

==================================
✅ All tests passed!
==================================

📊 Coverage report generated in htmlcov/index.html
   Open with: open htmlcov/index.html
```

---

## Common Workflows

### Initial Setup

```bash
# Clone repository
git clone https://github.com/Steake/Riemann-J-Cognitive-Architecture.git
cd Riemann-J-Cognitive-Architecture

# Set up environment (Linux/macOS)
./setup_venv.sh

# Or on Windows
setup_venv.bat

# Activate environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### Daily Development

```bash
# Activate environment (if not already active)
source venv/bin/activate

# Run tests before making changes
./test.sh

# Make code changes...

# Run tests again
./test.sh --coverage

# Run the application to test manually
./run.sh
```

### Testing Workflow

```bash
# Quick validation (unit tests only)
./test.sh --unit

# Full test suite with coverage
./test.sh --coverage

# Test specific component
pytest tests/unit/test_pn_driver.py -v

# Continuous testing while developing
pytest tests/unit/ --watch
```

### Troubleshooting

**Problem**: Script says "Permission denied"
```bash
# Solution (Linux/macOS)
chmod +x setup_venv.sh run.sh test.sh
```

**Problem**: "Python 3.10+ not found"
```bash
# Install Python 3.10+ for your system
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv

# macOS (with Homebrew)
brew install python@3.10

# Windows: Download from python.org
```

**Problem**: "Virtual environment not found"
```bash
# Solution: Run setup script first
./setup_venv.sh
```

**Problem**: "pytest not found"
```bash
# Solution: Install dev dependencies
pip install -r requirements-dev.txt
```

**Problem**: Import errors when running tests
```bash
# Solution: Reinstall package
pip install -e .
```

---

## Advanced Usage

### Manual Activation

If you prefer not to use the run scripts:

```bash
# Activate environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Run application directly
python -m riemann_j

# Or run script directly
python src/riemann_j/tui.py

# Run tests manually
pytest tests/ -v --cov=src/riemann_j
```

### Custom Test Runs

```bash
# Run specific test file
pytest tests/unit/test_config.py -v

# Run specific test class
pytest tests/unit/test_architecture.py::TestUserAttractor -v

# Run specific test function
pytest tests/unit/test_pn_driver.py::TestPNDriverRiemannZeta::test_calculate_pn_sigmoid_behavior -v

# Run tests matching pattern
pytest -k "pn_driver" -v

# Run with detailed output
pytest tests/ -vv --tb=long

# Run with profiling
pytest tests/ --profile
```

### Environment Management

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/        # Linux/macOS
rmdir /s /q venv    # Windows

# Recreate from scratch
./setup_venv.sh     # Will prompt to recreate
```

---

## Integration with IDEs

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.linting.enabled": true,
  "python.formatting.provider": "black"
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `venv/bin/python` (Linux/macOS) or `venv\Scripts\python.exe` (Windows)
4. Enable pytest: Settings → Tools → Python Integrated Tools → Testing → pytest

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: ./setup_venv.sh
      - run: source venv/bin/activate && ./test.sh --coverage
      - uses: codecov/codecov-action@v2
```

---

## Additional Resources

- **README.md**: Project overview and features
- **docs/architecture/OVERVIEW.md**: Design philosophy and architecture
- **docs/api/API_REFERENCE.md**: Complete API documentation
- **tests/README.md**: Testing guide with examples
- **TEST_RESULTS.md**: Latest test execution results

---

**Last Updated**: 2025-10-23  
**Version**: 4.0.0
