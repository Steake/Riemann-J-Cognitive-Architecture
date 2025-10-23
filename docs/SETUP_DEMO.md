# Setup and Usage Demonstration

## Quick Start Guide

### For Linux/macOS Users

```bash
# Step 1: Clone the repository
git clone https://github.com/Steake/Riemann-J-Cognitive-Architecture.git
cd Riemann-J-Cognitive-Architecture

# Step 2: Run the automated setup
./setup_venv.sh
```

**Expected Output:**
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
[Progress indicators...]

Install development dependencies (pytest, black, etc.)? (Y/n): y
Installing development dependencies...
[Progress indicators...]

📦 Installing riemann-j package in editable mode...
[Package installation...]

==================================
✅ Setup Complete!
==================================

To activate the virtual environment, run:
  source venv/bin/activate
```

```bash
# Step 3: Activate the environment
source venv/bin/activate

# Step 4: Run the application
./run.sh
```

**Expected Output:**
```
==================================
Riemann-J Cognitive Architecture
==================================

🚀 Starting Riemann-J TUI...

Commands:
  - Type messages to interact with the agent
  - /switch <username> - Switch user context
  - /exit - Exit the application

Press Ctrl+C to exit

==================================

Loading Causal Language Model and tokenizer...
Model loaded on device: cpu

[TUI interface launches with sparkline visualization]
```

```bash
# Step 5: Run tests
./test.sh --coverage
```

**Expected Output:**
```
==================================
Riemann-J Test Suite
==================================

🧪 Running all tests...

============================= test session starts ==============================
collected 23 items

tests/bdd/step_defs/test_pn_driver_steps.py ....                         [ 17%]
tests/unit/test_architecture.py ........                                 [ 52%]
tests/unit/test_config.py .....                                          [ 73%]
tests/unit/test_pn_driver.py ........                                    [100%]

============================= 23 passed in 10.02s ==============================

==================================
✅ All tests passed!
==================================

📊 Coverage report generated in htmlcov/index.html
```

### For Windows Users

```batch
# Step 1: Clone the repository
git clone https://github.com/Steake/Riemann-J-Cognitive-Architecture.git
cd Riemann-J-Cognitive-Architecture

# Step 2: Run the automated setup
setup_venv.bat
```

```batch
# Step 3: Activate the environment
venv\Scripts\activate

# Step 4: Run the application
run.bat
```

```batch
# Step 5: Run tests
test.bat --coverage
```

## Available Scripts

| Script | Linux/macOS | Windows | Purpose |
|--------|-------------|---------|---------|
| Setup Environment | `./setup_venv.sh` | `setup_venv.bat` | Create venv and install dependencies |
| Run Application | `./run.sh` | `run.bat` | Launch the Riemann-J TUI |
| Run Tests | `./test.sh` | `test.bat` | Execute test suite |

## Script Features

### setup_venv.sh / setup_venv.bat
- ✅ Detects Python 3.10+ automatically
- ✅ Creates isolated virtual environment
- ✅ Upgrades pip, setuptools, wheel
- ✅ Installs all core dependencies
- ✅ Optional development dependencies
- ✅ Installs package in editable mode
- ✅ Handles existing venv gracefully

### run.sh / run.bat
- ✅ Checks for virtual environment
- ✅ Auto-activates venv if needed
- ✅ Verifies package installation
- ✅ Launches TUI with helpful info
- ✅ Clean error messages

### test.sh / test.bat
- ✅ Multiple test modes (unit, bdd, integration, all)
- ✅ Coverage report generation
- ✅ Verbose output option
- ✅ Helpful error messages
- ✅ Exit codes for CI/CD integration

## Test Script Options

```bash
./test.sh --help
```

**Output:**
```
Usage: ./test.sh [OPTIONS]

Options:
  --unit          Run only unit tests
  --bdd           Run only BDD tests
  --integration   Run only integration tests
  --coverage      Generate coverage report
  -v, --verbose   Verbose output
  --help          Show this help message

Examples:
  ./test.sh                    # Run all tests
  ./test.sh --unit --coverage  # Run unit tests with coverage
  ./test.sh --bdd -v           # Run BDD tests verbosely
```

## Common Workflows

### First-Time Setup
```bash
git clone <repo-url>
cd Riemann-J-Cognitive-Architecture
./setup_venv.sh
source venv/bin/activate
./run.sh
```

### Daily Development
```bash
source venv/bin/activate
./test.sh --unit          # Quick validation
# [make code changes]
./test.sh --coverage      # Full validation
./run.sh                  # Manual testing
```

### Before Committing
```bash
./test.sh --coverage      # Ensure all tests pass
# Check coverage report
# Commit changes
```

### Troubleshooting

**Issue**: "Permission denied"
```bash
chmod +x setup_venv.sh run.sh test.sh
```

**Issue**: "Python 3.10+ not found"
- Install Python 3.10 or higher
- Ensure it's in your PATH

**Issue**: "pytest not found"
```bash
pip install -r requirements-dev.txt
```

**Issue**: Scripts don't detect activated venv
- Deactivate and reactivate: `deactivate && source venv/bin/activate`
- Or just run without pre-activation (scripts auto-activate)

## File Structure After Setup

```
riemann-j/
├── venv/                   # Virtual environment (created by setup script)
│   ├── bin/               # Executables (Linux/macOS)
│   ├── Scripts/           # Executables (Windows)
│   ├── lib/               # Installed packages
│   └── ...
├── setup_venv.sh          # Setup script (Linux/macOS)
├── setup_venv.bat         # Setup script (Windows)
├── run.sh                 # Run script (Linux/macOS)
├── run.bat                # Run script (Windows)
├── test.sh                # Test script (Linux/macOS)
├── test.bat               # Test script (Windows)
├── SCRIPTS_GUIDE.md       # Detailed script documentation
└── [rest of project files...]
```

## Next Steps

After running the setup:

1. **Read the Documentation**
   - `README.md` - Project overview
   - `SCRIPTS_GUIDE.md` - Detailed script usage
   - `docs/architecture/OVERVIEW.md` - Architecture details
   - `docs/api/API_REFERENCE.md` - API documentation

2. **Try the Application**
   - Run `./run.sh` or `run.bat`
   - Interact with the TUI
   - Try different user contexts with `/switch`

3. **Explore the Tests**
   - Run `./test.sh --coverage`
   - View coverage report in `htmlcov/index.html`
   - Read `tests/README.md` for testing guide

4. **Start Developing**
   - Follow the development workflow
   - Run tests frequently
   - Consult `SCRIPTS_GUIDE.md` for advanced usage

---

**Last Updated**: 2025-10-23  
**Version**: 4.0.0
