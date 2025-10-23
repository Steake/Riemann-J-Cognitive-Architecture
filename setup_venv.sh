#!/bin/bash
# setup_venv.sh
# Virtual environment setup script for Riemann-J Cognitive Architecture

set -e  # Exit on error

echo "=================================="
echo "Riemann-J Virtual Environment Setup"
echo "=================================="
echo ""

# Detect Python version
PYTHON_CMD=""
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$($python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PYTHON_VERSION" > "3.9" ]]; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3.10+ is required but not found."
    echo "   Please install Python 3.10 or higher."
    exit 1
fi

echo "✓ Found Python: $($PYTHON_CMD --version)"
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment 'venv' already exists."
    read -p "   Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing venv..."
        rm -rf venv
    else
        echo "ℹ️  Using existing venv. Run 'source venv/bin/activate' to activate."
        exit 0
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment."
    echo "   Try: $PYTHON_CMD -m pip install --user virtualenv"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
echo ""

# Install core dependencies
echo "Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

# Ask about dev dependencies
echo ""
read -p "Install development dependencies (pytest, black, etc.)? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install package in editable mode
echo ""
echo "📦 Installing riemann-j package in editable mode..."
pip install -e .

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  ./run.sh"
echo ""
echo "To run tests:"
echo "  ./test.sh"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate"
echo ""
