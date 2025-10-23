#!/bin/bash
# run.sh
# Run the Riemann-J Cognitive Architecture TUI

set -e  # Exit on error

echo "=================================="
echo "Riemann-J Cognitive Architecture"
echo "=================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found."
    echo "   Run './setup_venv.sh' first to set up the environment."
    exit 1
fi

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if package is installed
if ! python -c "import riemann_j" 2>/dev/null; then
    echo "❌ riemann-j package not found in environment."
    echo "   Run './setup_venv.sh' to install the package."
    exit 1
fi

echo "🚀 Starting Riemann-J TUI..."
echo ""
echo "Commands:"
echo "  - Type messages to interact with the agent"
echo "  - /switch <username> - Switch user context"
echo "  - /exit - Exit the application"
echo ""
echo "Press Ctrl+C to exit"
echo ""
echo "=================================="
echo ""

# Run the application
python -m riemann_j

echo ""
echo "✅ Application closed."
