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

echo "🚀 Starting Riemann-J..."
echo ""

# Parse command line arguments
MODE="cli"
ARGS=()

for arg in "$@"; do
    case $arg in
        --tui)
            MODE="tui"
            ARGS+=("$arg")
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

if [ "$MODE" == "tui" ]; then
    echo "Mode: Full-Screen TUI (Textual UI)"
    echo ""
    echo "Commands:"
    echo "  - Type messages to interact with the agent"
    echo "  - /help - Show all available commands"
    echo "  - /stats - Show cognitive metrics"
    echo "  - /introspect - Show current mental state"
    echo "  - /inject-state <name> [--pn=0.8] [--crisis] - Inject synthetic state"
    echo "  - /switch <username> - Switch user context"
    echo "  - /quit or /exit - Exit the application"
    echo ""
    echo "Dashboard updates live at 4 Hz"
else
    echo "Mode: Interactive CLI (Command-Line Interface)"
    echo ""
    echo "Commands:"
    echo "  - Type messages to interact with the agent"
    echo "  - /help - Show all available commands"
    echo "  - /stats - Show cognitive metrics"
    echo "  - /pn - Show PN trajectory (last 50 values)"
    echo "  - /introspect - Show current mental state"
    echo "  - /inject-state <name> [--pn=0.8] [--crisis] - Inject synthetic state"
    echo "  - /toggle-status - Toggle status bar display"
    echo "  - /exit - Exit the application"
    echo ""
    echo "Tip: Use --tui flag for full-screen dashboard"
fi

echo ""
echo "Press Ctrl+C to exit"
echo ""
echo "=================================="
echo ""

# Run the application with parsed arguments
python -m riemann_j "${ARGS[@]}"

echo ""
echo "✅ Application closed."
