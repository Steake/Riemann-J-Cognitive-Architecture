#!/bin/bash
# test.sh
# Run tests for Riemann-J Cognitive Architecture

set -e  # Exit on error

echo "=================================="
echo "Riemann-J Test Suite"
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

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found."
    echo "   Install dev dependencies: pip install -r requirements-dev.txt"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
RUN_DEMOS=false
DEMO_TYPE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --bdd)
            TEST_TYPE="bdd"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --demos)
            RUN_DEMOS=true
            DEMO_TYPE="all"
            shift
            ;;
        --demo-adversarial)
            RUN_DEMOS=true
            DEMO_TYPE="adversarial"
            shift
            ;;
        --demo-reasoning)
            RUN_DEMOS=true
            DEMO_TYPE="reasoning"
            shift
            ;;
        --demo-formative)
            RUN_DEMOS=true
            DEMO_TYPE="formative"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: ./test.sh [OPTIONS]"
            echo ""
            echo "Test Options:"
            echo "  --unit          Run only unit tests"
            echo "  --bdd           Run only BDD tests"
            echo "  --integration   Run only integration tests"
            echo "  --coverage      Generate coverage report"
            echo "  -v, --verbose   Verbose output"
            echo ""
            echo "Demo Options:"
            echo "  --demos               Run all demonstration scripts"
            echo "  --demo-adversarial    Run adversarial transparency demo"
            echo "  --demo-reasoning      Run uncertainty-gated reasoning demo"
            echo "  --demo-formative      Run formative experience demo"
            echo ""
            echo "Other Options:"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./test.sh                          # Run all tests"
            echo "  ./test.sh --unit --coverage        # Run unit tests with coverage"
            echo "  ./test.sh --bdd -v                 # Run BDD tests verbosely"
            echo "  ./test.sh --demos                  # Run all demos"
            echo "  ./test.sh --demo-adversarial       # Run adversarial demo only"
            echo "  ./test.sh --unit --demos           # Run unit tests then demos"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './test.sh --help' for usage information."
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src/riemann_j --cov-report=term-missing --cov-report=html"
fi

# Add test path based on type
case $TEST_TYPE in
    unit)
        echo "🧪 Running unit tests..."
        PYTEST_CMD="$PYTEST_CMD tests/unit/"
        ;;
    bdd)
        echo "🎭 Running BDD tests..."
        PYTEST_CMD="$PYTEST_CMD tests/bdd/"
        ;;
    integration)
        echo "🔗 Running integration tests..."
        PYTEST_CMD="$PYTEST_CMD tests/integration/"
        ;;
    all)
        echo "🧪 Running all tests..."
        PYTEST_CMD="$PYTEST_CMD tests/"
        ;;
esac

echo ""

# Run tests
$PYTEST_CMD

EXIT_CODE=$?

# Run demos if requested
if [ "$RUN_DEMOS" = true ]; then
    echo ""
    echo "=================================="
    echo "🎬 Running Demonstration Scripts"
    echo "=================================="
    echo ""
    
    DEMO_EXIT_CODE=0
    
    run_demo() {
        local demo_name=$1
        local demo_file=$2
        local timeout_duration=${3:-60}
        
        echo "▶ Running $demo_name..."
        if timeout $timeout_duration python "$demo_file" > /dev/null 2>&1; then
            echo "  ✅ $demo_name completed successfully"
        else
            echo "  ❌ $demo_name failed or timed out"
            DEMO_EXIT_CODE=1
        fi
    }
    
    case $DEMO_TYPE in
        adversarial)
            run_demo "Adversarial Transparency Demo" "demos/demo_adversarial_simple.py" 45
            ;;
        reasoning)
            run_demo "Uncertainty-Gated Reasoning Demo" "demos/demo_reasoning_simple.py" 45
            ;;
        formative)
            run_demo "Formative Experience Demo" "demos/demo_formative_simple.py" 45
            ;;
        all)
            run_demo "Adversarial Transparency Demo" "demos/demo_adversarial_simple.py" 45
            run_demo "Uncertainty-Gated Reasoning Demo" "demos/demo_reasoning_simple.py" 45
            run_demo "Formative Experience Demo" "demos/demo_formative_simple.py" 45
            ;;
    esac
    
    echo ""
    if [ $DEMO_EXIT_CODE -eq 0 ]; then
        echo "✅ All demos completed successfully!"
    else
        echo "❌ Some demos failed"
        EXIT_CODE=1
    fi
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================="
    echo "✅ All tests passed!"
    echo "=================================="
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo "📊 Coverage report generated in htmlcov/index.html"
        echo "   Open with: open htmlcov/index.html"
    fi
else
    echo "=================================="
    echo "❌ Some tests failed (exit code: $EXIT_CODE)"
    echo "=================================="
fi

echo ""
exit $EXIT_CODE
