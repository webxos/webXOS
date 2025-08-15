# main/server/mcp/scripts/run_tests.sh
#!/bin/bash

# Test runner script for Vial MCP Controller
set -e

# Configuration
VENV_DIR="venv"
TEST_DIR="tests"
COVERAGE_THRESHOLD=80

# Check if virtual environment exists, create if not
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run pytest with coverage
echo "Running tests with coverage..."
pytest $TEST_DIR \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=html:cov_html \
    --cov-fail-under=$COVERAGE_THRESHOLD \
    -v

# Check if tests passed
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
else
    echo "Tests failed!"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "Test run completed. Coverage report available at cov_html/index.html"
