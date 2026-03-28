#!/bin/bash

# Subject Frame Extractor - Unit Tests Runner
# Runs fast, mocked tests in tests/unit/

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found. Please install it."
    exit 1
fi

echo "Running unit tests (fast, mock-first)..."
echo "----------------------------------------"

uv run --no-sync pytest -n auto --cov-report=html:htmlcov tests/unit/ "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "Unit tests failed!"
    exit 1
fi

echo "----------------------------------------"
echo "Unit tests passed successfully."
