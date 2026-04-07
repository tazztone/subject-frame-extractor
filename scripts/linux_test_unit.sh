#!/bin/bash

# Subject Frame Extractor - Unit Tests Runner
# Runs fast, mocked tests in tests/unit/

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Ensure we are NOT in integration mode for unit tests to prevent loading real models
unset PYTEST_INTEGRATION_MODE
export PYTEST_INTEGRATION_MODE=false

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found. Please install it."
    exit 1
fi

echo "Running unit tests (fast, mock-first)..."
echo "----------------------------------------"

# Run tests with xdist (auto workers)
uv run --no-sync pytest -n auto --no-cov -q --tb=short tests/unit/ "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "Unit tests failed!"
    exit 1
fi

echo "----------------------------------------"
echo "Unit tests passed successfully."
