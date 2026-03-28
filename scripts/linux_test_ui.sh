#!/bin/bash

# Subject Frame Extractor - UI/E2E Test Runner
# Runs Playwright browser automation tests in tests/ui/

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found. Please install it."
    exit 1
fi

echo "Running UI/E2E tests (Playwright)..."
echo "----------------------------------------"

# Ensure results directories exist
mkdir -p tests/results/failures
mkdir -p tests/results/logs

# Run UI tests with xdist for parallel performance
# If no arguments are provided, use tests/ui/ as default
if [ $# -eq 0 ]; then
    uv run --no-sync pytest tests/ui/ -n auto -o "addopts=-v --tb=short"
else
    # Pass arguments directly to pytest
    uv run --no-sync pytest -n auto -o "addopts=-v --tb=short" "$@"
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "UI tests failed!"
    exit 1
fi

echo "----------------------------------------"
echo "UI tests passed successfully."
