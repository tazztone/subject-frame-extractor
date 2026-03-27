#!/bin/bash

# Subject Frame Extractor - SAM2 Tests Runner
# Runs both Unit and Integration tests specifically for SAM2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found. Please install it."
    exit 1
fi

echo "Running SAM2 tests (Unit + Integration)..."
echo "----------------------------------------"

# 1. Run SAM2 Unit Tests (found in tests/unit/ marked with -m sam2)
# 2. Run SAM2 Integration Tests (found in tests/integration/ marked with -m sam2)
# We use PYTEST_INTEGRATION_MODE=true to allow real inference in integration step
PYTEST_INTEGRATION_MODE=true uv run --no-sync pytest -m sam2 --no-cov "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "SAM2 tests failed!"
    exit 1
fi

echo "----------------------------------------"
echo "SAM2 tests passed successfully."
