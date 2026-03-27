#!/bin/bash

# Subject Frame Extractor - SAM3 Tests Runner
# Runs Integration tests specifically for SAM3 (Slow)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found. Please install it."
    exit 1
fi

echo "Running SAM3 tests (Heavyweight Integration/E2E)..."
echo "----------------------------------------"

# SAM3 currently mostly has integration/E2E tests in tests/integration/ 
# and some generic logic in tests/unit/ marked with -m sam3
PYTEST_INTEGRATION_MODE=true uv run --no-sync pytest -m sam3 --no-cov "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "SAM3 tests failed!"
    exit 1
fi

echo "----------------------------------------"
echo "SAM3 tests passed successfully."
