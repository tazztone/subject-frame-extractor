#!/bin/bash

# Subject Frame Extractor - Integration Tests Runner
# Runs tests in tests/integration/ that require GPU or real models

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found. Please install it."
    exit 1
fi

echo "Running integration tests (real-mode, GPU-ready)..."
echo "----------------------------------------------------"

# Export integration mode so global mocks are disabled
export PYTEST_INTEGRATION_MODE=true

# Integration tests are heavy and should run serially or with -n 1
uv run --no-sync pytest tests/integration/ -q --tb=short -m "integration or gpu_e2e" --no-cov "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "Integration tests failed!"
    exit 1
fi

echo "----------------------------------------------------"
echo "Integration tests passed successfully."
