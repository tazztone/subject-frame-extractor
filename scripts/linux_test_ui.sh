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

# Use PYTEST_WORKERS if set (e.g., in CI), otherwise fallback to auto
WORKERS=${PYTEST_WORKERS:-auto}
echo "Using workers: $WORKERS"

# Ensure results directories exist
mkdir -p tests/results/failures
mkdir -p tests/results/logs

# Run UI tests with xdist for parallel performance
# If no arguments are provided, use tests/ui/ as default
if [ $# -eq 0 ]; then
    uv run --no-sync pytest tests/ui/ -n "$WORKERS" -o "addopts=-v --tb=short"
else
    # Pass arguments directly to pytest but ensure it's scoped to tests/ui/
    # and excludes heavy integration/GPU tests to prevent accidental re-runs
    uv run --no-sync pytest tests/ui/ -n "$WORKERS" -o "addopts=-v --tb=short" -m "not integration and not gpu_e2e" "$@"
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "UI tests failed!"
    exit 1
fi

echo "----------------------------------------"
echo "UI tests passed successfully."
