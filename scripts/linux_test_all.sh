#!/bin/bash

# Subject Frame Extractor - Full Test Suite
# Runs all tests in order: Unit (fast), Integration (slower), UI (interactive)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "----------------------------------------"
echo "Subject Frame Extractor - Running ALL TESTS"
echo "----------------------------------------"

# 1. Unit Tests
echo "--- Stage 1: Unit Tests ---"
bash "$SCRIPT_DIR/linux_test_unit.sh"
if [ $? -ne 0 ]; then exit 1; fi

# 2. Integration Tests
echo ""
echo "--- Stage 2: Integration Tests ---"
PYTEST_INTEGRATION_MODE=true uv run --no-sync pytest tests/integration/ -o "addopts=-v --tb=short" -m "integration or gpu_e2e" --no-cov
if [ $? -ne 0 ]; then exit 1; fi

# 3. UI/E2E Tests
echo ""
echo "--- Stage 3: UI/E2E Tests ---"
PYTEST_INTEGRATION_MODE=true bash "$SCRIPT_DIR/linux_test_ui.sh" -o "addopts=-v --tb=short" --no-cov
if [ $? -ne 0 ]; then exit 1; fi

# 4. Regression Tests
echo ""
echo "--- Stage 4: Regression Tests ---"
uv run --no-sync pytest tests/regression/ "$@"
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "----------------------------------------"
echo "SUCCESS: ALL TESTS PASSED."
echo "----------------------------------------"
