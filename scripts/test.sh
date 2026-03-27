#!/bin/bash

# Subject Frame Extractor - Standard Quality Pass
# Runs Ruff, Unit Tests, and Integration Smoke Tests.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found. Please install it."
    exit 1
fi

echo "----------------------------------------"
echo "Subject Frame Extractor - Quality Pass"
echo "----------------------------------------"

# 1. Linting
echo "--- Stage 1: Linting (Ruff) ---"
uv run ruff check core/ ui/ tests/
if [ $? -ne 0 ]; then exit 1; fi

# 2. Unit Tests
echo ""
echo "--- Stage 2: Unit Tests (Fast) ---"
uv run pytest tests/unit/
if [ $? -ne 0 ]; then exit 1; fi

# 3. Integration Smoke Tests
echo ""
echo "--- Stage 3: Integration Smoke Tests ---"
uv run pytest tests/integration/test_integration_smoke.py --no-cov
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "----------------------------------------"
echo "SUCCESS: ALL CHECKS PASSED."
echo "----------------------------------------"
