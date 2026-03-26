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

# Ensure Playwright is installed (can be slow, but avoids crashes)
# uv run playwright install chromium

uv run --no-sync pytest tests/ui/ "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "UI tests failed!"
    exit 1
fi

echo "----------------------------------------"
echo "UI tests passed successfully."
