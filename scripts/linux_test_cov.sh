#!/bin/bash

# Subject Frame Extractor - Coverage Report
# Runs unit tests and generates a coverage report for core/ and ui/

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "Running tests with coverage (unit tests)..."
echo "----------------------------------------"

uv run --no-sync pytest --cov=core --cov=ui --cov-report=html:htmlcov tests/unit/ "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "Coverage run failed!"
    exit 1
fi

echo "----------------------------------------"
echo "Coverage results generated."
