#!/bin/bash

# Subject Frame Extractor - Playwright Setup
# Installs Chromium browser for UI/E2E tests.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "Installing Playwright (Chromium)..."
echo "----------------------------------------"

uv run --no-sync playwright install chromium

if [ $? -ne 0 ]; then
    echo "Error: Playwright setup failed!"
    exit 1
fi

echo "----------------------------------------"
echo "Playwright setup complete."
