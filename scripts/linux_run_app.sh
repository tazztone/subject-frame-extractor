#!/bin/bash

# Subject Frame Extractor - Linux Run Script
# This script automates the process of starting the application on Linux.

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    echo "This project prefers 'uv' for dependency management."
    echo "Please install it: https://github.com/astral-sh/uv"
    exit 1
fi

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment (.venv) not found."
    echo "Please ensure you have initialized the environment first."
    echo "Try running: uv sync"
    read -p "Press Enter to exit..."
    exit 1
fi

# Open browser to Gradio interface (Gradio default port is 7860)
echo "Opening browser to http://localhost:7860"
if command -v xdg-open &> /dev/null; then
    # Use xdg-open (Linux standard)
    xdg-open http://localhost:7860 > /dev/null 2>&1 &
elif command -v open &> /dev/null; then
    # Use open (macOS/other)
    open http://localhost:7860 > /dev/null 2>&1 &
else
    echo "Warning: Could not find a command to open the browser automatically."
    echo "Please manually navigate to http://localhost:7860 once the app starts."
fi

# Run the application
echo ""
echo "Starting Subject Frame Extractor..."
echo "-----------------------------------"
echo "The browser window should open automatically."
echo "If the page doesn't load immediately, please refresh after the app finishes starting."
echo ""

# Run via uv to ensure correct environment
uv run python app.py

# Keep the window open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with an error code."
    read -p "Press Enter to close..."
fi
