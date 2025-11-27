#!/bin/bash
set -e  # Exit on any error

echo "--- Jules Test Environment Setup ---"

# Note: Jules automatically clones the repo to /app and cd's there
# Note: Jules has common tools pre-installed (git, python3, etc.)

# 1. Initialize submodules (needed for app.py sys.path setup)
echo "[1/3] Setting up submodules..."
git submodule update --init --recursive

# 2. Create and activate Python virtual environment
echo "[2/3] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install only test dependencies (no heavy ML packages)
echo "[3/3] Installing test dependencies..."
python -m pip install --upgrade pip
pip install uv

# Install test requirements - these are the only packages tests need unmocked
uv pip install -r tests/requirements-test.txt

# Add OpenCV headless since tests import cv2 directly (not mocked)
uv pip install opencv-python-headless

echo "--- Jules Test Environment Complete ---"

# Verify core test imports work
echo "Verifying test dependencies..."
python -c "
import numpy
import cv2  
import pytest
import gradio
from PIL import Image
import yaml
from skimage import io
print('✓ All test dependencies verified')
"

echo "Environment ready for testing!"
