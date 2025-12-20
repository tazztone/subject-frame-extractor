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

# 3. Install dependencies 
echo "[3/3] Installing test dependencies..."
python -m pip install --upgrade pip
pip install uv

# Install requirements 
uv pip install -r requirements.txt
uv pip install -e SAM3_repo

echo "--- Jules Environment Complete ---"

