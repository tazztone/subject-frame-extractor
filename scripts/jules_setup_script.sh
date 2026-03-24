#!/bin/bash
set -e

echo "--- Jules Test Environment Setup ---"

# 1. Initialize submodules SHALLOWLY — source only, no SAM3 deps installed
echo "[1/3] Setting up submodules..."
git submodule update --init --recursive --depth 1

# 2. Create and activate Python virtual environment
echo "[2/3] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
echo "[3/3] Installing test dependencies..."
python -m pip install --upgrade pip
pip install uv

# Install project deps from lockfile (single source of truth for torch)
# Use --no-install-project to avoid redundant registrations
uv sync --frozen --no-install-project

# Explicitly exclude SAM3_repo from Python path to prevent double-init
export PYTHONPATH=$(python -c "
import sys, pathlib
paths = [p for p in sys.path if 'SAM3_repo' not in p and 'sam3' not in p.lower()]
print(':'.join(paths))
")

echo "--- Jules Environment Complete ---"

# Verify torch imports cleanly before running full suite
echo "--- Jules Health Check ---"
python -c "import torch; print(f'Torch OK: {torch.__version__}')" && \
uv run pytest tests/unit/ --import-mode=importlib -q --tb=short

