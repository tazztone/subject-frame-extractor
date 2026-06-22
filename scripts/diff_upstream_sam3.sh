#!/bin/bash
# Helper script to diff our local vendored sam3 against the official upstream repository.

set -e

REMOTE_NAME="upstream_sam3"
UPSTREAM_URL="https://github.com/facebookresearch/sam3"
# The exact commit hash that the submodule pointed to (bfbed072a07a6a52c8d5fdc75a7a186251a835b1)
DEFAULT_COMMIT="bfbed072a07a6a52c8d5fdc75a7a186251a835b1"

# 1. Add the remote if it doesn't exist
if ! git remote | grep -q "^${REMOTE_NAME}$"; then
    echo "Adding git remote '${REMOTE_NAME}' -> ${UPSTREAM_URL}"
    git remote add "$REMOTE_NAME" "$UPSTREAM_URL"
fi

# 2. Fetch the latest objects from upstream
echo "Fetching from upstream remote '${REMOTE_NAME}'..."
git fetch "$REMOTE_NAME" --quiet

# 3. Determine target commit/branch
TARGET="${1:-$DEFAULT_COMMIT}"
echo "Comparing 'sam3_vendored/sam3' against '${TARGET}:sam3'..."
echo "========================================================================="

# 4. Perform the diff
git diff "$TARGET:sam3" sam3_vendored/sam3
