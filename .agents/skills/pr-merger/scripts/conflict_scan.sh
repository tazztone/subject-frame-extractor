#!/usr/bin/env bash
# conflict_scan.sh — Detect leftover merge conflict markers in tracked files.
# Usage: bash scripts/conflict_scan.sh [path]
# Exit 0 = clean, Exit 1 = markers found

set -euo pipefail

SEARCH_PATH="${1:-.}"
MARKERS='(<{7}|={7}|>{7})'

echo "Scanning for conflict markers in: ${SEARCH_PATH}"

# Search only tracked text files, skip binary
FOUND=$(git grep -lnEI "${MARKERS}" -- "${SEARCH_PATH}" 2>/dev/null || true)

if [ -z "${FOUND}" ]; then
    echo "✓ No conflict markers found."
    exit 0
fi

echo ""
echo "✗ Conflict markers found in:"
echo "${FOUND}" | while IFS= read -r file; do
    COUNT=$(git grep -cEI "${MARKERS}" -- "${file}" 2>/dev/null || echo "0")
    echo "  ${file} (${COUNT} marker lines)"
done

echo ""
echo "Resolve these before pushing."
exit 1
