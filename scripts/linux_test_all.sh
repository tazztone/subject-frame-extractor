#!/bin/bash

# Subject Frame Extractor - Full Test Suite
# Runs all tests in order: Unit (fast), Integration (slower), UI (interactive)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Point to Python virtual environment bundled Nvidia libraries
NVIDIA_LIBS=$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -type d -name "lib" | paste -sd : -)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"

echo "----------------------------------------"
echo "Subject Frame Extractor - Running ALL TESTS"
echo "----------------------------------------"

# Create log directory
LOG_DIR="tests/results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_performance.log"

# Clear log file and add header
echo "--- Performance Log (Timings Only): $(date) ---" > "$LOG_FILE"

run_and_log() {
    local stage_name=$1
    shift
    local temp_log=$(mktemp)

    "$@" --durations=0 2>&1 | tee "$temp_log" | sed '/^=.*slowest .* durations/,$d'
    local exit_code=${PIPESTATUS[0]} # Capture original exit code from the first command in the pipe immediately

    # Append header and ONLY the timing lines to the final log
    echo "" >> "$LOG_FILE"
    echo "--- Stage: $stage_name ---" >> "$LOG_FILE"
    grep -E "^[0-9.]+s[[:space:]]+(call|setup|teardown)" "$temp_log" >> "$LOG_FILE"

    rm "$temp_log"
    return $exit_code
}

# 1. Unit Tests
echo "--- Stage 1: Unit Tests ---"
run_and_log "Unit" bash "$SCRIPT_DIR/linux_test_unit.sh"
if [ $? -ne 0 ]; then exit 1; fi

# 2. Integration Tests
echo ""
echo "--- Stage 2: Integration Tests ---"
run_and_log "Integration" bash "$SCRIPT_DIR/linux_test_integration.sh"
if [ $? -ne 0 ]; then exit 1; fi

# 3. UI/E2E Tests
echo ""
echo "--- Stage 3: UI/E2E Tests ---"
run_and_log "UI" bash "$SCRIPT_DIR/linux_test_ui.sh" --no-cov
if [ $? -ne 0 ]; then exit 1; fi

# 4. Regression Tests
echo ""
echo "--- Stage 4: Regression Tests ---"
run_and_log "Regression" env PYTEST_INTEGRATION_MODE=true uv run --no-sync pytest tests/regression/ --no-cov "$@"
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "----------------------------------------"
echo "SUCCESS: ALL TESTS PASSED."
echo "Performance details (timings) saved to: $LOG_FILE"
echo "To view slowest tests: sort -hr $LOG_FILE | head -n 20"
echo "----------------------------------------"
