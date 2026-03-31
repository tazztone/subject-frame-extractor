#!/bin/bash

# Subject Frame Extractor - Full Test Suite
# Runs all tests in order: Unit (fast), Integration (slower), UI (interactive)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "----------------------------------------"
echo "Subject Frame Extractor - Running ALL TESTS"
echo "----------------------------------------"

# Export integration mode so all child processes and xdist workers inherit it.
export PYTEST_INTEGRATION_MODE=true

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
    
    # Run tests with durations=0 (measure everything)
    # 1. Capture FULL output (including durations) to temp_log
    # 2. Show live output to console, but STOP at 'slowest durations' to keep it clean.
    "$@" --durations=0 2>&1 | tee "$temp_log" | sed '/^=.*slowest .* durations/,$d'
    
    # Append header and ONLY the timing lines to the final log
    echo "" >> "$LOG_FILE"
    echo "--- Stage: $stage_name ---" >> "$LOG_FILE"
    grep -E "^[0-9.]+s[[:space:]]+(call|setup|teardown)" "$temp_log" >> "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]} # Capture original exit code from the first command in the pipe
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
run_and_log "Regression" uv run --no-sync pytest tests/regression/ "$@"
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "----------------------------------------"
echo "SUCCESS: ALL TESTS PASSED."
echo "Performance details (timings) saved to: $LOG_FILE"
echo "To view slowest tests: sort -hr $LOG_FILE | head -n 20"
echo "----------------------------------------"
