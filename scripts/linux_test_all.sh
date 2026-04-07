#!/bin/bash
set -o pipefail

# Subject Frame Extractor - Full Test Suite
# Runs all tests in order: Unit (fast), Integration (slower), UI (interactive)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "----------------------------------------"
echo "Subject Frame Extractor - Running ALL TESTS"
echo "----------------------------------------"

# Clear integration mode so unit tests run with mocks by default.
# Individual stage runners (like linux_test_integration.sh) will set it as needed.
export PYTEST_INTEGRATION_MODE=false

# Create log directory
LOG_DIR="tests/results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_performance.log"
DURATION_THRESHOLD=0.1

# Clear log file and add header
echo "--- Performance Log (Threshold >= ${DURATION_THRESHOLD}s set in scripts/linux_test_all.sh): $(date) ---" > "$LOG_FILE"

run_and_log() {
    local stage_name=$1
    shift
    local safe_stage=$(echo "$stage_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
    local full_log="$LOG_DIR/stage_${safe_stage}_full.log"
    local temp_log=$(mktemp)
    
    local start_time=$SECONDS
    
    # Run tests with durations=0 (measure everything)
    # 1. Capture FULL output (including durations) to temp_log and full_log
    # 2. Show live output to console, excluding individual durations, the "slowest durations" banner, and the "hidden durations" summary
    "$@" --durations=0 2>&1 | tee "$temp_log" "$full_log" | grep -vE "^[0-9.]+s[[:space:]]|^=+.*slowest.*durations.*=+$|\([0-9]+ durations < .* hidden"
    local exit_code=${PIPESTATUS[0]} # Capture original exit code from the test command
    
    local duration=$((SECONDS - start_time))
    
    # Append header and ONLY the timing lines >= threshold to the final log
    echo "" >> "$LOG_FILE"
    echo "--- Stage: $stage_name ---" >> "$LOG_FILE"
    awk -v limit="$DURATION_THRESHOLD" '$1 ~ /^[0-9.]+s$/ { val = substr($1, 1, length($1)-1); if (val >= limit) print }' "$temp_log" >> "$LOG_FILE"
    
    rm -f "$temp_log"
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "✓ Stage $stage_name: PASSED (${duration}s)"
    else
        echo "✗ Stage $stage_name: FAILED (${duration}s)"
    fi
    
    return $exit_code
}

# 1. Unit Tests
echo "--- Stage 1: Unit Tests (Mocked) ---"
export PYTEST_INTEGRATION_MODE=false
run_and_log "Unit" bash "$SCRIPT_DIR/linux_test_unit.sh" "$@"
if [ $? -ne 0 ]; then exit 1; fi

# 2. Integration Tests
echo ""
echo "--- Stage 2: Integration Tests ---"
run_and_log "Integration" bash "$SCRIPT_DIR/linux_test_integration.sh" "$@"
if [ $? -ne 0 ]; then exit 1; fi

# 3. UI/E2E Tests
echo ""
echo "--- Stage 3: UI/E2E Tests ---"
run_and_log "UI" bash "$SCRIPT_DIR/linux_test_ui.sh" --no-cov "$@"
if [ $? -ne 0 ]; then exit 1; fi

# 4. Regression Tests
echo ""
echo "--- Stage 4: Regression Tests ---"
run_and_log "Regression" uv run --no-sync pytest tests/regression/ -q --tb=short --no-cov "$@"
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "----------------------------------------"
echo "SUCCESS: ALL TESTS PASSED."
echo "Performance details (timings) saved to: $LOG_FILE"
echo "To view slowest tests: sort -t' ' -k1,1rn $LOG_FILE | head -n 20"
echo "----------------------------------------"
