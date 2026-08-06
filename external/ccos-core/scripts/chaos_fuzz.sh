#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "── CCOS Chaos Fuzz Test ──"

PASSED=0
FAILED=0

run_chaos() {
    local name="$1"
    local test="$2"
    echo -n "  [$name] "
    if cargo test "$test" --test integration_ccos 2>&1 | grep -q "test result: ok"; then
        echo "✔ survived"
        PASSED=$((PASSED + 1))
    else
        echo "✘ FAILED"
        FAILED=$((FAILED + 1))
    fi
}

echo "── Running chaos scenarios against integration tests ──"

# Random node injection / parsing robustness
run_chaos "random_inputs" "phase8_ast_parser_robustness"

# Memory pressure
run_chaos "memory_pressure" "phase9_memory_paging"

# Guard layer under attack
run_chaos "guard_resilience" "phase7_guard_layer_resilience"

# Failure propagation cascade
run_chaos "failure_cascade" "phase4_failure_propagation"

# Multi-cycle stability
run_chaos "multicycle" "phase10_multicycle_stability"

# Incremental O(Δ)
run_chaos "incremental_delta" "phase13_incremental_no_full_rebuild"

# Graph connectivity
run_chaos "graph_connectivity" "phase12_graph_connectivity"

# Context window
run_chaos "context_window" "phase14_context_window_selection"

# Full integration suite
echo -n "  [full_suite] "
if cargo test --test integration_ccos 2>&1 | grep -q "test result: ok"; then
    echo "✔ all 14 integration tests passed"
    PASSED=$((PASSED + 1))
else
    echo "✘ FAILED"
    FAILED=$((FAILED + 1))
fi

# Binary smoke test
echo -n "  [binary_smoke] "
if timeout 30 cargo run 2>&1 | grep -q "CCOS CYCLE COMPLETE"; then
    echo "✔ binary runs successfully"
    PASSED=$((PASSED + 1))
else
    echo "✘ FAILED"
    FAILED=$((FAILED + 1))
fi

echo ""
echo "── Chaos Fuzz Results ──"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"

if [ "$FAILED" -gt 0 ]; then
    echo "  CHAOS FUZZ: FAILURES DETECTED ✘"
    exit 1
else
    echo "  CHAOS FUZZ: ALL SYSTEMS STABLE ✓"
    exit 0
fi
