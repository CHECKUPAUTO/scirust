#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "── CCOS Graph Structure Fuzzing ──"

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local cmd="$2"
    echo -n "  [$name] "
    if eval "$cmd" 2>&1 | grep -q "test result: ok"; then
        echo "✔"
        PASS=$((PASS + 1))
    else
        echo "✘"
        FAIL=$((FAIL + 1))
    fi
}

echo "  Injecting graph anomalies..."

# Fuzz 1: Ghost nodes (nodes that exist but no edges reference them)
echo "  [ghost_nodes] Running graph connectivity tests..."
run_test "ghost_nodes" \
    "cargo test phase12_graph_connectivity --test integration_ccos 2>&1"

# Fuzz 2: Invalid edges (edges pointing to non-existent nodes)
echo "  [invalid_edges] Running event-graph consistency tests..."
run_test "invalid_edges" \
    "cargo test detect_missing_events --test event_graph_consistency 2>&1"

# Fuzz 3: Duplicate edges
echo "  [duplicate_edges] Running duplicate detection tests..."
run_test "duplicate_edges" \
    "cargo test detect_duplicate_events --test event_graph_consistency 2>&1"

# Fuzz 4: Out-of-order events
echo "  [out_of_order] Running order detection tests..."
run_test "out_of_order" \
    "cargo test detect_out_of_order_events --test event_graph_consistency 2>&1"

# Fuzz 5: Rollback simulation
echo "  [rollback] Running rollback divergence tests..."
run_test "rollback" \
    "cargo test rollback_simulation_divergence_detected --test event_graph_consistency 2>&1"

# Fuzz 6: AST parser under broken syntax
echo "  [broken_syntax] Running parser robustness..."
run_test "broken_syntax" \
    "cargo test phase8_ast_parser_robustness --test integration_ccos 2>&1"

# Fuzz 7: Snapshot divergence detection
echo "  [snapshot_divergence] Running snapshot differential..."
run_test "snapshot_divergence" \
    "cargo test snapshot_differential_drift_detection --test snapshot_differential 2>&1"

# Fuzz 8: Failure propagation through invalid edges
echo "  [failure_propagation] Running failure propagation..."
run_test "failure_propagation" \
    "cargo test phase4_failure_propagation --test integration_ccos 2>&1"

echo ""
echo "── Graph Fuzzing Results ──"
echo "  Passed: $PASS / $((PASS + FAIL))"
echo "  Failed: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    echo "  GRAPH FUZZING: ❌ ANOMALIES DETECTED"
    exit 1
else
    echo "  GRAPH FUZZING: ✅ ALL ANOMALIES HANDLED CORRECTLY"
    exit 0
fi
