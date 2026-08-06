#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "── CCOS Replay Consistency Check ──"

# Run the deterministic replay test 5 times and verify identical outputs
PASS=0
FAIL=0

echo "  Running phase6_deterministic_replay test 5 times..."

OUTPUTS=()
for i in $(seq 1 5); do
    output=$(cargo test phase6_deterministic_replay --test integration_ccos 2>&1)
    result_line=$(echo "$output" | grep "test result" || echo "NOT FOUND")
    OUTPUTS+=("$result_line")
    echo "    Run $i: $result_line"
done

# Check all runs passed
for i in $(seq 1 5); do
    if echo "${OUTPUTS[$((i-1))]}" | grep -q "ok"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

# Check all results are identical
UNIQUE=$(printf '%s\n' "${OUTPUTS[@]}" | sort -u | wc -l)
if [ "$UNIQUE" -eq 1 ]; then
    echo "    ✓ All 5 runs produced identical results (deterministic replay)"
    PASS=$((PASS + 1))
else
    echo "    ✘ Results diverged across runs"
    FAIL=$((FAIL + 1))
fi

# Also test the event-graph consistency replay
echo "  Running replay_consistency_full_cycle test..."
if cargo test replay_consistency_full_cycle --test event_graph_consistency 2>&1 | grep -q "test result: ok"; then
    echo "    ✓ Full cycle replay consistency verified"
    PASS=$((PASS + 1))
else
    echo "    ✘ Full cycle replay consistency failed"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "── Replay Consistency Results ──"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "  REPLAY CONSISTENCY: DETERMINISTIC ✓"
exit 0
