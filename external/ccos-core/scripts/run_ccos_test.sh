#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                     CCOS — FULL VALIDATION SUITE                    ║"
echo "║           Causal Context Operating System Test Harness              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL=0
PASSED=0
FAILED=0

run_test() {
    local name="$1"
    local cmd="$2"
    TOTAL=$((TOTAL + 1))
    echo "── [$TOTAL] $name ──"
    if eval "$cmd" 2>&1; then
        echo "  ✔ PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  ✘ FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# ── 1. Build check ─────────────────────────────────────────────────
echo "═══════ PHASE 1: Build Verification ═══════"
run_test "cargo build" "cargo build 2>&1"

# ── 2. Unit tests ──────────────────────────────────────────────────
echo "═══════ PHASE 2: Unit Tests ═══════"
run_test "cargo test" "cargo test 2>&1"

# ── 3. Integration tests (lib) ─────────────────────────────────────
echo "═══════ PHASE 3: Integration Tests ═══════"
run_test "cargo test --test integration_ccos" "cargo test --test integration_ccos 2>&1"

# ── 4. Main binary smoke test ──────────────────────────────────────
echo "═══════ PHASE 4: Binary Smoke Test ═══════"
run_test "cargo run" "timeout 60 cargo run 2>&1"

# ── 5. Replay validation ───────────────────────────────────────────
echo "═══════ PHASE 5: Replay Validation ═══════"
run_test "replay_validator.sh" "bash scripts/replay_validator.sh 2>&1"

# ── 6. Fault injection ─────────────────────────────────────────────
echo "═══════ PHASE 6: Fault Injection ═══════"
run_test "inject_faults.sh" "bash scripts/inject_faults.sh 2>&1"

# ── 7. Chaos fuzz ──────────────────────────────────────────────────
echo "═══════ PHASE 7: Chaos Fuzz ═══════"
run_test "chaos_fuzz.sh" "bash scripts/chaos_fuzz.sh 2>&1"

# ── 8. Stress 1000 cycles ──────────────────────────────────────────
echo "═══════ PHASE 8: Stress 1000 Cycles ═══════"
run_test "stress_1000_cycles.sh" "bash scripts/stress_1000_cycles.sh 2>&1"

# ── Final verdict ──────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        FINAL VERDICT                                ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Total:  $TOTAL                                                      "
echo "║  Passed: $PASSED                                                     "
echo "║  Failed: $FAILED                                                     "
echo "╚══════════════════════════════════════════════════════════════════════╝"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "SYSTEM VERDICT: ❌ NOT VALID — $FAILED test(s) failed"
    exit 1
else
    echo ""
    echo "SYSTEM VERDICT: ✅ FULLY VALID — all tests passed"
    exit 0
fi
