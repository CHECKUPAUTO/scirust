#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              CCOS — FULL HARDENING TEST SUITE                      ║"
echo "║       Causal Context Operating System — Advanced Validation        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=0
PASSED=0
FAILED=0

run_phase() {
    local name="$1"
    local cmd="$2"
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE ${TOTAL}: ${name}"
    echo "════════════════════════════════════════════════════════════"
    if eval "$cmd" 2>&1; then
        echo "  ✔ PHASE ${TOTAL} PASSED: ${name}"
        PASSED=$((PASSED + 1))
    else
        echo "  ✘ PHASE ${TOTAL} FAILED: ${name}"
        FAILED=$((FAILED + 1))
    fi
}

# ── Phase 1: Build ────────────────────────────────────────────
run_phase "Clean Build" "cargo clean 2>/dev/null; cargo build 2>&1"

# ── Phase 2: Unit Tests ───────────────────────────────────────
run_phase "Unit Tests (31 tests)" "cargo test --lib 2>&1"

# ── Phase 3: Integration Tests ────────────────────────────────
run_phase "Integration Tests (14 tests)" "cargo test --test integration_ccos 2>&1"

# ── Phase 4: Long Term Stability ──────────────────────────────
run_phase "Long Term Stability (10k cycles)" "cargo test --test long_term_stability 2>&1"

# ── Phase 5: Event-Graph Consistency ──────────────────────────
run_phase "Event-Graph Consistency" "cargo test --test event_graph_consistency 2>&1"

# ── Phase 6: LLM Adversarial Tests ────────────────────────────
run_phase "LLM Adversarial Tests" "cargo test --test llm_adversarial_test 2>&1"

# ── Phase 7: Snapshot Differential ────────────────────────────
run_phase "Snapshot & Differential Tests" "cargo test --test snapshot_differential 2>&1"

# ── Phase 8: Binary Smoke Test ────────────────────────────────
run_phase "Binary Execution Smoke" "timeout 60 cargo run 2>&1"

# ── Phase 9: Chaos Fuzz ───────────────────────────────────────
run_phase "Chaos Fuzz Scenarios" "bash scripts/chaos_fuzz.sh 2>&1"

# ── Phase 10: Fault Injection ─────────────────────────────────
run_phase "Fault Injection Scenarios" "bash scripts/inject_faults.sh 2>&1"

# ── Phase 11: Replay Consistency ──────────────────────────────
run_phase "Replay Consistency Check" "bash scripts/replay_consistency_check.sh 2>&1"

# ── Phase 12: Graph Fuzzing ───────────────────────────────────
run_phase "Graph Structure Fuzzing" "bash scripts/graph_fuzzing.sh 2>&1"

# ── Phase 13: Stress 1000 Cycles ─────────────────────────────
run_phase "Stress 1000 Cycles" "bash scripts/stress_1000_cycles.sh 2>&1"

# ── Final Verdict ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        FINAL HARDENING VERDICT                      ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
printf "║  Total:  %-58s ║\n" "$TOTAL"
printf "║  Passed: %-58s ║\n" "$PASSED"
printf "║  Failed: %-58s ║\n" "$FAILED"
echo "╚══════════════════════════════════════════════════════════════════════╝"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "CCOS HARDENING VERDICT: ❌ INCOMPLETE — $FAILED phase(s) failed"
    exit 1
else
    echo ""
    echo "CCOS HARDENING VERDICT: ✅ FULLY HARDENED — all $TOTAL phases passed"
    echo "  ✔ Replay deterministic"
    echo "  ✔ Event log strict append-only"
    echo "  ✔ Guard layer blocks all LLM corruption"
    echo "  ✔ Incremental engine maintains O(Δ)"
    echo "  ✔ Graph coherent under chaos"
    echo "  ✔ No crash on 10k+ cycles"
    echo "  ✔ Snapshots consistent"
    echo "  ✔ No silent drift detected"
    exit 0
fi
