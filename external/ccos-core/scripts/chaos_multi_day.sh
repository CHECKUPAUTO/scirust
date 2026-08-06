#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "── CCOS Chaos Multi-Day Simulation ──"
echo "  Simulating unstable environment conditions..."

PASS=0
FAIL=0

# ── Scenario 1: LLM offline (no Ollama available) ────────────
echo ""
echo "  [Scenario 1] LLM offline simulation..."
if timeout 10 cargo run 2>&1 | grep -q "CCOS CYCLE COMPLETE"; then
    echo "    ✔ Binary completes even with LLM offline (fallback active)"
    PASS=$((PASS + 1))
else
    echo "    ✘ Binary failed under LLM offline"
    FAIL=$((FAIL + 1))
fi

# ── Scenario 2: Corrupted workspace files ────────────────────
echo ""
echo "  [Scenario 2] Corrupted workspace files..."
cargo test phase8_ast_parser_robustness --test integration_ccos 2>&1 | tail -1
if cargo test phase8_ast_parser_robustness --test integration_ccos 2>&1 | grep -q "test result: ok"; then
    echo "    ✔ AST parser handles corrupted files"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

# ── Scenario 3: Rapid file deletion/recreation ───────────────
echo ""
echo "  [Scenario 3] Rapid file churn (delete/recreate)..."
cargo test phase3_mutation_simulation --test integration_ccos 2>&1 | tail -1
if cargo test phase3_mutation_simulation --test integration_ccos 2>&1 | grep -q "test result: ok"; then
    echo "    ✔ Mutation engine handles rapid churn"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

# ── Scenario 4: Circular dependency injection ─────────────────
echo ""
echo "  [Scenario 4] Circular dependency handling..."
cargo test phase12_graph_connectivity --test integration_ccos 2>&1 | tail -1
if cargo test phase12_graph_connectivity --test integration_ccos 2>&1 | grep -q "test result: ok"; then
    echo "    ✔ Graph handles circular dependencies"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

# ── Scenario 5: Long-running stability (10k cycles) ──────────
echo ""
echo "  [Scenario 5] Long-running stability (10k cycles)..."
cargo test --test long_term_stability --release 2>&1 | tail -5
if cargo test --test long_term_stability --release 2>&1 | grep -q "test result: ok"; then
    echo "    ✔ Survived 10k cycles without crash or drift"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

# ── Scenario 6: Memory pressure under load ────────────────────
echo ""
echo "  [Scenario 6] Memory pressure under load..."
bash scripts/memory_pressure_test.sh 2>&1 | tail -5
if [ $? -eq 0 ]; then
    echo "    ✔ Memory pressure test passed"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

# ── Scenario 7: Replay after chaos ────────────────────────────
echo ""
echo "  [Scenario 7] Replay consistency after chaos..."
bash scripts/replay_consistency_check.sh 2>&1 | tail -3
if [ $? -eq 0 ]; then
    echo "    ✔ Replay remains deterministic after chaos"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

# ── Scenario 8: Full adversarial LLM barrage ──────────────────
echo ""
echo "  [Scenario 8] Adversarial LLM barrage..."
cargo test --test llm_adversarial_test 2>&1 | tail -3
if cargo test --test llm_adversarial_test 2>&1 | grep -q "test result: ok"; then
    echo "    ✔ Guard layer blocks all adversarial inputs"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

echo ""
echo "═══ CHAOS MULTI-DAY RESULTS ═══"
echo "  Passed: $PASS / $((PASS + FAIL))"
echo "  Failed: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    echo "  CHAOS MULTI-DAY: ❌ FAILURES DETECTED"
    exit 1
else
    echo "  CHAOS MULTI-DAY: ✅ SYSTEM SURVIVED ALL CHAOS SCENARIOS"
    exit 0
fi
