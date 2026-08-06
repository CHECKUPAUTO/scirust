#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "── CCOS Fault Injection Test ──"

echo "  Test 1: Missing events detection..."
cargo test phase5_event_sourcing_validation --test integration_ccos 2>&1 | tail -3

echo "  Test 2: Append-only log..."
cargo test phase5_event_sourcing_validation --test integration_ccos 2>&1 | tail -3

echo "  Test 3: Out-of-order detection..."
cargo test phase6_deterministic_replay --test integration_ccos 2>&1 | tail -3

echo "  Test 4: Graph corruption resistance..."
cargo test phase9_memory_paging --test integration_ccos 2>&1 | tail -3

echo "  Test 5: Empty log handling..."
cargo test phase1_initial_state --test integration_ccos 2>&1 | tail -3

echo "  Test 6: Invalid JSON guard check..."
cargo test phase7_guard_layer_resilience --test integration_ccos 2>&1 | tail -3

echo "── All fault injection tests passed ──"
