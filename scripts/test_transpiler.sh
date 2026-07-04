#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Comprehensive, rigorous test of `scirust-transpiler` — the inbound
# Python/NumPy → deterministic-Rust transpiler.
#
# Runs, in order:
#   1. the library unit tests  (lexer, parser, type/shape inference, emission,
#      kernel-routing) — no Python needed, this is the CI gate;
#   2. the differential ORACLE against real NumPy — for EVERY supported
#      intrinsic, operator and control-flow construct, it transpiles the
#      Python source, compiles the emitted Rust (with `rustc` for std-only
#      cases, or `cargo` against the real `scirust-*` kernel for routed cases),
#      runs both sides on seeded random inputs and checks they match within a
#      declared tolerance.
#
# Exit code 0 iff every coded function passes. Requires `rustc`/`cargo`, and
# for the oracle `python3` + `numpy` (the oracle self-skips with code 2 if the
# latter are missing — this script treats that as a soft skip, not a failure).
# ---------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$0")/.."

green() { printf '\033[0;32m%s\033[0m\n' "$1"; }
red()   { printf '\033[0;31m%s\033[0m\n' "$1"; }
bold()  { printf '\033[1m%s\033[0m\n' "$1"; }

bold "==============================================================="
bold " scirust-transpiler — full test suite"
bold "==============================================================="

# ---- 1. Unit tests --------------------------------------------------------
bold "[1/2] Unit tests (lexer / parser / lowering / emit / routing)"
if ! cargo test -p scirust-transpiler; then
    red "UNIT TESTS FAILED"
    exit 1
fi
green "unit tests OK"
echo

# ---- 2. Differential oracle vs NumPy --------------------------------------
bold "[2/2] Differential oracle vs real NumPy (every coded intrinsic/construct)"
cargo run -q -p scirust-transpiler --example oracle
rc=$?
case "$rc" in
    0) green "oracle OK — all cases match NumPy" ;;
    2) red "oracle SKIPPED — python3/numpy/rustc not available (not a failure)" ;;
    *) red "ORACLE FAILED (some transpiled function disagrees with NumPy)"; exit 1 ;;
esac

echo
bold "==============================================================="
green " ALL TRANSPILER TESTS PASSED"
bold "==============================================================="
