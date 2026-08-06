#!/usr/bin/env sh
# PostToolUse hook: the transparent "hardware intercept" that feeds the agent's
# side effects (files read/edited, cargo failures) into the CCOS causal memory
# at .ccos/workspace.ccos. See external/ccos-core/docs/SELF_ANALYSIS.md (Mode B).
#
# One writer per workspace: this hook is the writer. Do not run a persistent
# `ccos mcp .ccos/workspace.ccos` server at the same time.
# Always exits 0 — a memory hook must never block or fail the agent.

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

BIN="${CCOS_BIN:-}"
if [ -z "$BIN" ]; then
  for C in /usr/local/bin/ccos "$HOME/.cargo/bin/ccos" "$HOME/.local/bin/ccos" \
           "$ROOT/external/ccos-core/target/release/ccos"; do
    if [ -x "$C" ]; then BIN="$C"; break; fi
  done
fi
[ -n "$BIN" ] || exit 0   # not built yet — silently do nothing

mkdir -p "$ROOT/.ccos" 2>/dev/null || true

CCOS_BIN="$BIN" \
CCOS_WORKSPACE="${CCOS_WORKSPACE:-$ROOT/.ccos/workspace.ccos}" \
  python3 "$ROOT/external/ccos-core/scripts/ccos_self_feed.py" || true
exit 0
