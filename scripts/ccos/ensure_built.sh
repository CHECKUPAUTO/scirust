#!/usr/bin/env sh
# SessionStart hook: make sure the `ccos` binary exists so the causal-memory
# feed (scripts/ccos/self_feed_hook.sh) has something to talk to.
#
# Fast no-op when a binary is already installed; otherwise kicks off the build
# in the background (never blocks session start) and logs to .ccos/build.log.
# Always exits 0 — a memory hook must never block the agent.

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p "$ROOT/.ccos" 2>/dev/null || true

for C in /usr/local/bin/ccos "$HOME/.cargo/bin/ccos" "$HOME/.local/bin/ccos" \
         "$ROOT/external/ccos-core/target/release/ccos"; do
  [ -x "$C" ] && exit 0
done

nohup sh "$ROOT/scripts/ccos/install.sh" > "$ROOT/.ccos/build.log" 2>&1 &
echo "ccos: binary absent — build lancé en arrière-plan (.ccos/build.log)"
exit 0
