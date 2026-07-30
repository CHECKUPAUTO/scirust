#!/usr/bin/env sh
# Build the vendored CCOS Core (external/ccos-core) and install the `ccos`
# binary. Idempotent: safe to re-run; rebuilds only what cargo decides.
#
#   sh scripts/ccos/install.sh
#   PREFIX=/opt/bin CCOS_FEATURES=llm,license,learned-embed sh scripts/ccos/install.sh
#
# The `ccos` binary requires the `llm` feature — keep `llm` in CCOS_FEATURES.
set -eu

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
SRC="$ROOT/external/ccos-core"
FEATURES="${CCOS_FEATURES:-llm,license}"
BIN="$SRC/target/release/ccos"

echo "==> building ccos (release, --features $FEATURES)"
cd "$SRC"
cargo build --release --features "$FEATURES"

if [ ! -x "$BIN" ]; then
  echo "error: build produced no binary at $BIN" >&2
  exit 1
fi

for DIR in "${PREFIX:-/usr/local/bin}" "$HOME/.cargo/bin" "$HOME/.local/bin"; do
  if [ -d "$DIR" ] && [ -w "$DIR" ]; then
    install -m 755 "$BIN" "$DIR/ccos"
    echo "==> installed $DIR/ccos"
    "$DIR/ccos" doctor || true
    exit 0
  fi
done

echo "==> no writable install dir; binary available at $BIN"
"$BIN" doctor || true
