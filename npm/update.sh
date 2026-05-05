#!/usr/bin/env bash
set -euo pipefail

SCIRUST_HOME="${SCIRUST_HOME:-$HOME/.scirust}"
INSTALL_DIR="$SCIRUST_HOME"
REPO="https://github.com/CHECKUPAUTO/scirust.git"
BRANCH="feat/portable-simd-and-views"

if [ ! -d "$INSTALL_DIR/src" ]; then
    echo "⚠  SciRust is not installed. Running install instead..."
    exec bash "${BASH_SOURCE%/*}/install.sh"
fi

OLD_VERSION=$(cat "$INSTALL_DIR/.version" 2>/dev/null || echo "unknown")
echo "→ Current version: $OLD_VERSION"
echo "→ Checking for updates..."

cd "$INSTALL_DIR/src"
git fetch origin
LOCAL_SHA=$(git rev-parse --short HEAD)
REMOTE_SHA=$(git rev-parse --short "origin/$BRANCH")

if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
    echo "✅ Already up to date ($LOCAL_SHA)"
    exit 0
fi

echo "→ Update available: $LOCAL_SHA → $REMOTE_SHA"
git checkout "$BRANCH"
git pull origin "$BRANCH"

echo "→ Rebuilding SciRust..."
cargo build --release -p scirust

cp target/release/scirust "$INSTALL_DIR/bin/scirust"
chmod +x "$INSTALL_DIR/bin/scirust"

NEW_VERSION=$(git rev-parse --short HEAD)
echo "$NEW_VERSION" > "$INSTALL_DIR/.version"

echo ""
echo "✅ SciRust updated: $OLD_VERSION → $NEW_VERSION"
