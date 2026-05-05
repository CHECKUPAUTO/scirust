#!/usr/bin/env bash
set -euo pipefail

SCIRUST_HOME="${SCIRUST_HOME:-$HOME/.scirust}"
INSTALL_DIR="$SCIRUST_HOME"
REPO="https://github.com/CHECKUPAUTO/scirust.git"
BRANCH="feat/portable-simd-and-views"

echo "╔══════════════════════════════════════════════╗"
echo "║   SciRust Installer v0.13 — Pure Rust ML     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Prerequisites
for cmd in cargo git; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "❌ Missing: $cmd. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
done

mkdir -p "$INSTALL_DIR/bin"

if [ -d "$INSTALL_DIR/src" ]; then
    echo "→ Repository exists, pulling latest..."
    cd "$INSTALL_DIR/src"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "→ Cloning repository..."
    git clone --branch "$BRANCH" "$REPO" "$INSTALL_DIR/src"
fi

echo "→ Building SciRust (release mode)..."
cd "$INSTALL_DIR/src"
cargo build --release -p scirust

cp target/release/scirust "$INSTALL_DIR/bin/scirust"
chmod +x "$INSTALL_DIR/bin/scirust"

# Record version
VERSION=$(git rev-parse --short HEAD)
echo "$VERSION" > "$INSTALL_DIR/.version"

echo ""
echo "✅ SciRust installed: $INSTALL_DIR/bin/scirust"
echo "✅ Version: $VERSION"
echo ""
echo "Add to PATH:"
echo "  export PATH=\"$INSTALL_DIR/bin:\$PATH\""
echo ""
echo "Then run: scirust"
