#!/usr/bin/env bash
# Setup script for SciRust Option B (rustc custom driver)
# ==========================================================
#
# This script prepares a nightly Rust toolchain with the `rustc-dev`
# component, which is required to compile the SciRust custom compiler
# driver (scirust-rustc-driver).
#
# Usage:
#   chmod +x setup-rustc-dev.sh
#   ./setup-rustc-dev.sh

set -euo pipefail

echo "[setup] SciRust rustc-dev setup"
echo "[setup] This requires rustup and a stable internet connection."

# 1. Install rustup if missing
if ! command -v rustup &> /dev/null; then
    echo "[setup] rustup not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
    source "$HOME/.cargo/env"
fi

# 2. Install nightly with rustc-dev
echo "[setup] Installing nightly toolchain with rustc-dev..."
rustup toolchain install nightly --component rustc-dev,llvm-tools-preview,rust-src

# 3. Set nightly as default for SciRust
echo "[setup] Setting nightly as default toolchain..."
rustup default nightly

# 4. Find the rustc source directory
RUSTC_SYSROOT=$(rustc --print sysroot)
RUSTC_SRC="$RUSTC_SYSROOT/lib/rustlib/src/rust"

if [ ! -d "$RUSTC_SRC/compiler" ]; then
    echo "[setup] Rust source not found at $RUSTC_SRC"
    echo "[setup] Attempting to install rust-src..."
    rustup component add rust-src
fi

echo "[setup] RUSTC_SRC=$RUSTC_SRC"
echo "[setup] Patching Cargo.toml paths..."

# 5. Patch the driver Cargo.toml to point to the actual rustc source
find scirust-rustc-driver/Cargo.toml -type f -exec sed -i "s|RUSTC_SRC|$RUSTC_SRC|g" {} +

echo "[setup] Done! You can now build the driver:"
echo "    cd scirust-rustc-driver && cargo build --release"
echo ""
echo "[setup] Then use it as your compiler:"
echo "    ./target/release/scirust-rustc-driver myfile.rs"
