#!/usr/bin/env bash
set -euo pipefail

. "$HOME/.cargo/env"

export LD_LIBRARY_PATH="$(rustc +nightly --print sysroot)/lib:${LD_LIBRARY_PATH:-}"

echo "[build-driver] Building scirust-rustc-driver with rustc +nightly"
cargo +nightly build --release

echo "[build-driver] Built: ./target/release/scirust-rustc-driver"
echo "[build-driver] Usage:"
echo "    export LD_LIBRARY_PATH=\"$(rustc +nightly --print sysroot)/lib:\$LD_LIBRARY_PATH\""
echo "    ./target/release/scirust-rustc-driver myfile.rs -o mybin"
