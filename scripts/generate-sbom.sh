#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="docs/sbom"
OUT_FILE="$OUT_DIR/scirust.cdx.json"

mkdir -p "$OUT_DIR"

# Prefer the cargo subcommand if available (newer installs), otherwise fallback
if command -v cargo >/dev/null 2>&1 && cargo cyclonedx --version >/dev/null 2>&1; then
  echo "Running: cargo cyclonedx --workspace --output $OUT_FILE"
  cargo cyclonedx --workspace --output "$OUT_FILE"
elif command -v cargo-cyclonedx >/dev/null 2>&1; then
  echo "Running: cargo-cyclonedx --workspace --output $OUT_FILE"
  cargo-cyclonedx --workspace --output "$OUT_FILE"
else
  echo "ERROR: cargo-cyclonedx not found. Ensure the 'Install cargo-cyclonedx' step ran successfully." >&2
  exit 1
fi

echo "SBOM generated at $OUT_FILE"
