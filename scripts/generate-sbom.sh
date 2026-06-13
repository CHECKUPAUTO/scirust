#!/usr/bin/env bash
# Generate a reproducible CycloneDX SBOM for the scirust facade crate.
#
# Output: docs/sbom/scirust.cdx.json (CycloneDX 1.5 JSON) — a single aggregate
# SBOM for the `scirust` facade crate and its full transitive dependency
# closure.
#
# Note: cargo-cyclonedx always describes every workspace member (writing one
# SBOM next to each Cargo.toml). We keep only the root facade SBOM
# (`scirust.cdx.json`) and remove the per-member files it also emits.
#
# Reproducibility: the BOM timestamp is pinned to the current commit's date via
# SOURCE_DATE_EPOCH, so the same source tree always yields a byte-identical SBOM
# (cargo-cyclonedx emits no random serial number). This matches the project's
# determinism guarantee.
#
# Requirements: `cargo install cargo-cyclonedx --locked`.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! cargo cyclonedx --version >/dev/null 2>&1; then
    echo "error: cargo-cyclonedx not found — run: cargo install cargo-cyclonedx --locked" >&2
    exit 2
fi

# Pin the BOM timestamp to the last commit (overridable by the caller / CI).
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}"

cargo cyclonedx \
    --manifest-path Cargo.toml \
    --format json \
    --spec-version 1.5 \
    -q

# Keep only the root facade SBOM; drop the per-member files cargo-cyclonedx
# also writes throughout the workspace.
find . -name "*.cdx.json" \
    -not -path "./docs/*" \
    -not -path "./target/*" \
    -not -path "./scirust.cdx.json" \
    -delete

mkdir -p docs/sbom
mv -f scirust.cdx.json docs/sbom/scirust.cdx.json

echo "wrote docs/sbom/scirust.cdx.json (SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH)"
