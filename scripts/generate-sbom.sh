#!/bin/sh
# Generate a CycloneDX SBOM (JSON) for the SciRust workspace at
# docs/sbom/scirust.cdx.json.
#
# Requires `cargo-cyclonedx` (installed in CI via taiki-e/install-action). The
# CI job is informational (continue-on-error), but this script is self-contained
# and idempotent so it can also be run locally:
#
#   cargo install cargo-cyclonedx && ./scripts/generate-sbom.sh
set -eu

out_dir="docs/sbom"
mkdir -p "$out_dir"

# cargo-cyclonedx writes one "<crate>.cdx.json" next to each member's manifest.
cargo cyclonedx --format json

# Collect the top-level package's SBOM (falling back to any generated one).
if [ -f "scirust.cdx.json" ]; then
    mv -f "scirust.cdx.json" "$out_dir/scirust.cdx.json"
else
    found="$(find . -name '*.cdx.json' -not -path "./$out_dir/*" 2>/dev/null | head -n1)"
    if [ -n "$found" ]; then
        cp -f "$found" "$out_dir/scirust.cdx.json"
    fi
fi

# Remove the stray per-crate SBOMs so the working tree stays clean.
find . -name '*.cdx.json' -not -path "./$out_dir/*" -delete 2>/dev/null || true

test -f "$out_dir/scirust.cdx.json"
echo "Wrote $out_dir/scirust.cdx.json"
