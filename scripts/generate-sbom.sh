#!/bin/sh
# Generate one deterministic CycloneDX JSON document covering every Cargo
# workspace member and its resolved dependency graph.
set -eu

out_dir="docs/sbom"
output="$out_dir/scirust.cdx.json"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM
mkdir -p "$out_dir"

SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}"
export SOURCE_DATE_EPOCH

# cargo-cyclonedx emits one BOM beside every workspace manifest. The merge
# step below retains every component/dependency edge and verifies that every
# package reported by Cargo metadata is represented in the aggregate.
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_merge_cyclonedx.py
cargo +stable cyclonedx --format json
cargo +stable metadata --locked --no-deps --format-version 1 > "$tmp_dir/metadata.json"

find . -name '*.cdx.json' -not -path "./$out_dir/*" -print | sort > "$tmp_dir/boms.txt"
test -s "$tmp_dir/boms.txt"
python3 scripts/merge-cyclonedx.py \
    --output "$output" \
    --cargo-metadata "$tmp_dir/metadata.json" \
    --input-list "$tmp_dir/boms.txt"

# Remove per-member intermediates so regeneration does not dirty every crate.
while IFS= read -r bom; do
    rm -f -- "$bom"
done < "$tmp_dir/boms.txt"

test -s "$output"
echo "Wrote complete workspace SBOM: $output"
