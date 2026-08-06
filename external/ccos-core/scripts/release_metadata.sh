#!/usr/bin/env bash
set -Eeuo pipefail

readonly repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$repo_root"

readonly commit="$(git rev-parse HEAD)"
readonly out_root="$repo_root/release-metadata"
readonly out_dir="$out_root/$commit"
install -d -m 0700 -- "$out_dir"

cargo +1.89.0 cyclonedx --format json --all --all-features
mapfile -t boms < <(find "$repo_root" -maxdepth 3 -type f -name '*.cdx.json' -print | sort)
if [[ ${#boms[@]} -eq 0 ]]; then
    printf '%s\n' 'cargo-cyclonedx produced no JSON SBOM' >&2
    exit 1
fi
for bom in "${boms[@]}"; do
    cp -- "$bom" "$out_dir/$(basename -- "$bom")"
done

{
    printf 'commit=%s\n' "$commit"
    printf 'rustc=%s\n' "$(rustc +1.89.0 --version --verbose | tr '\n' ';')"
    printf 'cargo=%s\n' "$(cargo +1.89.0 --version)"
    printf 'target=%s\n' "$(rustc +1.89.0 -vV | sed -n 's/^host: //p')"
    printf 'features=%s\n' 'all-features (SBOM); release profiles must record their exact subset'
    printf 'cargo_lock_sha256=%s\n' "$(sha256sum Cargo.lock | cut -d' ' -f1)"
} >"$out_dir/provenance.txt"

(
    cd -- "$out_dir"
    sha256sum -- ./* >SHA256SUMS
)

