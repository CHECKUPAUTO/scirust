#!/usr/bin/env bash
set -Eeuo pipefail

readonly repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$repo_root"

readonly tree_file="$(mktemp)"
trap 'rm -f -- "$tree_file"' EXIT

cargo +1.89.0 tree --locked --edges normal >"$tree_file"
if grep -Eq '(^| )scirust v|(^| )octasoma v|(^| )octacore v|(^| )rsi v' "$tree_file"; then
    printf '%s\n' 'default dependency tree contains a premium crate' >&2
    exit 1
fi

for relax in slhav2-full rsi-dgm rsi-full neural-embed; do
    if cargo +1.89.0 tree --locked --edges features | grep -Fq "feature \"$relax\""; then
        printf 'default feature graph unexpectedly enables REPLAY-RELAX feature %s\n' "$relax" >&2
        exit 1
    fi
done

printf '%s\n' 'community and deterministic feature boundaries verified'

