#!/usr/bin/env bash
set -Eeuo pipefail

readonly repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$repo_root"

cycles=100000
if [[ ${1:-} == '--smoke' ]]; then
    cycles=2000
elif [[ $# -ne 0 ]]; then
    printf 'usage: %s [--smoke]\n' "${0##*/}" >&2
    exit 2
fi

readonly commit="$(git rev-parse HEAD)"
readonly run_id="${commit}-cycles-${cycles}"
readonly out_dir="$repo_root/benchmark-results/$run_id"
if [[ -e "$out_dir" ]]; then
    printf 'refusing to overwrite benchmark result: %s\n' "$out_dir" >&2
    exit 1
fi
install -d -m 0700 -- "$out_dir"

CCOS_BENCH_COMMIT="$commit" \
    cargo +1.89.0 run --quiet --locked --bin ccos-bench-repro -- "$cycles" \
    >"$out_dir/result.json"

{
    printf 'commit=%s\n' "$commit"
    printf 'rustc=%s\n' "$(rustc +1.89.0 --version)"
    printf 'cargo=%s\n' "$(cargo +1.89.0 --version)"
    printf 'target=%s\n' "$(rustc +1.89.0 -vV | sed -n 's/^host: //p')"
    printf 'uname=%s\n' "$(uname -a)"
    printf 'dataset=synthetic-edit-cycle-v1\n'
    printf 'dataset_version=repository@%s\n' "$commit"
    printf 'seed=deterministic-cycle-index\n'
    printf 'cycles=%s\n' "$cycles"
    printf 'paging_cap=200\n'
} >"$out_dir/environment.txt"
(
    cd -- "$out_dir"
    sha256sum -- result.json environment.txt >SHA256SUMS
)
