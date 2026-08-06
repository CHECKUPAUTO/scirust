#!/usr/bin/env bash
#
# Refuse generated SCIAGENT corpora inside a source checkout.  Ignore files
# cannot protect grep -R or find, so the durable invariant is physical: these
# directories must not exist below the repository root.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
forbidden=("$repo_root/data/crates_raw")

status=0
for path in "${forbidden[@]}"; do
    if [[ -e "$path" || -L "$path" ]]; then
        printf 'ERROR: generated SCIAGENT data is inside the checkout: %s\n' "$path" >&2
        printf 'Run: cargo run -p scirust-sciagent --bin sciagent-corpus -- migrate --apply\n' >&2
        status=1
    fi
done

if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi

printf 'SCIAGENT generated-data location check: clean\n'
