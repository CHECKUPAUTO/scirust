#!/usr/bin/env sh
# Extract the co-change ground truth defined in docs/CCOS_COCHANGE_PROTOCOL.md §3.
#
# One line per retained commit: the tab-separated list of `.rs` files it touched.
# Selection is deterministic and takes no tuning knob:
#
#   - non-merge commits only (a merge carries no content of its own);
#   - between 2 and 8 `.rs` files touched;
#   - only files still present in the working tree (a deleted file can be neither
#     recalled nor searched);
#   - after that filter, at least 2 files must remain.
#
#   sh scripts/ccos/cochange_cases.sh > cases.tsv
set -eu

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

git log --no-merges --format='%H' | while read -r commit; do
    files=$(git show --name-only --format='' "$commit" 2>/dev/null \
            | grep '\.rs$' || true)
    [ -z "$files" ] && continue

    # The 2..8 bound applies to what the commit touched, before the existence
    # filter — so a mass refactor is excluded on its real size, not on the
    # remnant that happens to survive in today's tree.
    touched=$(printf '%s\n' "$files" | sort -u | wc -l)
    [ "$touched" -lt 2 ] && continue
    [ "$touched" -gt 8 ] && continue

    present=$(printf '%s\n' "$files" | sort -u | while read -r f; do
        [ -f "$f" ] && printf '%s\n' "$f"
    done)
    [ -z "$present" ] && continue
    [ "$(printf '%s\n' "$present" | wc -l)" -lt 2 ] && continue

    printf '%s\n' "$present" | paste -sd '\t' -
done
