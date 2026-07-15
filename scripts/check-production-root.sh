#!/usr/bin/env bash
#
# check-production-root.sh — fail if any pinned provenance/licensing root is
# still the PUBLIC DEMO root.
#
# Every pinned root ships as the public demo root so tests run out of the box —
# but a demo-signed artifact or license proves NOTHING (the demo seed is public).
# This guard exists so a release can never accidentally go out demo-signed.
#
# It is intentionally NOT part of the default PR CI (which legitimately runs on
# the demo roots). Wire it into your RELEASE pipeline instead, e.g.:
#
#     # .github/workflows/release.yml
#     - name: Refuse demo provenance roots
#       run: scripts/check-production-root.sh
#
# Exit: 0 all roots are production · 1 at least one is still the demo root ·
#       2 a constant could not be located (update this script alongside the code).
set -euo pipefail

# Public demo values (kept in lockstep with the crates' demo defaults).
DEMO_ROOT="82728023e3de7243e982d04ab09a7aa20a7fdb1fa10a0df2920060abc93a7f02"
# = first 8 bytes (LE) of SHA-256("SRL.canary" || demo_root || "scirust-autodiff")
DEMO_PROV_TAG="0x5bb6bb08d19746dd"

# Where to look (override for testing: --root-dir <dir>).
root_dir=""
while [ $# -gt 0 ]; do
    case "$1" in
        --root-dir) root_dir="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
if [ -z "$root_dir" ]; then
    root_dir="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
fi

emit_file="$root_dir/scirust-provenance/src/lib.rs"     # EMIT_ROOT_HEX
gpu_file="$root_dir/scirust-gpu/src/license.rs"         # GPU_LICENSE_ROOT_HEX
tag_file="$root_dir/scirust-autodiff/src/lib.rs"        # PROV_TAG (canary)

fail=0
missing=0

# Extract the single 64-hex literal from a root file (each contains exactly one).
extract_root() {
    grep -oiE '[0-9a-f]{64}' "$1" 2>/dev/null | head -1 | tr 'A-F' 'a-f'
}
# Extract the PROV_TAG u64 literal, underscores stripped, lower-cased.
extract_prov_tag() {
    grep -E 'const[[:space:]]+PROV_TAG' "$1" 2>/dev/null \
        | grep -oiE '0x[0-9a-f_]+' | head -1 | tr -d '_' | tr 'A-F' 'a-f'
}

check_root() {
    local name="$1" file="$2" got
    if [ ! -f "$file" ]; then
        echo "?? $name: file not found ($file)"; missing=1; return
    fi
    got="$(extract_root "$file")"
    if [ -z "$got" ]; then
        echo "?? $name: no 64-hex root literal found in $file — update this guard"; missing=1; return
    fi
    if [ "$got" = "$DEMO_ROOT" ]; then
        echo "FAIL $name: still the PUBLIC DEMO root ($file)"; fail=1
    else
        echo "ok   $name: production root pinned (${got:0:8}…)"
    fi
}

echo "Checking pinned provenance/licensing roots under: $root_dir"
check_root "EMIT_ROOT_HEX (scirust-provenance)" "$emit_file"
check_root "GPU_LICENSE_ROOT_HEX (scirust-gpu)" "$gpu_file"

# PROV_TAG (only meaningful if you ship the autodiff `canary` feature).
if [ ! -f "$tag_file" ]; then
    echo "?? PROV_TAG (scirust-autodiff): file not found ($tag_file)"; missing=1
else
    tag="$(extract_prov_tag "$tag_file")"
    if [ -z "$tag" ]; then
        echo "?? PROV_TAG (scirust-autodiff): const not found — update this guard"; missing=1
    elif [ "$tag" = "$DEMO_PROV_TAG" ]; then
        echo "FAIL PROV_TAG (scirust-autodiff): still the demo-derived value ($tag_file)"; fail=1
    else
        echo "ok   PROV_TAG (scirust-autodiff): production value pinned ($tag)"
    fi
fi

echo
if [ "$missing" -eq 1 ]; then
    echo "ERROR: one or more constants could not be located — the source moved."
    echo "       Update this script so the guard keeps protecting them."
    exit 2
fi
if [ "$fail" -eq 1 ]; then
    echo "REFUSING: at least one pinned root is still the public demo root."
    echo "          Generate a production seed and pin its root before releasing."
    echo "          See docs/PROVENANCE_OPERATIONS.md."
    exit 1
fi
echo "All pinned roots are production values. OK to release."
exit 0
