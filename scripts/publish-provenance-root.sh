#!/usr/bin/env bash
#
# publish-provenance-root.sh — record a SciRust provenance/licensing public root
# in a GPG-signed, annotated git tag (and, if available, an OpenTimestamps proof)
# so the root provably predates any suspect copy.
#
# The forensic value of the Lamport/Merkle signatures rests on proving the public
# root existed BEFORE a defendant's artifact. A git tag is timestamped by the tag
# object and, once pushed, by the forge; OpenTimestamps adds an independent,
# blockchain-anchored proof. Run this the moment you pin a production root, and
# BEFORE distributing anything signed under it.
#
# Usage:
#   scripts/publish-provenance-root.sh --root <64-hex> [--tag <name>] [--push]
#
#   --root <64-hex>   the PUBLIC 32-byte Merkle root (from `license-tool keygen`)
#   --tag  <name>     tag name (default: provenance-root-vN, auto-incremented)
#   --push            push the tag to origin (default: off — you push it yourself,
#                     since pushing is an outward-facing action)
#
# This script never sees the secret seed and refuses the public demo root.
set -euo pipefail

# The public demo root — signing under it proves nothing, so refuse to publish it.
DEMO_ROOT="82728023e3de7243e982d04ab09a7aa20a7fdb1fa10a0df2920060abc93a7f02"

root=""
tag=""
push=0

while [ $# -gt 0 ]; do
    case "$1" in
        --root) root="${2:-}"; shift 2 ;;
        --tag)  tag="${2:-}";  shift 2 ;;
        --push) push=1;        shift ;;
        -h|--help)
            sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

# --- validate the root ------------------------------------------------------
root="$(printf '%s' "$root" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
if [ -z "$root" ]; then
    echo "error: --root <64-hex> is required" >&2
    exit 2
fi
if ! printf '%s' "$root" | grep -Eq '^[0-9a-f]{64}$'; then
    echo "error: --root must be exactly 64 lower-case hex chars (32 bytes)" >&2
    exit 2
fi
if [ "$root" = "$DEMO_ROOT" ]; then
    echo "error: that is the PUBLIC DEMO root — it proves nothing. Generate a" >&2
    echo "       production seed in an HSM first (see docs/PROVENANCE_OPERATIONS.md)." >&2
    exit 2
fi

# --- must be inside the repo, on a clean-enough state ------------------------
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "error: run this inside the scirust git repository" >&2
    exit 2
fi

# --- pick a tag name (auto-increment provenance-root-vN if not given) --------
if [ -z "$tag" ]; then
    n=1
    while git rev-parse -q --verify "refs/tags/provenance-root-v${n}" >/dev/null 2>&1; do
        n=$((n + 1))
    done
    tag="provenance-root-v${n}"
fi
if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null 2>&1; then
    echo "error: tag '${tag}' already exists — pick another --tag" >&2
    exit 2
fi

# --- compute a stamp (UTC date + SHA-256 of the root bytes) -----------------
# Note: this timestamp is informational; the AUTHORITATIVE priority proof is the
# tag's own commit/push time and any OpenTimestamps proof, not this string.
stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
head_commit="$(git rev-parse HEAD)"

# SHA-256 over the raw 32 root bytes (portable: prefer sha256sum, else shasum).
root_sha256=""
if command -v sha256sum >/dev/null 2>&1; then
    root_sha256="$(printf '%s' "$root" | xxd -r -p 2>/dev/null | sha256sum | awk '{print $1}')" || true
elif command -v shasum >/dev/null 2>&1; then
    root_sha256="$(printf '%s' "$root" | xxd -r -p 2>/dev/null | shasum -a 256 | awk '{print $1}')" || true
fi

message="$(cat <<EOF
SciRust provenance/licensing public root — ${tag}

root=${root}
root_sha256=${root_sha256:-<unavailable: xxd/sha256 not found>}
published_utc=${stamp}
repo_head=${head_commit}

This tag records the PUBLIC Merkle root that verifies SciRust license and
emitted-artifact signatures. The secret master seed is NOT here and never will
be. Its purpose is to establish that this root existed at (or before) this tag's
timestamp, so signatures made under it cannot be dismissed as planted after a
dispute. See docs/PROVENANCE_OPERATIONS.md.
EOF
)"

# --- create the tag: GPG-signed if a signing key is configured, else annotated
if git config --get user.signingkey >/dev/null 2>&1; then
    echo "Creating GPG-signed tag '${tag}'..."
    if ! git tag -s "${tag}" -m "${message}"; then
        echo "warning: signed tag failed (GPG not usable?); falling back to annotated." >&2
        git tag -a "${tag}" -m "${message}"
    fi
else
    echo "note: no git signing key configured (git config user.signingkey);" >&2
    echo "      creating an ANNOTATED (unsigned) tag. A GPG-signed tag is stronger." >&2
    git tag -a "${tag}" -m "${message}"
fi

echo "Created tag '${tag}':"
git --no-pager tag -v "${tag}" 2>/dev/null || git --no-pager show --no-patch "${tag}"

# --- optional OpenTimestamps proof ------------------------------------------
if command -v ots >/dev/null 2>&1; then
    ots_file="${tag}.root.txt"
    printf '%s\n' "$message" > "$ots_file"
    echo "Stamping ${ots_file} with OpenTimestamps..."
    ots stamp "$ots_file" || echo "warning: 'ots stamp' failed; do it manually later." >&2
    echo "  -> commit ${ots_file} and ${ots_file}.ots; upgrade later with: ots upgrade ${ots_file}.ots"
else
    echo "note: OpenTimestamps client 'ots' not found — a git tag alone is fine to"
    echo "      start, but installing 'ots' and stamping gives an independent,"
    echo "      blockchain-anchored proof. See https://opentimestamps.org"
fi

# --- push (only if asked) ---------------------------------------------------
if [ "$push" -eq 1 ]; then
    echo "Pushing tag '${tag}' to origin..."
    git push origin "${tag}"
    echo "Done. The root is now timestamped by the forge as well."
else
    echo
    echo "Tag created locally. To publish (this makes the timestamp externally"
    echo "verifiable), push it yourself:"
    echo "    git push origin ${tag}"
fi
