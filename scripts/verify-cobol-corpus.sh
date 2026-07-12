#!/usr/bin/env bash
# Corpus-integrity gate for the validated public COBOL reference corpus
# (external/cobol-reference/selected), added in PR #384/#385.
#
# The corpus establishes GnuCOBOL syntax/compiler-acceptance baselines only (NOT
# runtime equivalence — see its README). This script makes those baselines
# tamper-evident and self-checking, mirroring how scirust-finmigrate's `finaudit`
# verifies its Golden Baselines:
#
#   1. Every file matches the committed SHA-256 manifest (no silent edits).
#   2. Every recorded compiler exit code is 0 (all programs still accepted).
#   3. The intentional compiler warnings are preserved verbatim (7 for EFITA3B8,
#      8 for EFITA3N8) — the README states they "must not be silently removed".
#
# Exits non-zero on any drift. No toolchain required beyond coreutils.
set -euo pipefail

# Resolve the corpus root relative to this script, so it runs from anywhere.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
corpus="$root/external/cobol-reference"
selected="$corpus/selected"

fail() { echo "FAIL: $*" >&2; exit 1; }

[ -d "$selected" ] || fail "corpus not found at $selected"

# 1. SHA-256 manifest. Paths in the manifest are relative to $corpus (they start
#    with 'selected/'), so verify from there.
echo "==> verifying SHA-256 manifest"
( cd "$corpus" && sha256sum -c selected/metadata/SHA256SUMS ) >/dev/null \
  || fail "SHA-256 manifest mismatch (corpus file edited without regenerating SHA256SUMS)"
n_files="$(wc -l < "$selected/metadata/SHA256SUMS")"
echo "    ok: $n_files files verified"

# 2. Compiler exit codes must all be 0.
echo "==> checking recorded compiler exit codes"
for f in "$selected"/logs/*.exit-code; do
    code="$(cat "$f")"
    [ "$code" = "0" ] || fail "$(basename "$f") = $code (expected 0)"
done
echo "    ok: all $(ls "$selected"/logs/*.exit-code | wc -l) exit codes are 0"

# 3. Preserved-warning invariants (README: warnings are intentional).
echo "==> checking preserved compiler warnings"
check_warnings() {
    local prog="$1" want="$2"
    local got
    got="$(grep -c 'warning:' "$selected/logs/$prog.stderr" || true)"
    [ "$got" = "$want" ] || fail "$prog: $got warnings (expected $want) — corpus warnings changed"
}
check_warnings "EFITA3B8.cob" 7
check_warnings "EFITA3N8.cob" 8
echo "    ok: EFITA3B8=7, EFITA3N8=8 warnings preserved"

echo "COBOL reference corpus: OK"
