#!/usr/bin/env bash
# fleet_collect.sh — pull CCOS field data from a fleet to a local hub and extract
# an analytics-ready JSON record per node. Local-first: just rsync + ccos, no
# central telemetry server (see docs/SELF_ANALYSIS.md → "Collecting field data").
#
# For each remote workspace it: rsyncs <workspace>.ccos and <workspace>.ccos.oplog
# into <out>/<host>/, then runs `ccos postmortem <ws> --json` to produce
# <out>/<host>/session.json (stats, hash-chain integrity, timeline, working set).
# Integrity is verified offline — a truncated/tampered transfer shows up as
# integrity.valid=false in the record.
#
# Usage:
#   scripts/fleet_collect.sh --out ./fleet \
#       user@node1:~/agent/workspace.ccos  user@node2:~/agent/workspace.ccos
#   # or read remotes from a file (one "user@host:/path/workspace.ccos" per line):
#   scripts/fleet_collect.sh --out ./fleet --hosts hosts.txt
#
# Env: CCOS (path to the ccos binary, default ./target/release/ccos).
set -euo pipefail

out="./fleet"
ccos="${CCOS:-./target/release/ccos}"
hosts_file=""
remotes=()

while [ $# -gt 0 ]; do
  case "$1" in
  --out)
    out="$2"
    shift 2
    ;;
  --hosts)
    hosts_file="$2"
    shift 2
    ;;
  --ccos)
    ccos="$2"
    shift 2
    ;;
  -h | --help)
    sed -n '2,20p' "$0"
    exit 0
    ;;
  *)
    remotes+=("$1")
    shift
    ;;
  esac
done

if [ -n "$hosts_file" ]; then
  while IFS= read -r line; do
    line="${line%%#*}"                       # strip comments
    line="$(echo "$line" | tr -d '[:space:]')" # strip whitespace
    [ -n "$line" ] && remotes+=("$line")
  done <"$hosts_file"
fi

if [ "${#remotes[@]}" -eq 0 ]; then
  echo "fleet_collect: no remotes given (pass user@host:/path/workspace.ccos or --hosts FILE)" >&2
  exit 2
fi
if [ ! -x "$ccos" ]; then
  echo "fleet_collect: ccos binary not found/executable at '$ccos' (build it, or set \$CCOS)" >&2
  exit 2
fi

mkdir -p "$out"
ok=0
fail=0
echo "Collecting ${#remotes[@]} node(s) → $out"
for remote in "${remotes[@]}"; do
  # remote = user@host:/path/workspace.ccos
  host="${remote%%:*}"        # user@host
  host="${host##*@}"          # host
  wsname="$(basename "${remote##*:}")" # workspace.ccos
  dest="$out/$host"
  mkdir -p "$dest"
  echo "  ── $remote"

  if ! rsync -az --timeout=30 "$remote" "$remote.oplog" "$dest/" 2>/dev/null; then
    # the .oplog may not exist (snapshot-only workspace) — retry without it
    if ! rsync -az --timeout=30 "$remote" "$dest/" 2>/dev/null; then
      echo "     ! rsync failed" >&2
      fail=$((fail + 1))
      continue
    fi
  fi

  if "$ccos" postmortem "$dest/$wsname" --json >"$dest/session.json" 2>/dev/null; then
    valid="$(grep -o '"valid": *[a-z]*' "$dest/session.json" | head -1 | grep -o '[a-z]*$' || echo '?')"
    len="$(grep -o '"timeline_len": *[0-9]*' "$dest/session.json" | grep -o '[0-9]*' || echo '?')"
    echo "     ✓ session.json — integrity.valid=$valid, timeline_len=$len"
    ok=$((ok + 1))
  else
    echo "     ! ccos postmortem --json failed" >&2
    fail=$((fail + 1))
  fi
done

echo "Done: $ok ok, $fail failed. Records under $out/<host>/session.json"
[ "$fail" -eq 0 ]
