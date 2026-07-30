#!/usr/bin/env bash
# check-no-research-components.sh — CI guardrail: CCOS Core must never depend
# on, expose, or re-enable RSI / Forge / Research-Lab components (mission §27).
#
# Fails the build if any forbidden signal is found. False positives must be
# justified in security/forbidden-core-dependencies.toml (minimal, versioned).
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
note() { echo "::error::$1"; fail=1; }

FORBIDDEN_PKGS='^(ccos-rsi|rsi|forge-core|forge-bridge|forge-cli|ccos-forge|ccos-research-lab|ccos-sandbox|ccos-scirust|scirust|ccos-octacore|octacore|rsi_mcp|forge_mcp)$'

echo "== 1/4 cargo metadata: forbidden packages in the dependency graph =="
if command -v cargo >/dev/null; then
  cargo metadata --format-version 1 --locked 2>/dev/null \
    | python3 -c '
import json,sys,re
m=json.load(sys.stdin)
pat=re.compile(sys.argv[1], re.I)
bad=[p["name"] for p in m["packages"] if pat.match(p["name"])]
print("\n".join(bad))
sys.exit(1 if bad else 0)
' "$FORBIDDEN_PKGS" || note "forbidden package present in cargo metadata"
fi

echo "== 2/4 cargo tree: forbidden transitive dependencies =="
for feat in "" "--all-features" "--no-default-features"; do
  if cargo tree --workspace $feat 2>/dev/null | grep -Ei "$FORBIDDEN_PKGS"; then
    note "forbidden transitive dependency in cargo tree $feat"
  fi
done

echo "== 3/4 source scan: forbidden symbols and process execution =="
# Allowlist: security/forbidden-core-dependencies.toml documents accepted matches.
if grep -RInE 'rsi_bridge|mcp_ext|GuardedDgm|promote_to_live|self_modify|self_improvement|forge-core|ccos-rsi' \
  --include='*.rs' --include='*.toml' src crates tests examples benches 2>/dev/null \
  | grep -vFf <(grep -E '^[[:space:]]*[^#[:space:]]' security/forbidden-core-dependencies.toml 2>/dev/null | sed 's/#.*//' | sed 's/^\s*//;s/\s*$//' | grep -v '^$' || true); then
  note "forbidden symbol found in sources"
fi

echo "== 4/4 process execution policy =="
# Core must not spawn processes at runtime (mission §4.1). Exceptions are
# listed in security/process-execution-allowlist.toml with justification.
if grep -RInE 'Command::new|process::Command|tokio::process' --include='*.rs' src crates 2>/dev/null \
  | grep -vFf <(grep -E '^[[:space:]]*[^#[:space:]]' security/process-execution-allowlist.toml 2>/dev/null | sed 's/#.*//' | sed 's/^\s*//;s/\s*$//' | grep -v '^$' || true); then
  note "process execution outside the allowlist"
fi

if [ "$fail" -ne 0 ]; then
  echo "FORBIDDEN COMPONENT CHECK: FAILED"
  exit 1
fi
echo "FORBIDDEN COMPONENT CHECK: PASS"
