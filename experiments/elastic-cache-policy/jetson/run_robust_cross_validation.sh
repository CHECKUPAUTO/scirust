#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
POLICY_BRANCH="research/elastic-cache-policy-discovery"
TRACE_PATH="${TRACE_PATH:-}"
STEPS="${STEPS:-1800}"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
command -v cargo >/dev/null || fail "cargo introuvable"
command -v python3 >/dev/null || fail "python3 introuvable"
[ -d "$SCIRUST_SOURCE/.git" ] || fail "dépôt SciRust absent: $SCIRUST_SOURCE"

SCIRUST_REMOTE="$(git -C "$SCIRUST_SOURCE" remote get-url origin)"
git -C "$SCIRUST_SOURCE" fetch origin --prune
if [ ! -d "$SCIRUST_RUN/.git" ]; then
    git clone --shared --no-checkout "$SCIRUST_SOURCE" "$SCIRUST_RUN"
fi
git -C "$SCIRUST_RUN" remote set-url origin "$SCIRUST_REMOTE"
git -C "$SCIRUST_RUN" fetch --prune origin \
    "+refs/heads/$POLICY_BRANCH:refs/remotes/origin/$POLICY_BRANCH"
git -C "$SCIRUST_RUN" checkout -B "$POLICY_BRANCH" \
    "refs/remotes/origin/$POLICY_BRANCH"

if [ -z "$TRACE_PATH" ]; then
    TRACE_PATH="$(
        find "$WORK_ROOT/results" -type f -name dream_counterfactual_trace.csv \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )"
fi
[ -n "$TRACE_PATH" ] && [ -f "$TRACE_PATH" ] || fail "trace Dream réelle introuvable"

OUTPUT_DIR="$(dirname "$TRACE_PATH")/robust-cross-validation"
mkdir -p "$OUTPUT_DIR"

cargo +nightly-2026-07-02 build --release \
    --manifest-path "$SCIRUST_RUN/experiments/elastic-cache-policy/Cargo.toml"

BINARY="$SCIRUST_RUN/experiments/elastic-cache-policy/target/release/scirust-cache-policy"
python3 \
    "$SCIRUST_RUN/experiments/elastic-cache-policy/jetson/robust_cross_validation.py" \
    --binary "$BINARY" \
    --trace "$TRACE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --seed 20260804 \
    --steps "$STEPS" \
    --quality-budget 0.05 \
    --calibration-budget-fraction 0.50 \
    --tail-quality-quantile 0.90 \
    --tail-penalty-weight 4.0 \
    --folds 5
