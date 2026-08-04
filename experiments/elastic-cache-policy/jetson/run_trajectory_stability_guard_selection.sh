#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
POLICY_BRANCH="research/elastic-cache-policy-discovery"
POLICY_REPORT="${POLICY_REPORT:-}"
TRACE_PATH="${TRACE_PATH:-}"
OUTPUT="${OUTPUT:-}"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
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

if [ -z "$POLICY_REPORT" ]; then
    POLICY_REPORT="$(
        find "$WORK_ROOT/results" -type f \
            -name dream_robust_cross_validation_report.json \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )"
fi
if [ -z "$TRACE_PATH" ]; then
    TRACE_PATH="$(
        find "$WORK_ROOT/results" -type f \
            -path '*/confirmatory-*/dream_counterfactual_trace.csv' \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )"
fi
[ -n "$POLICY_REPORT" ] && [ -f "$POLICY_REPORT" ] || \
    fail "rapport de politiques figées introuvable"
[ -n "$TRACE_PATH" ] && [ -f "$TRACE_PATH" ] || \
    fail "trace confirmatoire introuvable"

if [ -z "$OUTPUT" ]; then
    OUTPUT="$(dirname "$TRACE_PATH")/dream_trajectory_stability_guard_selection.json"
fi

printf 'Sélection hors ligne sans recalcul GPU\n'
printf 'Politique: %s\n' "$POLICY_REPORT"
printf 'Trace de développement: %s\n' "$TRACE_PATH"
python3 \
    "$SCIRUST_RUN/experiments/elastic-cache-policy/jetson/select_trajectory_stability_guard.py" \
    --policy-report "$POLICY_REPORT" \
    --trace "$TRACE_PATH" \
    --output "$OUTPUT" \
    --vote-threshold 3 \
    --quality-budget 0.01 \
    --minimum-mean-compute-improvement 0.008 \
    --tail-quality-quantile 0.90

printf '\nGarde sélectionnée: %s\n' "$OUTPUT"
