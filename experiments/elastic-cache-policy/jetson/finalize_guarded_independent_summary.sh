#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
POLICY_BRANCH="research/elastic-cache-policy-discovery"
REPORT="${REPORT:-}"

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

if [ -z "$REPORT" ]; then
    REPORT="$(
        find "$WORK_ROOT/results" -type f \
            -name dream_guarded_independent_gsm8k_report.json \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )"
fi
[ -n "$REPORT" ] && [ -f "$REPORT" ] || \
    fail "rapport de validation indépendante gardée introuvable"
case "$REPORT" in
    "$WORK_ROOT"/*) ;;
    *) fail "REPORT doit se trouver sous $WORK_ROOT" ;;
esac

OUTPUT="$(dirname "$REPORT")/dream_guarded_independent_gsm8k_summary.json"
printf 'Résumé sans recalcul GPU: %s\n' "$REPORT"
python3 \
    "$SCIRUST_RUN/experiments/elastic-cache-policy/jetson/summarize_guarded_independent.py" \
    --report "$REPORT" \
    --output "$OUTPUT"

printf '\nRapport résumé: %s\n' "$OUTPUT"
