#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
POLICY_BRANCH="research/elastic-cache-policy-discovery"
DATASET="${DATASET:-}"
TRAJECTORY_POLICY_SEED="${TRAJECTORY_POLICY_SEED:-20260810}"
CRF_EPOCHS="${CRF_EPOCHS:-300}"
NSGA_POPULATION="${NSGA_POPULATION:-120}"
NSGA_GENERATIONS="${NSGA_GENERATIONS:-80}"
MINIMUM_HOLDOUT_COVERAGE="${MINIMUM_HOLDOUT_COVERAGE:-0.02}"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
command -v cargo >/dev/null || fail "cargo introuvable"
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

if [ -z "$DATASET" ]; then
    DATASET="$(
        find "$WORK_ROOT/results" -type f \
            -name dream_single_skip_trajectory_candidates.jsonl \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )"
fi
[ -n "$DATASET" ] && [ -f "$DATASET" ] || \
    fail "jeu de branches unitaires introuvable"
case "$DATASET" in
    "$WORK_ROOT"/*) ;;
    *) fail "DATASET doit se trouver sous $WORK_ROOT" ;;
esac

OUTPUT="$(dirname "$DATASET")/dream_trajectory_policy_development_report.json"
printf 'Découverte causale et séquentielle hors ligne: %s\n' "$DATASET"
printf 'Seed=%s CRF=%s NSGA-II=%sx%s couverture minimale=%s\n' \
    "$TRAJECTORY_POLICY_SEED" "$CRF_EPOCHS" \
    "$NSGA_POPULATION" "$NSGA_GENERATIONS" "$MINIMUM_HOLDOUT_COVERAGE"

cargo +nightly-2026-07-02 run --release \
    --manifest-path "$SCIRUST_RUN/experiments/elastic-cache-policy/Cargo.toml" \
    --bin trajectory_policy_discovery -- \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --seed "$TRAJECTORY_POLICY_SEED" \
    --crf-epochs "$CRF_EPOCHS" \
    --crf-learning-rate 0.03 \
    --crf-l2 0.002 \
    --nsga-population "$NSGA_POPULATION" \
    --nsga-generations "$NSGA_GENERATIONS" \
    --minimum-holdout-coverage "$MINIMUM_HOLDOUT_COVERAGE"

printf '\nRapport de politique trajectoire: %s\n' "$OUTPUT"
