#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
ELASTIC_ROOT="$WORK_ROOT/Elastic-Cache"
GSM8K_ROOT="$WORK_ROOT/grade-school-math"
POLICY_BRANCH="research/elastic-cache-policy-discovery"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
POLICY_REPORT="${POLICY_REPORT:-}"
GUARD_SELECTION="${GUARD_SELECTION:-}"
TRAJECTORY_SEED="${TRAJECTORY_SEED:-20260809}"
TRAJECTORY_SAMPLES="${TRAJECTORY_SAMPLES:-40}"
TRAJECTORY_WARMUP_SAMPLES="${TRAJECTORY_WARMUP_SAMPLES:-2}"
MAX_CANDIDATES="${MAX_CANDIDATES:-4}"
TRAJECTORY_POLICY_SEED="${TRAJECTORY_POLICY_SEED:-20260810}"
CRF_EPOCHS="${CRF_EPOCHS:-300}"
NSGA_POPULATION="${NSGA_POPULATION:-120}"
NSGA_GENERATIONS="${NSGA_GENERATIONS:-80}"
MINIMUM_HOLDOUT_COVERAGE="${MINIMUM_HOLDOUT_COVERAGE:-0.02}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/results/trajectory-development-$(date -u +%Y%m%dT%H%M%SZ)}"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
command -v docker >/dev/null || fail "docker introuvable"
command -v cargo >/dev/null || fail "cargo introuvable"
[ -d "$SCIRUST_SOURCE/.git" ] || fail "dépôt SciRust absent: $SCIRUST_SOURCE"
docker info >/dev/null 2>&1 || fail "daemon Docker inaccessible"
mkdir -p "$WORK_ROOT" "$OUTPUT_ROOT" "$WORK_ROOT/huggingface"

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
    POLICY_REPORT="$(find "$WORK_ROOT/results" -type f -name dream_robust_cross_validation_report.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [ -z "$GUARD_SELECTION" ]; then
    GUARD_SELECTION="$(find "$WORK_ROOT/results" -type f -name dream_trajectory_stability_guard_selection.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
[ -n "$POLICY_REPORT" ] && [ -f "$POLICY_REPORT" ] || fail "rapport de politique introuvable"
[ -n "$GUARD_SELECTION" ] && [ -f "$GUARD_SELECTION" ] || fail "sélection de garde introuvable"
for path in "$POLICY_REPORT" "$GUARD_SELECTION"; do
    case "$path" in "$WORK_ROOT"/*) ;; *) fail "les rapports doivent être sous $WORK_ROOT" ;; esac
done
POLICY_RELATIVE="${POLICY_REPORT#"$WORK_ROOT"/}"
GUARD_RELATIVE="${GUARD_SELECTION#"$WORK_ROOT"/}"

if [ ! -d "$ELASTIC_ROOT/.git" ]; then
    git clone https://github.com/VILA-Lab/Elastic-Cache.git "$ELASTIC_ROOT"
else
    git -C "$ELASTIC_ROOT" fetch origin --prune
fi
git -C "$ELASTIC_ROOT" reset --hard origin/main
git -C "$ELASTIC_ROOT" clean -fdx
ELASTIC_COMMIT="$(git -C "$ELASTIC_ROOT" rev-parse HEAD)"
python3 "$SCIRUST_RUN/experiments/elastic-cache-policy/real/dream/patch_elastic_cache.py" "$ELASTIC_ROOT"
python3 "$SCIRUST_RUN/experiments/elastic-cache-policy/real/dream/patch_trajectory_probe.py" "$ELASTIC_ROOT"

if [ ! -d "$GSM8K_ROOT/.git" ]; then
    git clone https://github.com/openai/grade-school-math.git "$GSM8K_ROOT"
else
    git -C "$GSM8K_ROOT" fetch origin --prune
fi
git -C "$GSM8K_ROOT" reset --hard origin/master
git -C "$GSM8K_ROOT" clean -fdx
GSM8K_COMMIT="$(git -C "$GSM8K_ROOT" rev-parse HEAD)"
GSM8K_TRAIN="$GSM8K_ROOT/grade_school_math/data/train.jsonl"
[ -f "$GSM8K_TRAIN" ] || fail "GSM8K train absent"

if ! docker image inspect "$PYTORCH_IMAGE" >/dev/null 2>&1; then
    docker pull "$PYTORCH_IMAGE"
fi
IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$PYTORCH_IMAGE" 2>/dev/null || true)"
cat > "$OUTPUT_ROOT/trajectory_development_manifest.json" <<EOF
{
  "schema_version": 1,
  "status": "single_skip_trajectory_development_manifest",
  "split": "gsm8k_train",
  "seed": $TRAJECTORY_SEED,
  "prompts": $TRAJECTORY_SAMPLES,
  "warmup_prompts": $TRAJECTORY_WARMUP_SAMPLES,
  "maximum_candidates_per_prompt": $MAX_CANDIDATES,
  "baseline": "always refresh with eligible-candidate enumeration",
  "intervention": "exactly one selected skip",
  "prior_test_prompts_read": false,
  "elastic_cache_commit": "$ELASTIC_COMMIT",
  "gsm8k_commit": "$GSM8K_COMMIT",
  "container_image": "$PYTORCH_IMAGE",
  "container_digest": "${IMAGE_DIGEST:-unknown}",
  "offline_policy_seed": $TRAJECTORY_POLICY_SEED,
  "offline_components": [
    "scirust-causal invariant causal prediction",
    "scirust-sequential linear-chain CRF",
    "scirust-gp Matern-5/2 uncertainty",
    "scirust-evo NSGA-II",
    "scirust-symreg symbolic surrogate"
  ],
  "minimum_internal_holdout_coverage": $MINIMUM_HOLDOUT_COVERAGE
}
EOF

printf 'Développement trajectoire: seed=%s prompts=%s candidats<=%s\n' "$TRAJECTORY_SEED" "$TRAJECTORY_SAMPLES" "$MAX_CANDIDATES"
printf 'Split exclusivement utilisé: GSM8K train\n'
printf 'Sortie: %s\n' "$OUTPUT_ROOT"

docker run --rm \
    --runtime=nvidia --gpus all --network=host --ipc=host --shm-size=16g \
    -e HF_HOME=/workspace/work/huggingface \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$SCIRUST_RUN:/workspace/scirust" \
    -v "$ELASTIC_ROOT:/workspace/Elastic-Cache" \
    -v "$GSM8K_ROOT:/workspace/grade-school-math:ro" \
    -v "$WORK_ROOT:/workspace/work" \
    -v "$OUTPUT_ROOT:/workspace/output" \
    -w /workspace "$PYTORCH_IMAGE" bash -lc "
set -euo pipefail
if [ ! -x /workspace/work/end-to-end-venv/bin/python ]; then
    python3 -m venv --system-site-packages /workspace/work/end-to-end-venv
fi
source /workspace/work/end-to-end-venv/bin/activate
PIP_CONSTRAINT=/dev/null python -m pip install --upgrade 'pip<27' 'setuptools<80' 'wheel<0.47' 'packaging<=25.0'
PIP_CONSTRAINT=/dev/null python -m pip install 'transformers==4.49.0' 'accelerate==0.34.2' hf_xet hf_transfer einops
python /workspace/scirust/experiments/elastic-cache-policy/real/dream/collect_trajectory_branch_dataset.py \
    --elastic-cache /workspace/Elastic-Cache \
    --policy-report '/workspace/work/$POLICY_RELATIVE' \
    --guard-selection '/workspace/work/$GUARD_RELATIVE' \
    --gsm8k-train /workspace/grade-school-math/grade_school_math/data/train.jsonl \
    --output-dir /workspace/output \
    --model Dream-org/Dream-v0-Instruct-7B \
    --seed '$TRAJECTORY_SEED' \
    --samples '$TRAJECTORY_SAMPLES' \
    --warmup-samples '$TRAJECTORY_WARMUP_SAMPLES' \
    --max-candidates '$MAX_CANDIDATES' \
    --max-new-tokens 128 \
    --window-length 32 \
    --decoding-threshold 0.9 \
    --vote-threshold 3 \
    --dtype bfloat16
"

DATASET="$OUTPUT_ROOT/dream_single_skip_trajectory_candidates.jsonl"
POLICY_OUTPUT="$OUTPUT_ROOT/dream_trajectory_policy_development_report.json"
[ -s "$DATASET" ] || fail "le collecteur n'a produit aucun candidat de trajectoire"

printf '\nDécouverte causale et séquentielle hors ligne\n'
cargo +nightly-2026-07-02 run --release \
    --manifest-path "$SCIRUST_RUN/experiments/elastic-cache-policy/Cargo.toml" \
    --bin trajectory_policy_discovery -- \
    --dataset "$DATASET" \
    --output "$POLICY_OUTPUT" \
    --seed "$TRAJECTORY_POLICY_SEED" \
    --crf-epochs "$CRF_EPOCHS" \
    --crf-learning-rate 0.03 \
    --crf-l2 0.002 \
    --nsga-population "$NSGA_POPULATION" \
    --nsga-generations "$NSGA_GENERATIONS" \
    --minimum-holdout-coverage "$MINIMUM_HOLDOUT_COVERAGE"

printf '\nRésultats de développement trajectoire: %s\n' "$OUTPUT_ROOT"
printf '\nRésumé des branches unitaires:\n'
cat "$OUTPUT_ROOT/dream_single_skip_trajectory_report.json"
printf '\nPolitique causale/séquentielle fail-closed:\n'
cat "$POLICY_OUTPUT"
