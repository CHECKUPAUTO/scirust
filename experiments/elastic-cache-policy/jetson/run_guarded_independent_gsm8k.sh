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
SOURCE_REPORT="${SOURCE_REPORT:-}"
GUARDED_SEED="${GUARDED_SEED:-20260808}"
GUARDED_SAMPLES="${GUARDED_SAMPLES:-60}"
GUARDED_WARMUP_SAMPLES="${GUARDED_WARMUP_SAMPLES:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/results/guarded-independent-$(date -u +%Y%m%dT%H%M%SZ)}"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
command -v docker >/dev/null || fail "docker introuvable"
command -v python3 >/dev/null || fail "python3 introuvable"
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
    POLICY_REPORT="$(
        find "$WORK_ROOT/results" -type f \
            -name dream_robust_cross_validation_report.json \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )"
fi
if [ -z "$GUARD_SELECTION" ]; then
    GUARD_SELECTION="$(
        find "$WORK_ROOT/results" -type f \
            -name dream_trajectory_stability_guard_selection.json \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )"
fi
if [ -z "$SOURCE_REPORT" ]; then
    SOURCE_REPORT="$(
        find "$WORK_ROOT/results" -type f \
            -name dream_frozen_ensemble_gsm8k_report.json \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )"
fi
[ -n "$POLICY_REPORT" ] && [ -f "$POLICY_REPORT" ] || \
    fail "rapport de politiques figées introuvable"
[ -n "$GUARD_SELECTION" ] && [ -f "$GUARD_SELECTION" ] || \
    fail "sélection de garde introuvable"
[ -n "$SOURCE_REPORT" ] && [ -f "$SOURCE_REPORT" ] || \
    fail "rapport GSM8K enregistré introuvable"
for path in "$POLICY_REPORT" "$GUARD_SELECTION" "$SOURCE_REPORT"; do
    case "$path" in
        "$WORK_ROOT"/*) ;;
        *) fail "les rapports doivent se trouver sous $WORK_ROOT" ;;
    esac
done
POLICY_RELATIVE="${POLICY_REPORT#"$WORK_ROOT"/}"
GUARD_RELATIVE="${GUARD_SELECTION#"$WORK_ROOT"/}"
SOURCE_RELATIVE="${SOURCE_REPORT#"$WORK_ROOT"/}"

if [ ! -d "$ELASTIC_ROOT/.git" ]; then
    git clone https://github.com/VILA-Lab/Elastic-Cache.git "$ELASTIC_ROOT"
else
    git -C "$ELASTIC_ROOT" fetch origin --prune
fi
git -C "$ELASTIC_ROOT" reset --hard origin/main
git -C "$ELASTIC_ROOT" clean -fdx
ELASTIC_COMMIT="$(git -C "$ELASTIC_ROOT" rev-parse HEAD)"
python3 \
    "$SCIRUST_RUN/experiments/elastic-cache-policy/real/dream/patch_elastic_cache.py" \
    "$ELASTIC_ROOT"

if [ ! -d "$GSM8K_ROOT/.git" ]; then
    git clone https://github.com/openai/grade-school-math.git "$GSM8K_ROOT"
else
    git -C "$GSM8K_ROOT" fetch origin --prune
fi
git -C "$GSM8K_ROOT" reset --hard origin/master
git -C "$GSM8K_ROOT" clean -fdx
GSM8K_COMMIT="$(git -C "$GSM8K_ROOT" rev-parse HEAD)"
GSM8K_TEST="$GSM8K_ROOT/grade_school_math/data/test.jsonl"
[ -f "$GSM8K_TEST" ] || fail "fichier GSM8K test absent"

if ! docker image inspect "$PYTORCH_IMAGE" >/dev/null 2>&1; then
    docker pull "$PYTORCH_IMAGE"
fi
IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$PYTORCH_IMAGE" 2>/dev/null || true)"

cat > "$OUTPUT_ROOT/guarded_independent_manifest.json" <<EOF
{
  "schema_version": 1,
  "policy_report": "$POLICY_REPORT",
  "guard_selection": "$GUARD_SELECTION",
  "source_registered_report": "$SOURCE_REPORT",
  "seed": $GUARDED_SEED,
  "samples": $GUARDED_SAMPLES,
  "warmup_samples": $GUARDED_WARMUP_SAMPLES,
  "sequence": "ABBA/BAAB",
  "repeats_per_mode_per_question": 2,
  "registered_indices_excluded": true,
  "elastic_cache_commit": "$ELASTIC_COMMIT",
  "gsm8k_commit": "$GSM8K_COMMIT",
  "container_image": "$PYTORCH_IMAGE",
  "container_digest": "${IMAGE_DIGEST:-unknown}",
  "quality_noninferiority_margin": 0.05,
  "minimum_latency_improvement": 0.005,
  "minimum_refresh_cost_improvement": 0.005,
  "bootstrap_samples": 10000,
  "policy_or_guard_fitting_on_evaluation_prompts": false
}
EOF

printf 'Politique figée: %s\n' "$POLICY_REPORT"
printf 'Garde figée: %s\n' "$GUARD_SELECTION"
printf 'Rapport enregistré préservé: %s\n' "$SOURCE_REPORT"
printf 'Validation: seed=%s échantillons=%s warmup=%s\n' \
    "$GUARDED_SEED" "$GUARDED_SAMPLES" "$GUARDED_WARMUP_SAMPLES"
printf 'Séquence: ABBA/BAAB sur de nouveaux indices GSM8K\n'
printf 'Elastic-Cache commit: %s\n' "$ELASTIC_COMMIT"
printf 'GSM8K commit: %s\n' "$GSM8K_COMMIT"
printf 'Image NVIDIA: %s\n' "${IMAGE_DIGEST:-$PYTORCH_IMAGE}"

set +e
docker run --rm \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    --ipc=host \
    --shm-size=16g \
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
    -w /workspace \
    "$PYTORCH_IMAGE" \
    bash -lc "
set -euo pipefail

python3 - <<'PY'
import torch
print('container_torch=', torch.__version__)
print('cuda_runtime=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('Le conteneur NVIDIA ne voit pas le GPU Thor')
print('device=', torch.cuda.get_device_name(0))
print('capability=', torch.cuda.get_device_capability(0))
a = torch.randn((256, 256), device='cuda', dtype=torch.bfloat16)
b = torch.randn((256, 256), device='cuda', dtype=torch.bfloat16)
value = (a @ b).float().abs().mean().item()
torch.cuda.synchronize()
print('bf16_gemm_checksum=', float(value))
PY

if [ ! -x /workspace/work/end-to-end-venv/bin/python ]; then
    python3 -m venv --system-site-packages /workspace/work/end-to-end-venv
fi
source /workspace/work/end-to-end-venv/bin/activate
PIP_CONSTRAINT=/dev/null python -m pip install --upgrade \
    'pip<27' 'setuptools<80' 'wheel<0.47' 'packaging<=25.0'
PIP_CONSTRAINT=/dev/null python -m pip install \
    'transformers==4.49.0' \
    'accelerate==0.34.2' \
    hf_xet \
    hf_transfer \
    einops

python /workspace/scirust/experiments/elastic-cache-policy/real/dream/benchmark_guarded_independent.py \
    --elastic-cache /workspace/Elastic-Cache \
    --policy-report '/workspace/work/$POLICY_RELATIVE' \
    --guard-selection '/workspace/work/$GUARD_RELATIVE' \
    --source-report '/workspace/work/$SOURCE_RELATIVE' \
    --gsm8k-test /workspace/grade-school-math/grade_school_math/data/test.jsonl \
    --output /workspace/output/dream_guarded_independent_gsm8k_report.json \
    --model Dream-org/Dream-v0-Instruct-7B \
    --seed '$GUARDED_SEED' \
    --samples '$GUARDED_SAMPLES' \
    --warmup-samples '$GUARDED_WARMUP_SAMPLES' \
    --max-new-tokens 128 \
    --window-length 32 \
    --decoding-threshold 0.9 \
    --vote-threshold 3 \
    --quality-noninferiority-margin 0.05 \
    --minimum-latency-improvement 0.005 \
    --minimum-refresh-cost-improvement 0.005 \
    --bootstrap-samples 10000 \
    --dtype bfloat16
"
STATUS=$?
set -e

printf '\nRésultats validation indépendante de la garde: %s\n' "$OUTPUT_ROOT"
if [ -f "$OUTPUT_ROOT/dream_guarded_independent_gsm8k_report.json" ]; then
    cat "$OUTPUT_ROOT/dream_guarded_independent_gsm8k_report.json"
fi
exit "$STATUS"
