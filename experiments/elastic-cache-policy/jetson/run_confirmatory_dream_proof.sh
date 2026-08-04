#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
ELASTIC_ROOT="$WORK_ROOT/Elastic-Cache"
POLICY_BRANCH="research/elastic-cache-policy-discovery"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
POLICY_REPORT="${POLICY_REPORT:-}"
CONFIRMATORY_SEED="${CONFIRMATORY_SEED:-20260805}"
CONFIRMATORY_TRAJECTORIES="${CONFIRMATORY_TRAJECTORIES:-60}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/results/confirmatory-$(date -u +%Y%m%dT%H%M%SZ)}"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
command -v docker >/dev/null || fail "docker introuvable"
command -v python3 >/dev/null || fail "python3 introuvable"
[ -d "$SCIRUST_SOURCE/.git" ] || fail "dépôt SciRust absent: $SCIRUST_SOURCE"
[ "$(uname -m)" = "aarch64" ] || printf 'AVERTISSEMENT: architecture détectée: %s\n' "$(uname -m)"
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
[ -n "$POLICY_REPORT" ] && [ -f "$POLICY_REPORT" ] || \
    fail "rapport de validation croisée robuste introuvable"

if [ ! -d "$ELASTIC_ROOT/.git" ]; then
    git clone https://github.com/VILA-Lab/Elastic-Cache.git "$ELASTIC_ROOT"
else
    git -C "$ELASTIC_ROOT" fetch origin --prune
    git -C "$ELASTIC_ROOT" checkout -B main origin/main
fi

if ! docker image inspect "$PYTORCH_IMAGE" >/dev/null 2>&1; then
    docker pull "$PYTORCH_IMAGE"
fi
IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$PYTORCH_IMAGE" 2>/dev/null || true)"
printf 'Politique figée: %s\n' "$POLICY_REPORT"
printf 'Trace confirmatoire: seed=%s trajectoires=%s\n' \
    "$CONFIRMATORY_SEED" "$CONFIRMATORY_TRAJECTORIES"
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

if [ ! -x /workspace/work/confirmatory-venv/bin/python ]; then
    python3 -m venv --system-site-packages /workspace/work/confirmatory-venv
fi
source /workspace/work/confirmatory-venv/bin/activate
PIP_CONSTRAINT=/dev/null python -m pip install --upgrade \
    'pip<27' 'setuptools<80' 'wheel<0.47' 'packaging<=25.0'
PIP_CONSTRAINT=/dev/null python -m pip install \
    'transformers==4.49.0' \
    'accelerate==0.34.2' \
    hf_xet \
    hf_transfer \
    einops

python /workspace/scirust/experiments/elastic-cache-policy/jetson/dream_trace_only.py \
    --elastic-cache /workspace/Elastic-Cache \
    --output-dir /workspace/output \
    --model Dream-org/Dream-v0-Instruct-7B \
    --trajectories '$CONFIRMATORY_TRAJECTORIES' \
    --max-new-tokens 64 \
    --window-length 16 \
    --seed '$CONFIRMATORY_SEED'
"
TRACE_STATUS=$?
set -e
[ "$TRACE_STATUS" -eq 0 ] || fail "collecte de la trace confirmatoire échouée"

TRACE_PATH="$OUTPUT_ROOT/dream_counterfactual_trace.csv"
TRACE_MANIFEST="$OUTPUT_ROOT/dream_trace_manifest.json"
[ -f "$TRACE_PATH" ] || fail "trace confirmatoire absente"
[ -f "$TRACE_MANIFEST" ] || fail "manifest confirmatoire absent"

set +e
python3 \
    "$SCIRUST_RUN/experiments/elastic-cache-policy/jetson/frozen_policy_confirmatory.py" \
    --policy-report "$POLICY_REPORT" \
    --trace "$TRACE_PATH" \
    --trace-manifest "$TRACE_MANIFEST" \
    --output-dir "$OUTPUT_ROOT" \
    --vote-threshold 3 \
    --quality-budget 0.05 \
    --tail-quality-quantile 0.90 \
    --minimum-compute-improvement 0.005 \
    --bootstrap-samples 10000 \
    --bootstrap-seed "$CONFIRMATORY_SEED"
STATUS=$?
set -e

printf '\nRésultats confirmatoires: %s\n' "$OUTPUT_ROOT"
if [ -f "$OUTPUT_ROOT/dream_frozen_policy_confirmatory_report.json" ]; then
    cat "$OUTPUT_ROOT/dream_frozen_policy_confirmatory_report.json"
fi
exit "$STATUS"
