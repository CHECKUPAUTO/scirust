#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
ELASTIC_ROOT="$WORK_ROOT/Elastic-Cache"
OUTPUT_ROOT="$WORK_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)"
POLICY_BRANCH="research/elastic-cache-policy-discovery"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
command -v docker >/dev/null || fail "docker introuvable"
[ -d "$SCIRUST_SOURCE/.git" ] || fail "dépôt SciRust absent: $SCIRUST_SOURCE"
[ "$(uname -m)" = "aarch64" ] || printf 'AVERTISSEMENT: architecture détectée: %s\n' "$(uname -m)"
docker info >/dev/null 2>&1 || fail "daemon Docker inaccessible"

mkdir -p "$WORK_ROOT" "$OUTPUT_ROOT" "$WORK_ROOT/huggingface" "$WORK_ROOT/cargo" "$WORK_ROOT/rustup"

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

if [ "${FINALIZE_ONLY:-0}" = "1" ]; then
    LATEST_OUTPUT=""
    for candidate in $(ls -1dt "$WORK_ROOT"/results/* 2>/dev/null || true); do
        if [ -f "$candidate/dream_counterfactual_trace.csv" ] && \
           [ -f "$candidate/scirust_discovery_output.txt" ]; then
            LATEST_OUTPUT="$candidate"
            break
        fi
    done
    [ -n "$LATEST_OUTPUT" ] || fail "aucun résultat Dream récupérable trouvé"
    printf 'Finalisation sans recalcul: %s\n' "$LATEST_OUTPUT"
    set +e
    python3 "$SCIRUST_RUN/experiments/elastic-cache-policy/jetson/finalize_dream_proof.py" \
        --output-dir "$LATEST_OUTPUT" \
        --quality-budget 0.05 \
        --seed 20260804
    STATUS=$?
    set -e
    printf '\nRapport: %s\n' "$LATEST_OUTPUT/dream_real_policy_report.json"
    cat "$LATEST_OUTPUT/dream_real_policy_report.json"
    exit "$STATUS"
fi

if [ ! -d "$ELASTIC_ROOT/.git" ]; then
    git clone https://github.com/VILA-Lab/Elastic-Cache.git "$ELASTIC_ROOT"
else
    git -C "$ELASTIC_ROOT" fetch origin --prune
    git -C "$ELASTIC_ROOT" checkout -B main origin/main
fi

# Remove only the abandoned experimental pip environment. The host Python and
# TensorRT-LLM installation are never modified.
rm -rf "$WORK_ROOT/venv" "$WORK_ROOT/container-venv"

printf 'Téléchargement du conteneur NVIDIA validé pour Thor: %s\n' "$PYTORCH_IMAGE"
docker pull "$PYTORCH_IMAGE"
IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$PYTORCH_IMAGE" 2>/dev/null || true)"
printf 'Image NVIDIA: %s\n' "${IMAGE_DIGEST:-$PYTORCH_IMAGE}"

# NVIDIA's Thor guide validates the NGC PyTorch container path. The container
# supplies the tested PyTorch/cuBLAS stack while all source, cache and result
# directories remain explicit host mounts.
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
    -e CARGO_HOME=/workspace/work/cargo \
    -e RUSTUP_HOME=/workspace/work/rustup \
    -v "$SCIRUST_RUN:/workspace/scirust" \
    -v "$ELASTIC_ROOT:/workspace/Elastic-Cache" \
    -v "$WORK_ROOT:/workspace/work" \
    -v "$OUTPUT_ROOT:/workspace/output" \
    -w /workspace \
    "$PYTORCH_IMAGE" \
    bash -lc '
set -euo pipefail

python3 - <<'"'"'PY'"'"'
import torch
print("container_torch=", torch.__version__)
print("container_torch_file=", torch.__file__)
print("cuda_runtime=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Le conteneur NVIDIA ne voit pas le GPU Thor")
print("device=", torch.cuda.get_device_name(0))
print("capability=", torch.cuda.get_device_capability(0))
print("bf16_supported=", torch.cuda.is_bf16_supported())
a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
b = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
print("bf16_gemm_checksum=", float(c.float().abs().mean().item()))
PY

python3 -m venv --system-site-packages /workspace/work/container-venv
source /workspace/work/container-venv/bin/activate
PIP_CONSTRAINT=/dev/null python -m pip install --upgrade "pip<27" "setuptools<80" "wheel<0.47" "packaging<=25.0"
PIP_CONSTRAINT=/dev/null python -m pip install \
    "transformers==4.49.0" \
    "accelerate==0.34.2" \
    hf_xet \
    hf_transfer \
    einops

python - <<'"'"'PY'"'"'
import torch
import transformers
print("runtime_torch=", torch.__version__)
print("runtime_transformers=", transformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA perdu après création du venv conteneur")
a = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
b = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
value = (a @ b).float().abs().mean().item()
torch.cuda.synchronize()
print("runtime_bf16_gemm_checksum=", float(value))
PY

export PATH="$CARGO_HOME/bin:$PATH"
if ! command -v rustup >/dev/null 2>&1; then
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain none --profile minimal
fi
rustup toolchain install nightly-2026-07-02 --profile minimal

set +e
python /workspace/scirust/experiments/elastic-cache-policy/jetson/dream_jetson_proof.py \
    --elastic-cache /workspace/Elastic-Cache \
    --scirust /workspace/scirust \
    --output-dir /workspace/output \
    --model Dream-org/Dream-v0-Instruct-7B \
    --trajectories 30 \
    --max-new-tokens 64 \
    --window-length 16 \
    --quality-budget 0.05 \
    --steps 1200 \
    --seed 20260804
PROBE_STATUS=$?
set -e

if [ -f /workspace/output/dream_counterfactual_trace.csv ] && \
   [ -f /workspace/output/scirust_discovery_output.txt ]; then
    set +e
    python /workspace/scirust/experiments/elastic-cache-policy/jetson/finalize_dream_proof.py \
        --output-dir /workspace/output \
        --quality-budget 0.05 \
        --seed 20260804
    FINAL_STATUS=$?
    set -e
    exit "$FINAL_STATUS"
fi
exit "$PROBE_STATUS"
'
STATUS=$?
set -e

printf '\nRésultats: %s\n' "$OUTPUT_ROOT"
if [ -f "$OUTPUT_ROOT/dream_real_policy_report.json" ]; then
    cat "$OUTPUT_ROOT/dream_real_policy_report.json"
fi
exit "$STATUS"
