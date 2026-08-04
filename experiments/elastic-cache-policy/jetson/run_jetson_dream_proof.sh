#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
ELASTIC_ROOT="$WORK_ROOT/Elastic-Cache"
VENV="$WORK_ROOT/venv"
OUTPUT_ROOT="$WORK_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)"
POLICY_BRANCH="research/elastic-cache-policy-discovery"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
command -v python3 >/dev/null || fail "python3 introuvable"
command -v cargo >/dev/null || fail "cargo introuvable"
command -v rustup >/dev/null || fail "rustup introuvable"
[ -d "$SCIRUST_SOURCE/.git" ] || fail "dépôt SciRust absent: $SCIRUST_SOURCE"
[ "$(uname -m)" = "aarch64" ] || printf 'AVERTISSEMENT: architecture détectée: %s\n' "$(uname -m)"

mkdir -p "$WORK_ROOT" "$OUTPUT_ROOT"

SCIRUST_REMOTE="$(git -C "$SCIRUST_SOURCE" remote get-url origin)"
git -C "$SCIRUST_SOURCE" fetch origin --prune
if [ ! -d "$SCIRUST_RUN/.git" ]; then
    git clone --shared --no-checkout "$SCIRUST_SOURCE" "$SCIRUST_RUN"
fi
# A shared local clone initially points origin at SCIRUST_SOURCE. Replace it with
# the source repository's actual GitHub remote so remote-only research branches
# are fetched correctly.
git -C "$SCIRUST_RUN" remote set-url origin "$SCIRUST_REMOTE"
git -C "$SCIRUST_RUN" fetch --prune origin \
    "+refs/heads/$POLICY_BRANCH:refs/remotes/origin/$POLICY_BRANCH"
git -C "$SCIRUST_RUN" checkout -B "$POLICY_BRANCH" \
    "refs/remotes/origin/$POLICY_BRANCH"

if [ ! -d "$ELASTIC_ROOT/.git" ]; then
    git clone https://github.com/VILA-Lab/Elastic-Cache.git "$ELASTIC_ROOT"
else
    git -C "$ELASTIC_ROOT" fetch origin --prune
    git -C "$ELASTIC_ROOT" checkout -B main origin/main
fi

# The host currently contains a CPU-only PyTorch used by other software. Never
# inherit it: the proof needs an isolated CUDA 13 ARM64 wheel and must not modify
# TensorRT-LLM's global Python environment.
if [ -x "$VENV/bin/python" ]; then
    if ! "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.version.cuda and torch.cuda.is_available() else 1)
PY
    then
        printf 'Recréation du venv expérimental CPU-only: %s\n' "$VENV"
        rm -rf "$VENV"
    fi
fi

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade 'pip<27' 'setuptools<80' wheel
    "$VENV/bin/python" -m pip install \
        --index-url "$TORCH_INDEX" \
        'torch==2.10.0+cu130'
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install \
    'transformers==4.49.0' \
    'accelerate==0.34.2' \
    'hf_xet' \
    'hf_transfer' \
    'einops'

python - <<'PY'
import torch
print("torch=", torch.__version__)
print("torch_file=", torch.__file__)
print("cuda_runtime=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.version.cuda or not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA 13 indisponible dans le venv isolé")
print("device=", torch.cuda.get_device_name(0))
print("capability=", torch.cuda.get_device_capability(0))
print("bf16_supported=", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 CUDA indisponible sur ce runtime Jetson")
# Validate the GEMM path used by Dream, not merely CUDA device discovery.
a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
b = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
print("bf16_gemm_checksum=", float(c.float().abs().mean().item()))
PY

rustup toolchain install nightly-2026-07-02 --profile minimal

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

python "$SCIRUST_RUN/experiments/elastic-cache-policy/jetson/dream_jetson_proof.py" \
    --elastic-cache "$ELASTIC_ROOT" \
    --scirust "$SCIRUST_RUN" \
    --output-dir "$OUTPUT_ROOT" \
    --model Dream-org/Dream-v0-Instruct-7B \
    --trajectories 30 \
    --max-new-tokens 64 \
    --window-length 16 \
    --quality-budget 0.05 \
    --steps 1200 \
    --seed 20260804

printf '\nRésultats: %s\n' "$OUTPUT_ROOT"
cat "$OUTPUT_ROOT/dream_real_policy_report.json"
