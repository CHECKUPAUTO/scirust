#!/usr/bin/env bash
set -euo pipefail

SCIRUST_SOURCE="${SCIRUST_SOURCE:-$HOME/scirust}"
WORK_ROOT="${WORK_ROOT:-$HOME/.local/share/scirust/dream-policy-proof}"
SCIRUST_RUN="$WORK_ROOT/scirust-policy"
ELASTIC_ROOT="$WORK_ROOT/Elastic-Cache"
VENV="$WORK_ROOT/venv"
OUTPUT_ROOT="$WORK_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)"

fail() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || fail "git introuvable"
command -v python3 >/dev/null || fail "python3 introuvable"
command -v cargo >/dev/null || fail "cargo introuvable"
command -v rustup >/dev/null || fail "rustup introuvable"
[ -d "$SCIRUST_SOURCE/.git" ] || fail "dépôt SciRust absent: $SCIRUST_SOURCE"
[ "$(uname -m)" = "aarch64" ] || printf 'AVERTISSEMENT: architecture détectée: %s\n' "$(uname -m)"

mkdir -p "$WORK_ROOT" "$OUTPUT_ROOT"

git -C "$SCIRUST_SOURCE" fetch origin --prune
if [ ! -d "$SCIRUST_RUN/.git" ]; then
    git clone --shared "$SCIRUST_SOURCE" "$SCIRUST_RUN"
fi
git -C "$SCIRUST_RUN" fetch origin --prune
git -C "$SCIRUST_RUN" checkout -B research/elastic-cache-policy-discovery \
    origin/research/elastic-cache-policy-discovery

if [ ! -d "$ELASTIC_ROOT/.git" ]; then
    git clone https://github.com/VILA-Lab/Elastic-Cache.git "$ELASTIC_ROOT"
else
    git -C "$ELASTIC_ROOT" fetch origin --prune
    git -C "$ELASTIC_ROOT" checkout -B main origin/main
fi

if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install \
    'transformers==4.49.0' \
    'accelerate==0.34.2' \
    'hf_xet' \
    'hf_transfer' \
    'einops'

python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda_runtime=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA indisponible dans le venv; ne pas installer un wheel torch générique")
print("device=", torch.cuda.get_device_name(0))
print("bf16_supported=", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 CUDA indisponible sur ce runtime Jetson")
PY

rustup toolchain install nightly-2026-07-02 --profile minimal

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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
