# SCIAGENT semantics-v2 production pretraining

This runbook is for the first fresh large-model run after the B33 head-local RoPE
semantics transition and the B49/B50 Thor production validation.

## Why the production run is launched as root on the Thor

The self-hosted GitHub Actions service is intentionally sandboxed by systemd. On the
production Thor, `/root` is replaced inside the runner service namespace by
`/systemd/inaccessible/dir`; the runner therefore cannot see `/root/scirust/data/shards_v4`
or production checkpoints. Keep that isolation. GitHub Actions proves the CUDA code
with temporary data, but the multi-day corpus run is launched from the root host shell.

Do **not** resume any pre-B33 checkpoint. Semantics version 2 uses head-local RoPE and
must start with new weights in a new checkpoint namespace.

## Verified production constants

- model preset: `350m` (`SciAgentConfig::sciagent_350m()`, 304,088,064 parameters);
- corpus v4: 1,030 direct `.bin` shards;
- corpus token count: 1,029,492,639 little-endian `u32` token ids;
- sequence length: 512;
- batch size: 8 (4,096 training tokens/update);
- raw-token-equivalent one-pass budget: `ceil(1,029,492,639 / 4,096) = 251,341` updates;
- peak LR: `3e-4`, minimum LR defaults to 10% of peak;
- warmup: 25,134 updates (10% of the fresh run budget);
- validation fraction: 2%; shuffle enabled;
- exact recovery checkpoint cadence: 6 wall-clock hours; keep the two latest exact
  optimizer checkpoints plus the model-only historical `best/`.

The 251,341-update budget is a raw-token-equivalent pass for capacity planning. Because
2% of deterministic windows are held out for validation, it is not a claim that every
train window is visited exactly once.

## Stage 1 — read-only/fresh preflight and corpus fingerprint

Run from the root shell on the Thor **after B52 is merged into `master`**. The checkpoint
path below is deliberately new. Do not create files in it before the preflight.

```bash
set -euo pipefail
cd /root/scirust

git fetch origin master
git checkout master
git pull --ff-only origin master

mkdir -p logs

export SCIAGENT_CONFIG=350m
export SCIAGENT_SHARDS=/root/scirust/data/shards_v4
export SCIAGENT_CKPT=/root/scirust/checkpoints/bpe350m_v5_semantics_v2
export SCIAGENT_REQUIRE_FRESH=1
export SCIAGENT_EXPECT_SHARDS=1030
export SCIAGENT_EXPECT_CORPUS_TOKENS=1029492639
export SCIAGENT_SEQ=512
export SCIAGENT_BATCH=8
export SCIAGENT_PREFLIGHT_ONLY=1

cargo +nightly-2026-07-02 run \
  -p scirust-sciagent --features cuda --release --example cuda_pretrain \
  | tee logs/sciagent-v2-preflight.log

HASH="$(sed -n 's/.*corpus_fnv64=\([0-9a-fA-F]\{16\}\).*/\1/p' \
  logs/sciagent-v2-preflight.log | tail -n 1)"
test "${#HASH}" -eq 16
printf 'SCIAGENT_V2_CORPUS_FNV64=%s\n' "$HASH"
```

A successful preflight prints one `SCIAGENT_PREFLIGHT_OK` line containing
`semantics=2`, `params=304088064`, `shards=1030`, `corpus_tokens=1029492639`, the
16-digit `corpus_fnv64`, the new checkpoint path, and `fresh=true`. It then exits before
optimizer update 1. If any invariant differs, it exits non-zero.

## Stage 2 — start the fresh run with the fingerprint locked

Only after Stage 1 succeeds, use the exact `HASH` produced there. `SCIAGENT_REQUIRE_FRESH`
stays enabled for the initial launch so an accidental checkpoint collision is fatal.

```bash
set -euo pipefail
cd /root/scirust

HASH="$(sed -n 's/.*corpus_fnv64=\([0-9a-fA-F]\{16\}\).*/\1/p' \
  logs/sciagent-v2-preflight.log | tail -n 1)"
test "${#HASH}" -eq 16

export SCIAGENT_CONFIG=350m
export SCIAGENT_SHARDS=/root/scirust/data/shards_v4
export SCIAGENT_CKPT=/root/scirust/checkpoints/bpe350m_v5_semantics_v2
export SCIAGENT_REQUIRE_FRESH=1
export SCIAGENT_EXPECT_SHARDS=1030
export SCIAGENT_EXPECT_CORPUS_TOKENS=1029492639
export SCIAGENT_EXPECT_CORPUS_FNV64="$HASH"
unset SCIAGENT_PREFLIGHT_ONLY
unset SCIAGENT_MAX_TOKENS
unset SCIAGENT_ALLOW_NONEXACT_RESUME

export SCIAGENT_SEQ=512
export SCIAGENT_BATCH=8
export SCIAGENT_TOTAL_STEPS=251341
export SCIAGENT_WARMUP=25134
export SCIAGENT_LR=0.0003
export SCIAGENT_CLIP=1.0
export SCIAGENT_EPS=0.00001
export SCIAGENT_VAL_FRAC=0.02
export SCIAGENT_SHUFFLE=1
export SCIAGENT_TELEMETRY=25
export SCIAGENT_SAVE_HOURS=6
export SCIAGENT_KEEP=2

nohup cargo +nightly-2026-07-02 run \
  -p scirust-sciagent --features cuda --release --example cuda_pretrain \
  > logs/sciagent-bpe350m-v5-semantics-v2.log 2>&1 &

echo $! | tee logs/sciagent-bpe350m-v5-semantics-v2.pid
```

The run is considered started only after the log confirms all of the following before
normal step telemetry: the 350m config, 1,030 shards, 1,029,492,639 tokens, the expected
FNV64 hash, `seq=512`, `batch=8`, and a fresh start at step 0.

## Exact restart after an interruption

After the first exact `step_N/` exists, a restart is no longer a fresh launch. **Unset
`SCIAGENT_REQUIRE_FRESH`** and point `SCIAGENT_CKPT` at the same namespace. Keep the
corpus identity gates and trajectory settings unchanged. `cuda_pretrain` loads the
newest exact checkpoint and restores model weights, AdamW m/v, bias-correction step,
shuffle/window cursor, LR schedule and the saved run contract; trajectory mismatches
fail closed unless a research-only non-exact override is explicitly requested.
