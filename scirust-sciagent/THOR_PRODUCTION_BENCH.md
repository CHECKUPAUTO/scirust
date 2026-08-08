# SCIAGENT Thor production benchmark

This benchmark is the hardware gate after Route-B phases B23–B38. It exists to
measure the remaining bottleneck rather than infer it from microbenchmarks.

## Scope

`examples/cuda_production_bench.rs` uses the real `SciAgentConfig::sciagent_350m()`
shape (~304M parameters), defaults to `seq_len=512`, exercises the production
`CudaTrainer::pretrain` path, sweeps true packed batch sizes, and reports both
training throughput and an ETA for one pass over the 1,029,492,639-token v4 corpus.
It then compares the original non-cached CUDA generator with the B31 resident KV
cache and requires greedy token parity.

The authoritative performance record is always the machine-readable stdout from the
Jetson AGX Thor. The verified B49 gate on 2026-08-08 produced the current reference
record below; reruns may replace it when hardware/software changes.

## Verified B49 Thor record — 2026-08-08

Hardware gate: NVIDIA Thor, driver 580.00, compute capability 11.0, CUDA 13.0.
Model shape: **304,088,064 parameters**, vocab 32,768, `d_model=1024`, 24 layers,
16 query heads / 4 KV heads, production context 512.

- training `B8×T512`: **3,947.734 tok/s**;
- v4 corpus size: **1,029,492,639 tokens**;
- estimated one-pass time at that measured rate: **3.018 days**;
- cached decode, prompt 128 + 8 greedy tokens: **33.603 tok/s**;
- naive full-forward decode: **23.473 tok/s**;
- KV-cache speedup: **1.432×**;
- strict greedy token parity: **`true`**.

The same gate also passed `rustfmt`, CUDA SciAgent Clippy and all six CUDA parity
tests, including cached-vs-naive greedy generation and exact optimizer resume.

## Standard run

```bash
cd /root/scirust
SCIAGENT_BENCH_BATCHES=1,2,4,8 \
SCIAGENT_BENCH_SEQ=512 \
SCIAGENT_BENCH_STEPS=8 \
SCIAGENT_BENCH_PROMPT=128 \
SCIAGENT_BENCH_DECODE_NEW=8 \
cargo run -p scirust-sciagent --features cuda --release --example cuda_production_bench \
  2>&1 | tee /tmp/sciagent-thor-production-bench.log
```

If a high batch size exceeds available unified-memory headroom, rerun each batch in
a fresh process so a failing size cannot hide lower-size results:

```bash
for b in 1 2 4 8; do
  SCIAGENT_BENCH_BATCHES="$b" SCIAGENT_BENCH_SEQ=512 SCIAGENT_BENCH_STEPS=8 \
    cargo run -p scirust-sciagent --features cuda --release \
      --example cuda_production_bench 2>&1 | tee "/tmp/sciagent-thor-b${b}.log" || break
done
```

The machine-readable records are prefixed with:

- `SCIAGENT_THOR_TRAIN` — measured production optimizer throughput;
- `SCIAGENT_THOR_PASS` — one-pass ETA for the configured corpus token count;
- `SCIAGENT_THOR_BEST_TRAIN` — best successful batch in the invocation;
- `SCIAGENT_THOR_DECODE` — cached vs naive CUDA decode and greedy parity.

## Profiling the winner

After choosing the fastest batch, profile that *single* configuration rather than
optimizing from intuition. When NVIDIA Nsight Systems is installed on the Thor:

```bash
b=4  # replace with the measured winner
nsys profile --stats=true -o /tmp/sciagent-b${b}-seq512 \
  env SCIAGENT_BENCH_BATCHES="$b" SCIAGENT_BENCH_SEQ=512 SCIAGENT_BENCH_STEPS=4 \
  cargo run -p scirust-sciagent --features cuda --release --example cuda_production_bench
```

The next optimization must follow the profile:

- attention-dominated: fuse QK/softmax/AV (FlashAttention-style) before changing the model;
- allocation/launch-dominated: introduce a reusable CUDA workspace/arena;
- GEMM-dominated with good occupancy: batching has done its job; do not add bespoke kernels without a measured gain;
- decode cache-copy dominated: replace the first B31 grow-by-concatenation cache with preallocated K/V storage.

## Production-training gate

Do **not** start the final long pretraining run from the historical pre-B33 weights as
if they were equivalent. B33 corrected GQA RoPE from full-projection-width frequency
bases to head-local `d_head=64`; those checkpoints were trained under different model
semantics.

Before the final run:

1. all B23+ correctness/CI gates must be green;
2. run the Thor sweep above and choose a measured batch size;
3. run the CUDA parity tests on the Thor, including cached decode and exact optimizer resume;
4. start a new post-B33 checkpoint directory against the full v4 corpus;
5. evaluate with the B36 distributed holdout and `cuda_eval`, not the historical tail split.
