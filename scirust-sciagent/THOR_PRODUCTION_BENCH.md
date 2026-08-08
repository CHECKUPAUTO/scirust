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

No performance number in this document is assumed or pre-filled. The authoritative
numbers are the lines printed on the Jetson AGX Thor.

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
