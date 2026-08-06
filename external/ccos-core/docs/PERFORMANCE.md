# Performance & bare-metal notes

**The honest framing first.** CCOS is *already* frugal by construction: building
the region map over its own 705-node tree takes ≈20 ms, and a recall / checkpoint /
compaction is microseconds-to-milliseconds. In an agent loop that work sits between
LLM inferences measured in **seconds**, so the CCOS kernel is **< 1 %** of the cycle.
By Amdahl's law, micro-optimising it cannot move the end-to-end needle. So this page
is deliberately short, and most "extreme config" knobs are explicitly **not worth
it** for this workload.

What *is* worth doing falls in two buckets: a **durability** guarantee (correctness),
and **reproducible measurement** (so the paper's numbers are trustworthy).

## 1. Durable checkpoints (default)

`ccos memory` and `ccos mcp` persist the snapshot (`workspace.ccos`) and the op-log
(`workspace.ccos.oplog`) **durably and atomically**: each write goes to a temp
sibling, is `fsync`-ed, then atomically renamed over the target, and the directory is
`fsync`-ed (`util::write_durable`). A plain `std::fs::write` only reaches the kernel
page cache, so a power loss or a killed daemon could leave a truncated/corrupt file —
which would break CCOS's "replayable after a crash" guarantee. The cost is one
`fsync` per checkpoint, negligible at inference cadence. This is **on by default**; the
guarantee is unconditional, not a flag.

## 2. Reproducible measurement on the Jetson

To make benchmark numbers (token counts, microbenchmark timings, inference latency)
stable run-to-run, pin the board to a fixed max-clock state — this kills
frequency/thermal jitter. On a Jetson AGX Thor (Tegra SoC, unified memory) the
controls are `nvpmodel` / `jetson_clocks` (there is no NUMA, and `nvidia-smi`
clock-locking does not apply):

```bash
sudo bash scripts/jetson_repro_env.sh          # nvpmodel -m 0 + jetson_clocks + governor=performance
# ... run cargo run --release -- eval / experiment / benchmark ...
sudo bash scripts/jetson_repro_env.sh --restore
```

This is for **measurement only** (max power, no thermal headroom) — it does not speed
up the kernel, it removes variance from the numbers.

## 3. Opt-in build knobs (measure, don't assume)

Both are free to try and almost certainly negligible for CCOS — provided here so you
can A/B them rather than guess.

- **`target-cpu=native`** — let the compiler use the exact ISA of the build host
  (e.g. ARMv9/SVE on the Thor). **Local builds only** — never commit it as a default,
  it produces a binary that won't run on a different CPU (breaks portability and
  cross-machine reproducibility):

  ```bash
  RUSTFLAGS="-C target-cpu=native" cargo build --release
  ```

- **`mimalloc`** — a drop-in global allocator, behind an off-by-default feature:

  ```bash
  cargo build --release --features mimalloc
  ```

A/B them with an existing workload and trust the measurement:

```bash
cargo build --release
RUSTFLAGS="-C target-cpu=native" cargo build --release --features mimalloc \
  --target-dir target/native
hyperfine -w2 \
  './target/release/ccos analyze src --out /tmp/a.json' \
  './target/native/release/ccos analyze src --out /tmp/b.json'
```

If the delta is in the noise (it likely is), keep the default build — frugality and
portability beat a sub-1% kernel win.

## Knobs that are *not* worth it here (and why)

| Knob | Why it doesn't help CCOS |
| ---- | ------------------------ |
| CPU isolation (`isolcpus`, cgroup pinning) | The kernel runs sporadically between multi-second inferences; L1/L2 is cold again by the next call regardless. |
| NUMA binding (`numactl`) | The Jetson is a single SoC with unified memory — no NUMA. CCOS's few-MB footprint makes cross-socket cost negligible even on x86. |
| HugePages (2 MB / 1 GB) | The graph is hundreds of nodes / a few MB — it fits in a handful of 4 KB pages; the TLB never sweats. HugePages pay on GB-scale traversals. |
| NVMe I/O scheduler (`none`/`kyber`) | A few-KB op-log written at inference cadence is a handful of writes/min, not thousands of IOPS. |
| `vm.dirty_*` tuning | Delaying flushes to "smooth" writes *reduces* durability — the opposite of §1. We `fsync` instead. |
| CUDA zero-copy / `cudaMemAdvise` | CCOS never touches GPU memory — it emits a **text** context window consumed by the inference server. There is no buffer to DMA. |
