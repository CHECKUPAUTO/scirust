# scirust-runtime

Experimental **deterministic inference runtime** built on a frozen forward subset
of SciRust. Not a general training framework and not a competitor to PyTorch /
Burn / candle — a focused artifact demonstrating three first-class guarantees for
edge and regulated inference: **bit-exact determinism, bounded latency,
auditability.**

## Scope

- **Forward inference only.** Training is offline tooling (train_artifact); the
  runtime (eval_artifact) only loads a frozen artifact and runs forward.
- Current layer set: Linear + ReLU (MLP). Conv/transformer paths exist in the
  core but are not yet exercised by the runtime.
- Depends on scirust-core by path; the golden-fingerprint test acts as a
  regression lock against core drift.

## The three guarantees (measured)

| Guarantee | Contract | Measured evidence (MNIST MLP 784-256-10, aarch64) |
|---|---|---|
| **#1 Determinism** | Bit-exact output for a fixed (binary, target), independent of thread count and across process restarts. | 5120 bit-exact comparisons, 0 divergences; logit fingerprint 0xde2d807686e4b47e stable across RAYON_NUM_THREADS in {1,2,4,8,16,64} and process restarts. Reload bit-exact: state_dict to SRT1 to fresh model to load to forward reproduces the fingerprint exactly. |
| **#2 Bounded latency** | Predictable per-request latency with a tight tail. | batch=1: p50 126us, p99 145us, p99.9 151us, p99/p50 = 1.15x; thread-invariant. batch=64 throughput scales 23k to 81k samples/s (1 to 8 threads). |
| **#3 Auditability** | Frozen artifact has a stable identity; inference is reproducible. | SRT1 weight file with sorted keys gives deterministic on-disk bytes (artifact hash 0xa812d335f822046b). Trained model: 97.73% accuracy (9773/10000), test-logit fingerprint 0xc96d25fa658f5611 stable inter-process. |

### Honest boundaries

- Determinism is bit-exact for a fixed compiled artifact on a given
  architecture. Cross-architecture bit-exactness (x86 vs aarch64: FMA, SIMD
  width, rounding) is out of scope by design — the audit model is "ship a
  pinned artifact, replay it forever on that target."
- Absolute batch=1 latency (~126us) is dominated by fixed per-call overhead
  (tape allocation, op recording), not compute. At sub-millisecond scale this is
  acceptable; no allocation arena was built because the data does not justify it.
- Thread count is a throughput knob (batched workloads); it does not affect
  single-request latency.

## SRT1 weight format

Deterministic, byte-stable on disk (enables artifact hashing):

    magic   : b"SRT1"            (4 bytes)
    count   : u32 LE             (number of tensors)
    per tensor, keys sorted ascending:
      key_len : u32 LE, key bytes (UTF-8)
      rows    : u32 LE
      cols    : u32 LE
      data_len: u64 LE, then data_len * f32 LE

## Usage

    # 1. Offline: train and freeze a model -> mnist_mlp.srt
    cargo run --release --bin train_artifact

    # 2. Runtime: load the frozen artifact, evaluate accuracy + fingerprint
    cargo run --release --bin eval_artifact
    #    -> Accuracy 97.73%, fingerprint 0xc96d25fa658f5611 (stable across reruns)

    # 3. Latency characterization
    cargo run --release --bin bench_latency

    # Golden persistence lock (default bin): reload is bit-exact
    cargo run --release

MNIST_DIR overrides the dataset path (default /root/scirust/data/mnist).
MNIST_MAX_TRAIN caps training set size.

## Status

Build-isolated crate (own [workspace]); to be promoted to a workspace member at
integration. Part of the SciRust research artifact, documenting human-directed
construction of a deterministic inference runtime over a churning research core.
