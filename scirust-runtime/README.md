# scirust-runtime

Experimental **deterministic inference runtime** built on a frozen forward subset
of SciRust. Not a general training framework and not a competitor to PyTorch /
Burn / candle — a focused artifact demonstrating three first-class guarantees for
edge and regulated inference: **bit-exact determinism, bounded latency,
auditability** — **generic over architecture** (any supported Sequential rebuilt
from a text manifest + SRT1 weights, no architecture hardcoded in the runtime).

## Scope

- **Forward inference only.** Training is offline tooling (train_artifact); the
  runtime loads a frozen artifact and runs forward.
- **Generic reconstruction.** A text manifest of layer specs + SRT1 weights
  rebuilds any supported Sequential, bit-exact. Supported layers: Linear, ReLU,
  Sigmoid, LayerNorm, BatchNorm2d, Conv2d, MaxPool2d (MLP and CNN demonstrated).
  BatchNorm2d runs in eval mode (frozen running statistics).
- Depends on scirust-core by path; the golden-fingerprint checks are the
  regression lock against core drift.

## The three guarantees (measured on MLP 784-256-10 and CNN Conv-Pool x2 -> MLP)

| Guarantee | Contract | Measured evidence |
|---|---|---|
| **#1 Determinism** | Bit-exact output for a fixed (binary, target), independent of thread count and across process restarts. | MLP: 5120 comparisons, 0 divergences, fp 0xde2d807686e4b47e stable across RAYON_NUM_THREADS in {1,2,4,8,16,64}. CNN: forward x3 bit-exact 0x1381e4b51d0eeba4, thread-invariant {1,2,4,8}. SRT1 reload bit-exact (incl. LayerNorm gamma/beta and BatchNorm2d running stats). |
| **#2 Bounded latency** | Predictable per-request latency, tight tail. | MLP batch=1: p50 126us, p99/p50 = 1.15x, thread-invariant. CNN batch=32: p50 45.9ms, p99/p50 = 1.20x. |
| **#3 Auditability** | Frozen artifact has stable identity; inference reproducible; reconstructable from a manifest. | SRT1 sorted keys give deterministic on-disk bytes (stable hashes). Trained MNIST MLP rebuilt from manifest + mnist_mlp.srt reproduces 97.73% accuracy and fp 0xc96d25fa658f5611 bit-for-bit. |

### Honest boundaries

- Manifest layer set: linear, relu, sigmoid, layernorm, batchnorm2d, conv2d,
  maxpool2d. Transformer layers use a 3D forward (Var3D) and are out of the
  current Sequential-2D manifest; supporting them would be a separate 3D path.
- Determinism is bit-exact for a fixed compiled artifact on a given architecture.
  Cross-architecture bit-exactness (x86 vs aarch64) is out of scope by design.
- Pure-Rust conv is slow in absolute throughput (~697 samples/s on the CNN); a
  performance axis, not a guarantee gap. The tail stays tight.
- Absolute MLP batch=1 latency (~126us) is fixed per-call overhead, not compute.

## Generic model reconstruction (manifest)

One layer per line; build_model(parse_manifest(text)) + load_weights rebuild any
supported Sequential bit-exact. Example (the CNN):

    conv2d 3 32 3 1 same 32 32
    relu
    maxpool2d 2 2 32 32 32
    conv2d 32 64 3 1 same 16 16
    relu
    maxpool2d 2 2 64 16 16
    linear 4096 256
    relu
    linear 256 10

Supported lines:

    linear      <in> <out>
    relu
    sigmoid
    layernorm   <d_model> <eps>
    batchnorm2d <channels>          # eval mode: uses frozen running stats
    conv2d      <in_c> <out_c> <kernel> <stride> <same|valid> <in_h> <in_w>
    maxpool2d   <kernel> <stride> <c> <h> <w>

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

    cargo run --release --bin train_artifact   # offline: train MLP -> mnist_mlp.srt
    cargo run --release --bin eval_artifact     # runtime: load + eval (97.73%, fingerprint)
    cargo run --release --bin bench_latency     # latency distribution
    cargo run --release --bin cnn_audit         # three guarantees on a CNN
    cargo run --release --bin generic_check     # manifest reconstruction (MLP + CNN)
    cargo run --release --bin layers_check      # Sigmoid + LayerNorm round-trip
    cargo run --release --bin bn_check          # BatchNorm2d (eval) round-trip
    cargo run --release                          # golden persistence lock (reload bit-exact)

MNIST_DIR overrides the dataset path (default /root/scirust/data/mnist).
MNIST_MAX_TRAIN caps training set size.

## Status

Build-isolated crate (own [workspace]); to be promoted to a workspace member at
integration. Part of the SciRust research artifact, documenting human-directed
construction of a deterministic, generic inference runtime over a churning
research core.
