<p align="center">
  <img src="https://github.com/user-attachments/assets/c36c292c-5893-44c2-94d9-3895ec0749e8" alt="SciRust" width="760">
</p>

# SciRust

[![CI](https://github.com/Memorithm/scirust/actions/workflows/ci.yml/badge.svg)](https://github.com/Memorithm/scirust/actions/workflows/ci.yml)
[![ARM64](https://github.com/Memorithm/scirust/actions/workflows/native-arm64.yml/badge.svg)](https://github.com/Memorithm/scirust/actions/workflows/native-arm64.yml)
[![Rust](https://img.shields.io/badge/Rust-1.89%2B-000000?logo=rust)](rust-toolchain.toml)
[![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE.md)

SciRust is an experimental, pure-Rust workspace for deterministic machine
learning, scientific computing, simulation, and industrial algorithms. Its
design priority is inspectability: reference implementations, explicit seeds,
oracle-based tests, and opt-in hardware backends live in the same repository.

SciRust is a research and engineering project. It is **not** a drop-in
replacement for PyTorch, a certified safety component, or a validated medical,
financial, or industrial control product.

## Project status

The repository is under active development and contains more than one hundred
workspace crates. APIs and crate boundaries may change before a stable release.
The root package is currently version `0.14.0` and declares Rust `1.89` as its
minimum supported version.

| Area | Current scope | Maturity |
|---|---|---|
| Tensor and autodiff core | Dense tensors, reverse-mode autodiff, neural-network layers, optimizers | Research implementation |
| Deterministic execution | Seeded examples, fixed-order reductions, inference artifacts and audit utilities | Tested for documented paths; not a universal cross-platform guarantee |
| CPU acceleration | Scalar and architecture-specific SIMD paths | Available |
| WGPU | Opt-in canonical tensor adapter and GPU kernels | Experimental |
| CUDA | Opt-in canonical tensor adapter; requires NVIDIA driver, NVRTC, and a CUDA device at runtime | Experimental; hardware-dependent |
| Scientific computing | Solvers, symbolic methods, statistics, simulation, signal processing, and domain crates | Scope varies by crate |
| Industrial and regulated domains | Reference algorithms and deterministic demonstrations | Educational/research use; no certification |
| SciAgent and RSI | Local model and algorithm-refinement experiments | Experimental |

The status above deliberately describes repository scope rather than claiming
production readiness. Crate-specific limitations belong in each crate's API
documentation or README.

## Design principles

- **Rust-first implementation.** Default compute paths do not wrap libtorch or
  ONNX Runtime. Optional operating-system, BLAS, WGPU, and CUDA integrations are
  documented where enabled.
- **Verification over headline metrics.** Numerical code is tested against
  analytic results, reference implementations, invariants, or CPU oracles where
  those checks exist.
- **Determinism is scoped.** Bitwise claims apply only to the exact code path,
  target, features, toolchain, and test described by the corresponding evidence.
- **Hardware backends are opt-in.** The default build does not activate WGPU or
  CUDA.
- **Unsafe code is localized.** Individual crates state their own unsafe-code
  policy; the repository does not make a blanket zero-unsafe claim.

## Quick start

### Requirements

- Git
- Rust installed through [rustup](https://rustup.rs/)
- the toolchain pinned in [`rust-toolchain.toml`](rust-toolchain.toml)

```bash
git clone https://github.com/Memorithm/scirust.git
cd scirust
cargo install --path scirust-cli
scirust help
scirust info
scirust quickstart
```

Without installing the CLI:

```bash
cargo run -p scirust-cli -- help
cargo run -p scirust-cli -- quickstart
```

The quickstart trains the repository's small deterministic classifier example.
It is a functional smoke test, not a performance benchmark.

## Library example

The CLI is a thin entry point over workspace crates. Applications can depend on
individual path crates while developing inside the workspace:

```toml
[dependencies]
scirust-core = { path = "scirust-core" }
scirust-solvers = { path = "scirust-solvers" }
```

For an end-to-end classifier, start with
[`examples/quickstart_v2`](examples/quickstart_v2) or run:

```bash
cargo run -p quickstart_v2
```

## Repository map

| Path | Purpose |
|---|---|
| [`scirust-core/`](scirust-core/) | Core tensor, autodiff, neural-network, and quantization code |
| [`scirust-cli/`](scirust-cli/) | Unified `scirust` command-line interface |
| [`scirust-tensor-core/`](scirust-tensor-core/) | Canonical tensor representation |
| [`scirust-tensor-runtime/`](scirust-tensor-runtime/) | Canonical execution runtime |
| [`scirust-gpu/`](scirust-gpu/) | CPU reference, WGPU, and CUDA adapters behind features |
| [`scirust-learning/`](scirust-learning/) | Learning algorithms, NLP utilities, and reinforcement learning |
| [`scirust-solvers/`](scirust-solvers/) | Numerical solvers and linear algebra |
| [`scirust-symbolic/`](scirust-symbolic/) | Symbolic expressions and differentiation |
| [`scirust-sim/`](scirust-sim/) | Deterministic simulation environments |
| [`scirust-signal/`](scirust-signal/) | Signal processing and radar-oriented reference algorithms |
| [`scirust-industrial/`](scirust-industrial/) | Industrial demonstration CLI |
| [`scirust-mcp/`](scirust-mcp/) | Model Context Protocol server |
| [`scirust-sciagent/`](scirust-sciagent/) | Experimental local language-model tooling |
| [`examples/`](examples/) | Runnable examples |
| [`docs/`](docs/) | Guides, design notes, evidence, and archived research notes |
| [`paper/`](paper/) | Technical reports and paper material; measurements are historical unless dated otherwise |
| [`external/`](external/) | Vendored external project snapshots; not part of the SciRust public API |

This map highlights entry points rather than listing every crate. Use
`cargo metadata --no-deps` for the authoritative workspace inventory.

## Features and hardware backends

The root crate has no default features. The canonical tensor facade is exposed
through explicit features:

| Feature | Effect |
|---|---|
| `tensor-canonical` | Canonical tensor facade with a caller-supplied backend |
| `tensor-canonical-cpu` | Facade plus the deterministic CPU reference adapter |
| `tensor-canonical-wgpu` | Facade plus the WGPU adapter |
| `tensor-canonical-cuda` | Facade plus the CUDA adapter |

Examples:

```bash
cargo run --example canonical_tensor_cpu --features tensor-canonical-cpu
cargo test -p scirust-gpu --features wgpu
cargo test -p scirust-gpu --features cuda
```

Enabling a feature proves only that the selected code is built and tested in
the current environment. It does not imply that compatible GPU hardware or
drivers are present.

## Verification

The main workflow is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
Run the standard local checks with:

```bash
cargo +nightly-2026-07-02 fmt --all -- --check
cargo +nightly-2026-07-02 clippy --workspace --all-targets --locked -- -D warnings
cargo +nightly-2026-07-02 build --workspace --all-targets --locked
cargo +nightly-2026-07-02 test --workspace --locked
```

Additional workflows cover native ARM64, release, SOS, and Studio-specific
paths. Some hardware tests require self-hosted runners and therefore provide a
narrower guarantee than portable hosted CI. See
[`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md) for the repository's acceptance
protocol and [`docs/evidence/`](docs/evidence/) for retained evidence.

Test totals and benchmark numbers are intentionally omitted here because they
become stale as the workspace changes. Reproduce a result from its dated report
and exact command instead.

## Documentation

- [Quickstart](docs/QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Command and API reference](docs/REFERENCE.md)
- [GPU status and usage](docs/GPU.md)
- [Test protocol](docs/TEST_PROTOCOL.md)
- [Release process](docs/RELEASING.md)
- [Security policy](SECURITY.md)
- [Licensing guide](LICENSING.md)
- [Changelog](CHANGELOG.md)
- [Research notes](docs/research-notes/)
- [Audit archive](docs/audits/)
- [Translations](docs/translations/)

Historical reports record the state and measurements of a particular revision.
They should not be read as guarantees for the current `master` branch.

## Contributing and security

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a change. Report
security issues through the private process in [`SECURITY.md`](SECURITY.md), not
through a public issue.

## License

SciRust is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md). Commercial use is not
granted by that license. See [`LICENSING.md`](LICENSING.md) for the separate
commercial licensing path.

Copyright © 2026 Tarek Zekriti.
