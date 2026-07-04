# SciRust — Functional Acceptance Protocol

A single command that certifies the **entire** platform end to end:

```bash
scripts/test-protocol.sh
```

It runs every quality gate CI enforces, executes **every crate's oracle tests**,
re-proves cross-process determinism, cross-compiles the aarch64 NEON/SVE paths,
builds the docs warning-free, and runs the supply-chain audit — then prints one
`PASS` / `FAIL` verdict and writes a timestamped **evidence bundle** suitable for
an industrial acceptance sign-off.

There are no stubs in this protocol. Every "functionality" it claims to verify is
backed by the same honest oracle test that ships inside the crate: a fixed-seed
RNG, a golden constant, or agreement against an independent reference. The
protocol's job is to *run them all* and report the result faithfully.

---

## Quick reference

| Command | Scope |
|---|---|
| `scripts/test-protocol.sh` | Full protocol (default) |
| `scripts/test-protocol.sh --quick` | `fmt + clippy + build + test + determinism` only |
| `scripts/test-protocol.sh --with-examples` | Also smoke-run the data-free example binaries |
| `scripts/test-protocol.sh --only test,doc` | Run just the named gates |
| `scripts/test-protocol.sh --skip gpu,deny` | Run everything except the named gates |
| `scripts/test-protocol.sh --strict` | A missing prerequisite **fails** instead of skipping |
| `scripts/test-protocol.sh --no-clean` | Keep `target/doc` and the incremental cache |
| `scripts/test-protocol.sh --list` | Print the gate plan and exit |

**Exit code:** `0` when every required gate passed; non-zero when a required gate
failed (or, under `--strict`, was skipped for a missing prerequisite).

---

## The gates

The eight gates below are exactly the commands CI enforces (`.github/workflows/ci.yml`)
and that the developer workflow runs locally — they stay identical so a green
local run means a green CI run. The protocol adds an explicit **determinism**
gate and a handful of opt-in extras.

| Gate | Command | What it proves | Required |
|---|---|---|:--:|
| `fmt` | `cargo +nightly-2026-07-02 fmt --all -- --check` | Source is in canonical form. Uses the nightly pinned in CI (`rustfmt.toml` has unstable options; rustfmt output is version-sensitive, so the version is fixed to keep local and CI identical). | ✓ |
| `clippy` | `cargo clippy --workspace --all-targets -- -D warnings` | No lint warnings anywhere — lib, bins, tests, examples. | ✓ |
| `build` | `cargo build --workspace --all-targets` | The whole workspace compiles, every target. | ✓ |
| `test` | `cargo test --workspace --no-fail-fast` | **Every crate's oracle tests** pass — this is the functionality gate. | ✓ |
| `simd` | `cargo test -p scirust-simd --features portable-simd` | The optional nightly portable-SIMD kernels are correct. | ✓ |
| `determinism` | determinism-tagged tests, run in **two processes**, compared | A computation reproduces bit-for-bit across process invocations. | ✓ |
| `doc` | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace` | The public API documents cleanly, no broken links. | ✓ |
| `aarch64` | `cargo check --workspace --all-targets --target aarch64-unknown-linux-gnu` | The NEON/SVE `cfg(target_arch)` paths type-check for ARM. | ✓¹ |
| `deny` | `cargo deny check` | Licenses and security advisories are clean. | ✓¹ |
| `clippy-gpu` | `cargo clippy -p scirust-gpu --features wgpu --all-targets -- -D warnings` | The optional wgpu feature lints cleanly. | optional |
| `gpu` | `cargo test -p scirust-gpu --features wgpu` | The real wgpu GEMM matches the CPU oracle on a Vulkan adapter. | optional² |
| `stable` | `cargo +stable build/test --workspace` | The workspace builds and passes on the **stable** toolchain. | optional² |
| `examples` | `cargo run -p <demo>` for the data-free demos | The bundled example binaries run clean. | optional³ |

¹ Required, but **skipped** (not failed) when the prerequisite is absent — the
aarch64 std component or `cargo-deny`. A skipped required gate yields the
`PASS (with gaps)` verdict so incomplete coverage is never reported as a clean
pass. Under `--strict` a missing prerequisite is a hard failure.
² Skipped automatically when no Vulkan adapter / no stable toolchain is present.
³ Only runs with `--with-examples`; data-dependent demos (MNIST/CIFAR/sentiment)
are excluded so the gate never produces a false failure.

---

## What "test ALL functionality" means here

SciRust is a workspace of ~90 crates. Each crate proves its own behaviour with
oracle tests — not mocks. The `test` gate runs `cargo test --workspace`, which
compiles the whole graph once and executes **every** unit, integration, and
documentation test in it. That single gate is the functional core of this
protocol; the others guard the properties those tests assume (lint-cleanliness,
portability, determinism, docs, licensing).

The protocol does **not** re-implement each crate's checks. It runs the checks
the crates already ship, and reports the aggregate (tests passed / failed /
ignored, and how many test groups) into the evidence bundle.

### Capability → guarantee → gate

| Capability area (crates) | The guarantee | Verified by |
|---|---|---|
| Determinism foundation (`scirust-core`, `scirust-simd`, `scirust-runtime`) | Bit-exact, order-independent reductions; fingerprint stable across threads & processes | `test` + `determinism` + `simd` |
| Autograd / deep learning (`scirust-core`, `scirust-learning`) | Every op finite-difference gradient-checked; trainers bit-identical for 1/2/4/8 threads | `test` + `determinism` |
| Certifiable inference (`scirust-core`: IBP/CROWN, conformal, calibration) | Provable output bounds; distribution-free coverage; lower ECE | `test` |
| Embedded int8 (`scirust-core`, `scirust-embedded`, `scirust-simd`) | Lossless int8 path; NEON kernels bit-exact vs scalar reference | `test` + `aarch64` |
| Signal & predictive maintenance (`scirust-signal`, `scirust-pdm`) | FFT/bearing diagnostics; Health-Index/RUL/CUSUM detectors | `test` |
| State estimation & navigation (`scirust-estimation`, `scirust-nav`) | Kalman/IMM/UD filters (covariance PSD by construction); GNSS-INS fusion; TDOA | `test` + `determinism` |
| Industrial verticals (`scirust-control`, `-bms`, `-grid`, `-shm`, `-hvac`, `-spc`, `-robotics`, `-metrology`, `-water`, `-biomed`, `-reliability`) | Domain laws validated against textbook / reference values, deterministically | `test` |
| Connectivity (`scirust-opcua`, `scirust-mqtt`) | OPC-UA sensor model; MQTT Sparkplug-B encoding | `test` |
| MLOps & functional safety (`scirust-mlops`, `scirust-func-safety`) | Drift / shadow / signed OTA; ISO 26262 ASIL, fault injection, **hash-chained audit log**, GMP golden-batch | `test` |
| OT security (`scirust-ids`) | Firmware attestation & PLC ladder integrity on a tamper-evident hash chain | `test` |
| GPU (`scirust-gpu`) | wgpu GEMM matches the deterministic CPU oracle | `gpu` (optional) |

---

## The determinism gate, in detail

Numeric determinism is the platform's headline guarantee, so the protocol proves
it explicitly rather than trusting it. Across the workspace, 100+ tests pin a
computation to a **golden constant** or a **fixed-seed** sequence (test names
containing `deterministic`, `determinism`, `reproducible`, `bit_exact`,
`bit_reproducible`, `golden`). Those run inside the `test` gate already — passing
them proves a result is reproducible *within* a process.

The `determinism` gate goes one step further: it runs that tagged subset in **two
independent `cargo test` processes** and compares the sorted set of passing tests.
Because each test asserts against a fixed oracle, two byte-identical green runs in
two separate processes demonstrate the computation is bit-reproducible *across
process invocations* — the property an auditor actually cares about. If the two
runs disagree, or either run goes red, or the filter matches zero tests, the gate
fails (it never silently passes a vacuous check).

---

## Evidence bundle

Every run writes a timestamped directory under `target/protocol-logs/run-<UTC>/`:

```
toolchain.txt          rustc / cargo / clippy / rustfmt versions + host
fmt.log clippy.log build.log test.log simd.log doc.log aarch64.log deny.log
determinism-run1.log   first determinism process
determinism-run2.log   second determinism process
determinism-sig1.txt   sorted passing-test signature, run 1
determinism-sig2.txt   sorted passing-test signature, run 2  (must equal sig1)
summary.txt            machine-readable: commit, branch, gate results, totals, verdict
```

`summary.txt` is the artifact to attach to a release or hand to QA: it records the
commit, the per-gate result with timing, the test tally, the determinism count,
and the final verdict.

---

## Interpreting the verdict

- **`PASS`** — every required gate green. Ship it.
- **`PASS (with gaps)`** — all *run* required gates green, but one or more were
  skipped for a missing prerequisite (e.g. no `cargo-deny`, or the aarch64 std
  component is not installed). Coverage is incomplete; install the prerequisite
  or re-run with `--strict` to demand full coverage.
- **`FAIL`** — at least one required gate failed. The offending gate's log
  (tail printed inline, full file in the bundle) shows exactly what broke.

To reproduce a single failing gate quickly, run its command from the table above
directly, or `scripts/test-protocol.sh --only <gate>`.

---

## Notes & limits (honest scope)

- The protocol mirrors the **documented local gates**. To additionally enforce
  CI's workspace-wide `RUSTFLAGS="-D warnings"` (which turns *every* warning,
  not just clippy lints, into an error and forces a full rebuild), run with
  `--strict`-style strictness by exporting `RUSTFLAGS="-D warnings"` before
  invoking. Lint-cleanliness is already enforced by the `clippy` gate.
- `gpu` needs a Vulkan adapter (CI uses Mesa **lavapipe**, a software adapter).
  Without one the gate is skipped, not failed.
- `--all-features` is intentionally **not** used: `scirust-core`'s
  `blas-openblas` / `blas-mkl` features are mutually exclusive backends that
  require system toolchains, so `--all-features` can never build. Feature paths
  that matter are gated explicitly (`simd`, `gpu`).
- The data-dependent example binaries (MNIST/CIFAR/sentiment) are not run by
  `--with-examples`; their training data is not part of the repository.
- **Disk:** a full `--all-targets` debug tree for the whole workspace is
  ~25–30 GB. The script sets `CARGO_INCREMENTAL=0` (as CI does) to keep the tree
  small, reclaims the aarch64 cross-tree right after that gate, and refuses to
  start with a clear message if free space is below the scope's need (~30 GB for
  build/test, ~3 GB for `--only fmt,clippy`). On a constrained machine, run the
  heavy gates one at a time and `cargo clean` between them, or run the light
  gates locally and let CI carry the full build/test/cross matrix.
