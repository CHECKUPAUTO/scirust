# scirust-hypercrypto

> ```text
> EXPERIMENTAL RESEARCH CONSTRUCTION
>
> This crate is a Phase-1 structural falsification harness for the
> SciRust-HyperCrypto v0.1 hypercomplex keyed-permutation candidate.
> Its first-order goal is to BREAK v0.1, not to endorse it.
>
> It is NOT a cipher, hash, KEM, signature, or post-quantum scheme.
> It has NOT received independent cryptanalysis.
> It MUST NOT be used to protect real data, credentials, financial records,
> health information, production secrets, or communication systems.
>
> Use established, standardized cryptographic primitives for production.
> ```

This crate is **not** a recommended cryptographic dependency. It exists to run
adversarial, falsification-oriented experiments against the construction defined
in [`docs/research/SCIRUST_HYPERCRYPTO_SPEC_V0_1.md`](../docs/research/SCIRUST_HYPERCRYPTO_SPEC_V0_1.md),
and to record the results in
[`docs/research/SCIRUST_HYPERCRYPTO_FALSIFICATION_PHASE1.md`](../docs/research/SCIRUST_HYPERCRYPTO_FALSIFICATION_PHASE1.md).

## What it contains

- exact scalar octonion arithmetic over `Z/2^k` (`k ∈ {2,4,8,16,64}`), integer-only;
- the exact v0.1 round function (`F-PROG`) and a balanced Feistel shell;
- deliberately-weakened **control** variants used to validate the analysis tools;
- experiments: matrix lifting, linearity/affinity, algebraic degree (exact ANF),
  norm/conjugation invariants, zero-divisor fibers, and subspace structure;
- a deterministic, fingerprinted machine-readable report and a research CLI.

Properties: pure Rust, zero FFI, `#![forbid(unsafe_code)]`, no floating point,
no OS entropy, deterministic, platform-independent, scalar (no SIMD).

## Run the falsification battery

```bash
export HYPERCRYPTO_GIT_COMMIT=$(git rev-parse HEAD)
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- report --sample 80000
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- help
```

Current Phase-1 verdict:

```text
PHASE-1 VERDICT: CONTINUE — NO GATING BREAK FOUND BY THESE EXPERIMENTS
```

**`CONTINUE` is not evidence of security.** See the Phase-1 report for the full
findings, the documented weak-key observation, limitations, and Phase-2 leads.

## Gates

```bash
cargo +nightly-2026-07-02 fmt   -p scirust-hypercrypto -- --check
cargo +nightly-2026-07-02 clippy -p scirust-hypercrypto --all-targets --locked -- -D warnings
cargo +nightly-2026-07-02 test  -p scirust-hypercrypto --locked
cargo +1.89.0             check -p scirust-hypercrypto --all-targets --locked
```
