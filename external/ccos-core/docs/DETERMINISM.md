# CCOS_EXTENDED — Determinism Boundary

The fusion plan's third invariant (see `docs/FUSION_PLAN.md` §F) is:

> **`replay == live` for the default build**; every full-kernel feature carries a
> `REPLAY-RELAX` marker and a documented divergence.

This document is the human-readable form of that boundary: which features keep
`replay == live` (a re-run is bit-for-bit identical to the original live run),
which features are a *permitted* relaxation, and why. The executable form is
`tests/determinism_boundary.rs`, whose `REPLAY_TABLE` is the single source this
table is kept in sync with.

## Posture

CCOS_EXTENDED runs a **DUAL determinism posture**:

- **Default build** (`cargo build`, no features): deterministic, std-only,
  bit-for-bit replayable. No network egress, no SIMD nondeterminism, no
  subprocess. This is the posture the core, the causal graph, Q-Page
  belief/decay/propagation, and the distilled recall path run in.
- **Full-kernel features** (`slhav2-full`, `rsi-dgm`, `rsi-full`, …): opt-in,
  Pro-gated, and each carries a `REPLAY-RELAX` marker describing the *one*
  permitted divergence. The default build compiles none of them.

The boundary is enforced three ways:

1. **Cargo feature gates** — the relaxed code is not compiled at all in the
   default build (`cargo tree` shows no `scirust`, no `rsi`).
2. **License gates** — even when compiled, relaxed paths are Pro-gated; the
   community tier is refused with `LicenseError::FeatureLocked` (no silent
   downgrade into a divergent path).
3. **Air-gap** — `ccos::egress::EgressAllowlist` (default localhost/loopback
   only) prevents any off-host network call, so the default build has no network
   source of replay divergence even with `llm`/`neural-embed` compiled in.

## Feature → replay posture

| Feature (`license::Feature`)            | Cargo feature(s)                    | Replay     | Divergence (if Relax)                                                                |
|-----------------------------------------|-------------------------------------|------------|--------------------------------------------------------------------------------------|
| `CustomAuthorityWeights`               | — (core)                            | **Safe**   | —                                                                                    |
| `TensionVisualization`                 | — (core)                            | **Safe**   | —                                                                                    |
| `AuditReports`                         | — (core)                            | **Safe**   | —                                                                                    |
| `SlhAv2Embeddings`                     | `slhav2`                            | **Safe**   | — (distilled grouped-INT4, deterministic)                                          |
| `AdaptiveRetrieval`                    | — (core)                            | **Safe**   | —                                                                                    |
| `OctaSomaMemory`                       | `octasoma`                          | **Safe**   | — (embedding index, deterministic given the embeddings)                              |
| `SlhAv2FullKernel`                     | `slhav2-full`                       | **RELAX**  | SIMD accumulation order + stateful importance tracking (H2O/attention-sink eviction) break bit-exact replay. |
| `RsiSelfImprovement`                   | `rsi`                               | **RELAX**  | The std-only RSI core stays deterministic; the relax is the *agent run* over a live knowledge base (audit is hash-chained and replayable, but the run itself is not bit-identical to a prior live run). |
| `RsiDgm`                                | `rsi-dgm`                           | **RELAX**  | The proposer/evaluator stay deterministic; the relax is the real `cargo --offline --frozen` build/test subprocess (filesystem + toolchain state is an environmental input). |

The relaxed set (`SlhAv2FullKernel`, `RsiSelfImprovement`, `RsiDgm`) is exactly
the Pro-gated set: a relaxed feature that were free would let replay diverge in
the default build and is therefore forbidden. `tests/determinism_boundary.rs::
replay_table_is_total_and_consistent` asserts this at runtime.

## Air-gap (network determinism)

The `llm`, `neural-embed`, and `eval` modules each construct a URL and pass it
through `ccos::egress::EgressAllowlist::from_env()` before any `reqwest` call:

| Call site                            | Default host             | On refusal                            |
|--------------------------------------|--------------------------|---------------------------------------|
| `llm::LlmClient::query_as`           | `http://localhost:11434` | fallback `ValidatedResponse` (no call)|
| `neural_embed::NeuralEncoder::try_new` | `http://127.0.0.1:11434` | `NeuralEmbedError::EgressDenied`      |
| `eval::ask` (Anthropic/OpenAI/Ollama) | `api.anthropic.com` / `api.openai.com` / `OLLAMA_ENDPOINT` | `None` (no LLM)   |

The default allowlist is `localhost`, `127.0.0.1`, `::1`, `[::1]`, `0.0.0.0`.
Any other host is refused (`EgressError::HostNotAllowed`) — **fail-closed**.
An operator widens it with `CCOS_EGRESS_ALLOW=host1,host2,...` (comma-separated,
case-insensitive). Missing/unset ⇒ localhost-only; the policy never opens by
default.

## `unsafe` boundary

`ccos-scirust` denies `unsafe` at the crate root (`#![deny(unsafe_code)]`) and
allows it only in two audited modules:

- `numa.rs` — NUMA policy, behind `#[cfg(all(feature = "numa", target_os = "linux"))]` (default off); every `unsafe` block carries a `// SAFETY:` justification, and the `Send`/`Sync` impls for `AlignedBuffer`/`NumaBuffer` are justified.
- `attention/slha_v2.rs` — SIMD intrinsics, `#[cfg(target_arch="x86_64")]` + `#[target_feature]`-gated, runtime-guarded by `is_x86_feature_detected!`, with a scalar reference path checked for equivalence.

The FFI crate `slha-c` is output-boundary-only: it never allocates/frees tiles,
borrows caller-owned memory, and `debug_assert!`s the tile alignment on entry.
See `crates/ccos-scirust-c/include/slha.h`.

## Verifying the boundary

```bash
# Default build: replay == live (no scirust, no rsi in the tree).
cargo check
cargo tree | grep -E 'scirust|rsi'   # empty

# The relaxed paths compile only when asked for.
cargo check --features slhav2-full
cargo check --features rsi,rsi-dgm
cargo check --features llm,neural-embed

# The executable determinism contract.
cargo test --test determinism_boundary
```

`tests/determinism_boundary.rs` asserts, in the default build: the egress
default denies a public host; the community tier refuses every REPLAY-RELAX
feature; and the `REPLAY_TABLE` is total and consistent (every relaxed feature
is Pro-gated).