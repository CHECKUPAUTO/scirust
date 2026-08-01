# Security Policy

## Reporting Vulnerabilities

Please report any vulnerability privately to **zekrititarek@gmail.com**
(maintainer; see `paper/SciRust-technical-report.md`). Do not open a public
issue for an exploitable flaw. Reports are acknowledged within 7 days.

## Scope and Guarantees

- **Pure Rust, with no embedded C/C++ library**: the active workspace includes
  no *consumed* FFI dependency (it does not link to a third-party C/C++
  library). The crate supply chain is limited to the dependencies listed in
  the committed `Cargo.lock` and audited by `cargo deny check` (RustSec
  advisories, licenses, and sources) in CI.

  > **Exported FFI note.** `scirust-runtime/src/enclave.rs` exports an
  > `extern "C"` entry point (`safe_enclave_infer`) intended for a TEE /
  > TrustZone `#![no_std]` environment. This is an exported Rust-to-C ABI (the
  > runtime can be called from C), not an embedded C library. Buffer sizes
  > (`dims`) are validated against Rust slices before the `unsafe` path
  > (`EnclaveRuntime::infer`), so inconsistent `dims` values are rejected
  > (`Err`) instead of causing an out-of-bounds read or write in the enclave.

  > **Archive note.** The `archive/` directory contains older code (notably
  > `archive/scirust-gpu/{cublas.rs,cuda_backend.rs}` and
  > `archive/scirust-simd/sve.rs`) that uses C/CUDA FFI. This code is not part
  > of the active workspace (it is outside `Cargo.toml`), is not compiled by
  > CI, and is retained for historical purposes. It is therefore outside the
  > scope of the guarantees above.

  > **Optional network/TLS features.** The “no consumed FFI” guarantee applies
  > to the default build (`cargo build --workspace`, verifiable with
  > `cargo tree --workspace`, which shows neither `ring`, `aws-lc-sys`, nor
  > `reqwest`). Three features that are disabled by default instead pull in a
  > TLS stack that links C/assembly: `scirust-trader/live` (→ `reqwest` +
  > `rustls` + `aws-lc-sys`), `scirust-rsi/anthropic`, and
  > `scirust-sciagent/fetch` (→ `ureq` + `ring`). These crates are therefore
  > listed in `Cargo.lock` (which includes all optional dependencies) but are
  > compiled and linked only when the feature is explicitly enabled;
  > `scirust-sciagent/Cargo.toml` already documents this trade-off. A
  > deployment that must remain 100% pure Rust should leave these features
  > disabled.

- **Confined and justified `unsafe` code**: `unsafe` appears in several
  modules (SIMD intrinsics in `scirust-simd/src/{dispatch,complex}.rs`, memory
  alignment in `scirust-arena/src/{slab,aligned,allocator}.rs` with
  `AlignBlock(128)` backing, autodiff/tensor/matrix code in `scirust-core`, and
  the enclave entry point described above). Each block is documented with a
  safety header covering alignment, invariants, and the versioning scheme used
  to prevent use-after-free. Callers do not need `unsafe` on public
  high-level API paths; the internal `unsafe` code is encapsulated.

- **Determinism**: inference is bit-exact and replayable (the SRT1 runtime), a
  property useful for forensic audits. The noise in the fault-injection
  campaign (`scirust-func-safety/src/fault_injection.rs`, `NoiseInjection`) is
  deterministic as well (an LCG with a fixed seed derived from the neuron
  index) so that runs remain reproducible; unseeded `rand::random` is not used
  on any inference path.

## SBOM (Software Bill of Materials)

- **CycloneDX 1.x (JSON)** — a format consumable by industrial scanners such
  as OWASP Dependency-Track and Grype.
- **Reproducible generation**: `./scripts/generate-sbom.sh` uses
  `cargo cyclonedx` and the committed `Cargo.lock`. A snapshot is versioned in
  [`docs/sbom/`](docs/sbom/) for immediate visibility.
- **CI/release**: the `sbom` job regenerates and publishes the SBOM as an
  artifact for every build; the release workflow attaches it to every `v*`
  tag (see `release v0.14`). The source of truth remains `Cargo.lock` plus
  regeneration; the committed snapshot must not be treated as the source.

## CI Supply Chain

- GitHub workflows use third-party actions. Pinning currently uses version
  tags (`@v2`, `@nightly`); to harden the supply chain, pinning these actions
  to commit SHAs is recommended (see the audit
  [`docs/audits/AUDIT_COMPLET.md`](docs/audits/AUDIT_COMPLET.md), finding S4).
- No workflow uses `pull_request_target` (the dangerous privilege-escalation
  pattern). The `release.yml` workflow restricts `permissions: contents: write`
  to the single operation that creates a release.

## Known Accepted Advisories

- RUSTSEC-2024-0436 (`paste`, unmaintained — not a vulnerability): a
  transitive dependency of nalgebra → simba, with no upstream fix; it is
  ignored with a justification in `deny.toml`.

## Certification Artifact Integrity

- The evidence records (`scirust-func-safety/src/evidence.rs`, FNV-1a hash
  chain) are **tamper-evident but not tamper-resistant**: they detect a naive
  edit (a field changed without recomputing the chain) but cannot resist an
  attacker who recomputes the entire chain (the algorithm is public and uses
  no secret key). Integrity therefore relies on write access control and on
  the runtime's proof argument (verifiable inference). Do not authenticate an
  untrusted record solely with `from_json().verify()`.
