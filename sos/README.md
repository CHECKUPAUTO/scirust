# SOS — Scientific Operating System (implementation)

This is the **implementation workspace** for the Scientific Operating System.
The architecture it realizes is specified in [`docs/sos/`](../docs/sos/)
(RFC-0002); the discovery-loop subsystem is specified in
[`docs/sde/`](../docs/sde/) (RFC-0001).

SOS is a **separate Cargo workspace** from the SciRust workspace at the
repository root (RFC-0002 §11.6): it is excluded from the root workspace build,
has its own `Cargo.lock`, and will consume SciRust only from the two backend
adapter crates. This keeps SciRust's "whole workspace builds on stable" gate
intact and lets SOS evolve on its own cadence.

## Status

Delivery is phased and **production-ready each phase** (RFC-0002 §12) — no
stubs, no TODOs, no placeholders cross a phase boundary.

| Phase | Scope | Status |
|-------|-------|--------|
| **P1 — Kernel & substrate** | `sos-core`, `sos-store`, `sos-provenance`, `sos-repro`, `sos-registry` | **in progress** |

### Landed

- **`sos-core`** — the kernel. The immutable, content-addressed
  [`Object`](sos-core/src/object.rs) envelope with deterministic canonical
  hashing, the honest four-level [`DeterminismLevel`](sos-core/src/determinism.rs)
  taxonomy, and full provenance / reproducibility metadata. Pure Rust, no FFI,
  `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`.
- **`sos-store`** — the Storage Layer (the kernel's filesystem). A
  content-addressed [`ObjectStore`](sos-store/src/store.rs) with typed
  [`put_object`/`get_object`](sos-store/src/store.rs) that **verify the content
  address on read and write**, content-addressed [`BlobRef`](sos-store/src/blob.rs)
  blobs, mutable named refs, and reachability [`gc`](sos-store/src/store.rs). Ships
  a complete in-memory backend ([`MemoryStore`](sos-store/src/mem.rs)); persistent
  backends implement the same trait.
- **`sos-provenance`** — the Provenance Engine. A queryable
  [`ProvenanceGraph`](sos-provenance/src/graph.rs) over any store — `ancestors`
  ("why do we believe X"), `descendants` ("what breaks if X is retracted"),
  `roots`, `tips` — plus deterministic [environment capture](sos-provenance/src/env.rs)
  for the reproducibility key. (Signing is deferred to `sos-scirust`, keeping this
  crate backend-agnostic per Invariant VIII.)
- **`sos-registry`** — the Plugin System. Content-pinned
  [`PluginDescriptor`](sos-registry/src/descriptor.rs)s (name + version +
  content hash + [`Role`](sos-registry/src/descriptor.rs) + determinism level +
  capabilities + domains), a [`Registry`](sos-registry/src/registry.rs) that
  resolves by semantic version and **detects content-hash drift**, and
  least-privilege [capability authorization](sos-registry/src/capability.rs)
  (refuse-by-default).

## Engineering standards (the gate)

Every crate must pass, on every change:

```sh
cargo fmt   --manifest-path sos/Cargo.toml --all --check
cargo clippy --manifest-path sos/Cargo.toml --all-targets -- -D warnings
cargo test  --manifest-path sos/Cargo.toml
```

- Rust **stable**, MSRV **1.89**.
- 100 % documented public API (`#![deny(missing_docs)]`).
- Deterministic + property-based tests (seeded generators; no unseeded
  randomness, no wall-clock in hashed state).
- No `unsafe` (`#![forbid(unsafe_code)]`), no FFI.

> Note: SOS is not built by the repository's root CI (it is a separate,
> excluded workspace). A dedicated `sos` CI job is a P1 follow-up; until then the
> commands above are the gate, run locally and reported in the PR.
