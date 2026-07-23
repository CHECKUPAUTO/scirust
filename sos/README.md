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
| **P1 — Kernel & substrate** | `sos-core`, `sos-store`, `sos-provenance`, `sos-registry` (+ SOS CI) | substrate **done**; `sos-repro` pending (needs the workflow engine for full `verify`/`rerun`) |
| **P2 — Knowledge & Reasoning** | `sos-knowledge`, `sos-reasoning` | **done** (deterministic cores landed; Datalog / e-graph / theorem-proving deferred to `sos-scirust` per Invariant VIII) |
| **P4 — Curiosity & Theory** | `sos-curiosity`, `sos-theory` | **cores landed** (both need only the P2 substrate; information-gain / analogy / Bayes-factor ranking / discriminating-experiment planning await P3's `sos-planner` and `scirust-*` per Invariant VIII) |

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
- **`sos-knowledge`** — the Knowledge Engine (typed semantic graph). First-class
  relation [`Edge`](sos-knowledge/src/edge.rs)s (a typed
  [`Relation`](sos-knowledge/src/relation.rs) between two objects, sealed as
  content-addressed objects) and a deterministic
  [`KnowledgeGraph`](sos-knowledge/src/graph.rs) view with structural queries —
  `neighbors`, `in_neighbors`, `related`, shortest `path`. (Datalog / e-graph /
  analogy-by-isomorphism reasoning is deferred to `sos-reasoning` + `sos-scirust`
  per Invariant VIII.)
- **`sos-reasoning`** — the Reasoning Engine (deterministic, **LLM-free** core).
  Sound entailment over the knowledge graph — a directly-asserted edge, or a
  chain of a **transitive** relation — via [`Reason::entails`](sos-reasoning/src/reason.rs),
  returning a [`Conclusion`](sos-reasoning/src/reason.rs) whose
  [`Derivation`](sos-reasoning/src/derivation.rs) is itself a content-addressed,
  re-verifiable object that cites the exact edges used. Every result carries an
  honest [`Soundness`](sos-reasoning/src/soundness.rs) label (`Proof` vs a
  deterministic `Check`), and "not found" is `Undetermined`, never a false
  disproof. [`Reason::contradictions`](sos-reasoning/src/reason.rs) surfaces
  incompatibilities (asserted `contradicts` edges and mutual-`supersedes` cycles)
  as first-class [`Contradiction`](sos-reasoning/src/contradiction.rs) objects.
  (Datalog inference, SAT/SMT, e-graph saturation, theorem proving, and analogy
  by subgraph isomorphism are deferred to `sos-scirust` per Invariant VIII.)
- **`sos-curiosity`** — the Curiosity Engine (the OS **idle daemon**;
  deterministic, **LLM-free**). [`BeCurious::sweep`](sos-curiosity/src/sweep.rs)
  scans the knowledge graph and emits ranked
  [`ScientificQuestion`](sos-curiosity/src/question.rs)s, each a content-addressed
  object grounded in the real nodes it concerns and carrying a `Derivation`
  explaining *why* it is worth asking. Three deterministic lenses
  ([`Strategy`](sos-curiosity/src/strategy.rs)): **contradiction-hunt** (reusing
  `sos-reasoning`'s contradiction detection), **under-connected** (weakly-linked
  nodes), and **weakly-supported** (claims refuted yet unsupported). Scoring is an
  explicit, versioned [`CuriosityPolicy`](sos-curiosity/src/policy.rs) —
  **integer fixed-point, saturating** (bit-exact `L3` ranking, no opaque
  priorities, overflow-proof). (Information-gain scoring via `sos-planner`,
  cross-domain analogy via `scirust-graph`, and cognitive proposals via
  `sos-ccos` are deferred per Invariant VIII.)
- **`sos-theory`** — the Theory Engine (deterministic). Theories are
  **first-class, immutable, evolving** objects: a
  [`Theory`](sos-theory/src/theory.rs) records all ten mandate fields (axioms,
  assumptions, equations, [`Scope`](sos-theory/src/scope.rs) domain of validity,
  supporting **and** contradicting evidence, confidence, citations, revision
  parent, competitors) as ids into the graph — a view over provenance, not a
  document. [`Theory::revise`](sos-theory/src/theory.rs) evolves a theory into a
  *new* node that **retains its anomalies** (contradicting evidence is never
  hidden) and links its parent; the [`Theories`](sos-theory/src/engine.rs) engine
  walks the full [`revision_chain`](sos-theory/src/engine.rs) (old theories stay
  queryable) and [`compare`](sos-theory/src/engine.rs)s rivals over their shared
  domain, so competitors coexist rather than being forced to a single winner.
  (Bayes-factor `Confidence` ranking and discriminating-experiment planning are
  deferred to the statistics backend + `sos-planner` per Invariant VIII.)

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

> SOS is a separate, excluded workspace, so the repository's root CI does not
> build it. A dedicated **SOS CI** workflow
> ([`.github/workflows/sos-ci.yml`](../.github/workflows/sos-ci.yml)) gates it
> upstream with the commands above — fmt (on the repo's pinned nightly, since
> `rustfmt.toml` uses unstable options), clippy `-D warnings`, `test` on stable,
> and an MSRV 1.89 check — path-filtered to run only when `sos/` changes. The
> workspace's `Cargo.lock` is committed so CI builds with `--locked` are
> reproducible.
