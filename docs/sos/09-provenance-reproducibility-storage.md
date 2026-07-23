# 09 · Provenance, Reproducibility & Storage

> [← Workflow & Simulation](./08-workflow-and-simulation.md) · [Plugins, Backends & Interfaces →](./10-plugins-backends-interfaces.md)

These three substrate subsystems are the kernel's "filesystem," "version
control," and "package manager." Their mechanics are specified in
[SDE 03](../sde/03-object-model.md) and [SDE 06](../sde/06-provenance-and-reproducibility.md)
and inherited; this chapter states the storage architecture, the serialization /
hashing / versioning contracts the mandate calls out explicitly, and the one
genuinely new subsystem at SOS scope — the **Reproducibility Engine** (the Nix
analogy).

---

## 1. Storage architecture — the filesystem

The Storage Layer (`sos-store`) is a content-addressed object store, Git-shaped:

- **`put(obj) -> ObjectId` is idempotent** — storing an object you already hold is
  a no-op returning the same id. This is the root of dedup, memoization, and
  cross-lab agreement.
- **`get` / `has`** are the read path; **loose/pack** splits hot objects (loose)
  from cold history (packed), as Git does.
- **Large payloads are content-addressed side blobs** (tensors, datasets,
  checkpoints) referenced by hash, safetensors-style — the object graph stays
  small and diffable ([SDE 03 §7–8](../sde/03-object-model.md#7-the-object-store)).
- **Backends are pluggable**: an embedded local store for a laptop; an
  object-storage (S3-class) backend for a shared lab; the trait is small enough
  that an existing LIMS or data lake can back it.
- **GC is explicit and reachability-based** from named refs (programs,
  publications, tags). A discarded experimental branch is scientifically
  meaningful, so it is pruned only by an opt-in, logged GC — "nothing is lost by
  default; you may deliberately forget."

```rust
pub trait Store {
    fn put(&self, obj: &Object<Bytes>) -> ObjectId;   // idempotent
    fn get(&self, id: ObjectId) -> Option<Object<Bytes>>;
    fn has(&self, id: ObjectId) -> bool;
    fn put_blob(&self, bytes: &[u8]) -> BlobRef;       // large side payloads
    fn refs(&self) -> Vec<NamedRef>;                   // GC roots
}
```

The store's append-only, hash-chained discipline is the same one already in
`scirust-func-safety::audit`, `scirust-discovery::audit`, and
`scirust-sciagent::CcosLog` — SOS generalizes it from a log to a full DAG.

---

## 2. Serialization

Two forms, one normative (from [SDE 03 §4, §8](../sde/03-object-model.md#4-identity-hashing-and-canonical-serialization)):

- **Canonical binary** — deterministic (fixed field order, canonical numbers,
  sorted maps, NFC strings, omitted nulls). The *only* form used for hashing;
  byte-identical for equal objects on any machine.
- **JSON Lines interchange** — one object per line, pinned field order, for
  human diffing and tooling; identical discipline to
  `scirust-bench-schema`'s JSONL.
- **Blob side-format** — columnar / safetensors for bulk numeric payloads,
  referenced by content hash.

The wire format for plugins/MCP is the canonical binary, length-prefixed — one
format local and remote.

---

## 3. Hashing

- **`id = BLAKE3(domain_tag ‖ canonical(obj))`**, domain-separated per `Kind`
  (mirroring `scirust-provenance`'s `DIGEST_DOMAIN` discipline).
- Because `parents` are inside the hashed body, an id is a **Merkle hash over the
  entire lineage** — altering any ancestor changes every descendant id, so
  tampering is detectable end to end.
- **Level-aware hashing** reconciles hashing with the determinism taxonomy: L3
  integer/symbolic payloads hash exactly; L2 numeric payloads hash their
  *quantized* canonical form at the certified precision and attach a
  `Certificate`; L0 observations hash the *recording*. Nothing hashes a
  non-portable raw `f64` bit-pattern.

---

## 4. The determinism taxonomy, carried through

The four levels (L3 bit-exact · L2 numeric-within-ε · L1 statistical · L0
recorded) from [01 §6 of the SDE RFC](../sde/01-vision-and-philosophy.md#6-determinism-honestly-the-taxonomy)
are the spine of every reproducibility claim in SOS. They are **declared** by
plugins (reusing `scirust-bench-schema::Certificate.determinism`, D0–D3),
**propagated** as the minimum over any dependency path, and **certified** at L2
(a `bound_ulps` / tolerance / `κ_rt` round-trip bound). SOS grounds L3 on
`scirust-gpu`'s bit-exact `CpuBackend` oracle and on `scirust-modalg`'s exact
integer/finite-field algebra.

---

## 5. Versioning

- **`Kind = (name, schema_version)`** is part of every hash; **`version`** tracks
  content lineage ([03 §1](./03-object-model.md#1-the-universal-envelope)).
- **Old objects never rewrite.** The graph is poly-schema; readers keep old
  deserializers; new work produces new schemas. A **`Migration`** object records
  any re-encoding, so schema evolution is itself provenance.
- **A hash change to an unchanged object is a breaking change** and bumps
  `schema_version` — because it invalidates every downstream content address
  ([01 §5](./01-vision-and-principles.md#5-governance--the-rfc-process)).

---

## 6. The Provenance Engine — version control for reasoning

Provenance is not a subsystem bolted on; it **is** the edge set of SOS-IR
([SDE 06 §1](../sde/06-provenance-and-reproducibility.md#1-provenance-is-not-a-log--it-is-the-graph)).
The Provenance Engine (`sos-provenance`) adds three things over raw edges:

- **Environment capture** — the mandate's full traceability list (datasets,
  parameters, code, compiler, OS, CPU, GPU, Rust version, SciRust version,
  plugins, seeds) is captured into `ReproMeta`/`EnvRecord` and hashed into the
  `env_digest` ([03 §1](./03-object-model.md#1-the-universal-envelope)).
- **Signing** — Merkle/Lamport attestation via `scirust-provenance`
  (`sign_artifact` / `verify_artifact -> Verdict`, SHA-256, deterministic) for
  objects that must be attributable (`Publication`, `Decision`, `Review`). The
  honest caveats are inherited verbatim: this is provenance/attribution, not an
  anti-clone shield.
- **Cognitive attestation anchoring** — every CCOS proposal arrives with a
  `CcosLog` hash-chain entry (`sequence, model_version, input_hash, output_hash,
  prev_hash, chain_hash`). SOS anchors that chain into provenance, so even
  untrusted cognitive contributions are tamper-evidently recorded — you can prove
  *which* model produced a proposal and that the chain is unbroken
  (`CcosLog::verify()`).

---

## 7. The Reproducibility Engine — the package manager (Nix analogy)

This is the subsystem SOS adds beyond SDE, and the mandate's "Nix of
reproducibility" claim rests on it. Where the Provenance Engine *records* the
environment, the Reproducibility Engine **pins and re-realizes** it hermetically.

- **Hermetic environment pins.** A workflow's `env_digest` is a lockfile: exact
  toolchain (this repo's `rust-toolchain.toml`), exact backend crate versions +
  content hashes, hardware class. Re-execution binds the *same* pins or declares
  the drift — no "works on my machine."
- **`sos verify <object>`** re-executes the sub-DAG and diffs: it reports the
  first node (if any) whose realized value or determinism level violates its
  declaration — L3 nodes must match to the bit, L2 within their certificate, L1
  in distribution given the recorded seed, L0 identically by replay. A green
  `verify` is the machine-checkable form of "reproduced"
  ([SDE 06 §6](../sde/06-provenance-and-reproducibility.md#6-the-reproducibility-contract-stated-precisely)).
- **Pure-Rust is what makes this attainable.** Because the trusted path has no
  FFI (Invariant X), the build is hermetic and bit-stable; a NumPy/C++ dependency
  would make L3 re-execution impossible, which is precisely why such backends are
  confined to out-of-process, L0/L1-declared plugins behind the effect boundary.

```rust
pub trait Reproduce {
    fn pin(&self, workflow: ObjectId) -> EnvLock;                 // the "lockfile"
    fn verify(&self, object: ObjectId) -> VerifyReport;           // re-execute + diff
    fn rerun(&self, workflow: ObjectId, lock: &EnvLock) -> RunLedger;
}
```

---

## 8. The reproducibility contract (restated for SOS)

> **Contract.** Given any object `X`, an independent party who clones its sub-DAG
> and re-runs it under the pinned environment obtains objects whose ids match
> `X`'s **exactly at every L3 node**, **within the certificate at every L2 node**,
> **in distribution (given the recorded seed) at every L1 node**, and
> **identically-by-replay at every L0 node**. Any deviation is localized to a
> specific node and its declared level — never a mystery.

That contract is the whole point of the substrate: it turns "reproducible" from a
hope into a checkable property of a graph.

---

> [← Workflow & Simulation](./08-workflow-and-simulation.md) · [Plugins, Backends & Interfaces →](./10-plugins-backends-interfaces.md)
