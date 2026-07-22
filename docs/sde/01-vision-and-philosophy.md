# 01 · Vision & Philosophy

> [← README](./README.md) · [Architecture Overview →](./02-architecture.md)

---

## 1. The problem SDE exists to solve

Science has a reproducibility crisis and a **provenance crisis**, and the second
causes much of the first. When a result cannot be reproduced, the usual
post-mortem discovers that the *reasoning that produced it was never recorded*:
which hypotheses were in play, what the prior was, which analysis was chosen
(and how many were silently discarded), what seed the simulation used, which
solver version, which compiler, which machine. The scientific *artifact* (a
figure, a p-value, a fitted parameter) survives; the **process** that justifies
it evaporates.

Software solved a structurally identical problem twice:

- **Source code** used to be un-reproducible — until Git made history an
  immutable, content-addressed DAG that anyone can clone, diff, and replay.
- **Compilation** used to be per-vendor and per-architecture — until LLVM
  introduced a stable IR with pluggable frontends and backends, so a new
  language or a new chip is a plugin, not a rewrite.

Discovery deserves the same treatment. SDE's thesis is that **the scientific
method is a computation** — a pipeline of typed transformations over immutable
objects — and that treating it as one buys reproducibility, composability, and
automation the same way Git and LLVM did for their domains.

---

## 2. Vision

> **A researcher describes a scientific problem once — as objects and
> plugins — and the engine can, forever after, re-derive every conclusion,
> explain every step, quantify every belief, and recommend the single most
> informative next experiment.**

Concretely, when SDE is mature:

1. **A discovery is a repository.** `sde clone` a study and you get its entire
   reasoning DAG: every question, hypothesis, prediction, experiment,
   observation, and the belief updates between them — re-runnable to identical
   outputs where components are deterministic.
2. **A theory is a versioned object with a changelog.** Each revision cites the
   evidence that forced it. "Why do we believe X?" resolves to a path through
   the DAG, not a person's memory.
3. **The next experiment is a recommendation, not a guess.** The planner ranks
   candidate experiments by expected information gain per unit cost and can
   defend the ranking numerically.
4. **A domain is a plugin.** Physics, finance, biology, ML, or robotics enters
   by implementing one small contract — not by forking the engine.
5. **A backend is a plugin.** SciRust is the default compute substrate; a Python
   stack, an HPC cluster, or a wet-lab robot attaches through the same adapter
   seam.

---

## 3. Who it is for

| Audience | What SDE gives them |
|---|---|
| **Computational & simulation scientists** | A reproducible harness that versions every run, its seed, and its environment; a planner that spends compute where it buys the most information. |
| **Experimental labs (wet-lab, hardware, robotics)** | Pre-registered, hashed analysis plans (p-hacking becomes structurally hard); record/replay of instrument observations; an auditable belief ledger. |
| **ML / AutoML researchers** | Model search and Bayesian optimization expressed as first-class hypothesis-ranking and experiment-planning, with full provenance of every trial. |
| **Quantitative & applied fields (finance, engineering, chemistry)** | Domain-agnostic machinery for competing-model comparison, confidence estimation, and contradiction detection. |
| **Meta-scientists & reviewers** | A machine-checkable record: re-execute a paper's figures from its DAG; inspect what was tried and not reported. |

---

## 4. Design philosophy — the seven invariants

These are **non-negotiable**. Every crate, trait, and object is judged against
them. They are the spec that the rest of this document elaborates.

### I. Everything is an explicit object
No hidden state. A prior, a likelihood, a stopping rule, a random seed, a
solver tolerance — each is a named, serializable object, not an implicit
argument or a global. If it influences a conclusion, it is in the graph.

### II. Everything is immutable
Objects are never mutated in place. A "revision" is a *new* object that points
at its parent. History is append-only, exactly as in Git. This is what makes
provenance total and replay possible.

### III. Everything is content-addressed and deterministically hashed
An object's identity is the hash of its canonical serialization (including the
IDs of its inputs). Identical reasoning yields identical IDs on any machine.
This gives free deduplication, tamper-evidence, and cross-lab agreement on
"the same object."

### IV. Everything is reproducible — and honest about *which kind*
Re-running an identical workflow reproduces identical outputs **whenever
deterministic components are used.** Where bit-exactness is unattainable
(cross-hardware floating point, GPU reductions, genuinely stochastic
instruments), the engine does not pretend — it records a **determinism level**
and the tolerances/certificates that bound the difference (see §6).

### V. Everything is versioned
Objects, plugins, domains, the engine, *and* the environment carry versions.
A workflow pins the exact versions it used; a rerun either matches them or
declares the drift.

### VI. No opaque heuristics
Any component that ranks, selects, or scores must be able to explain its output
in terms of the objects it consumed — a number, a distribution, or a
derivation, never "the model felt confident." Heuristics are allowed; *opaque*
heuristics are not. An LLM hypothesis generator is fine — its outputs enter the
DAG as ordinary, criticizable hypotheses with recorded prompts and seeds.

### VII. The pure/effect boundary is explicit and thin
All computation is pure except a single, clearly-marked boundary — `Execution`
and `Observation` — where the engine contacts the world. Effects are performed
through a capability object and **recorded**, so replay reuses the recording.
This is what lets a fundamentally stochastic experiment still sit inside a
reproducible workflow.

---

## 5. Consequences of the invariants (the good kind)

The invariants are demanding, but they pay for themselves:

- **Pre-registration is free.** Because an experiment's design and its analysis
  plan are immutable objects hashed *before* execution, you cannot retrofit the
  analysis to the data. The hash is the pre-registration. (This repo already
  pre-registers studies by hand — see `docs/research/ANEE_PHASE_D_PREREGISTRATION`;
  SDE makes it structural.)
- **The file-drawer disappears.** Every hypothesis considered and every analysis
  branch tried is a node in the DAG, whether or not it "worked." Selective
  reporting is visible as an unreferenced subgraph.
- **Incremental recompute is free.** Content-addressing means an unchanged
  sub-DAG is never recomputed — the same mechanism that gives reproducibility
  gives caching (see [04](./04-workflow-engine.md)).
- **Collaboration is `git`-shaped.** Two labs studying the same question hold
  two branches of one object graph; reconciling them is a merge with explicit
  conflict objects, not an email thread.
- **Auditability is total.** "Show me every input to this conclusion, transitively"
  is a graph traversal, and the answer is complete by construction.

---

## 6. Determinism, honestly (the taxonomy)

Reproducibility claims are worthless if they are binary and aspirational.
Borrowing the *certificate* and *determinism-level* culture already present in
this workspace (CANR §6.1; `scirust-bench-schema`'s mandatory `seed` and
optional `Certificate`), every SDE component **declares** one of four levels,
and the engine propagates the weakest level along any dependency path.

| Level | Guarantee | Typical source |
|---|---|---|
| **L3 · Bit-reproducible** | Byte-identical output for identical input, seed, and object versions — on *any* conforming machine. | Pure integer/symbolic code; fixed-order f64 with a pinned reduction order. |
| **L2 · Numerically-reproducible** | Identical up to a declared tolerance `ε`; a certificate bounds the deviation. | Cross-hardware BLAS, GPU reductions, iterative solvers to a tolerance. |
| **L1 · Statistically-reproducible** | Identical *distribution* given the recorded seed and sampler; summary statistics reproduce within CI. | Monte Carlo, stochastic optimization, sampling-based inference. |
| **L0 · Non-deterministic (recorded)** | Not reproducible in principle (a physical measurement, a live feed, a human) — but the *observation is recorded*, so downstream replay is L3 against the recording. | Wet-lab instruments, markets, human judgment. |

The point is not to force everything to L3. It is to make the level **explicit,
propagated, and certified**, so a consumer of any object knows exactly what
"reproducible" means for it. A workflow's overall level is the minimum over its
realized path — and that number is itself an object in the DAG.

---

## 7. Relationship to existing SciRust work

SDE is layered *on top of* SciRust and is deliberately continuous with its
existing idioms rather than a foreign body:

- **Records-as-types with mandatory seeds** — SDE objects generalize
  `scirust-bench-schema::BenchRecord` (the CANR §9 "one row shape, seed
  required" pattern) from benchmarks to the entire pipeline.
- **Deterministic content signing** — SDE's provenance layer reuses
  `scirust-provenance` / `scirust-license::hashsig` (SHA-256 Merkle/Lamport) for
  tamper-evident, court-usable object attestation.
- **Hash-chained audit logs** — the append-only object store mirrors the
  hash-chain discipline already in `scirust-func-safety::audit` and
  `scirust-discovery::audit`.
- **Consent-scoped capabilities** — the effect boundary's authorization model is
  patterned on `scirust-discovery`'s `ScopeAuthorization` (signed, time-boxed,
  least-privilege), so "run this experiment" is an authorized, logged action.
- **A research culture to formalize** — `docs/kb/` and `docs/research/` are the
  manual precursor; SDE is their executable form.

> **Naming note (important):** the crate `scirust-discovery` already exists and
> means *OT/IT network asset discovery* (protocol-native, Nmap-safe probing) —
> **not** scientific discovery. SDE therefore lives in its own `sde-*` crate
> namespace to avoid collision. See
> [09-workspace-and-crates.md](./09-workspace-and-crates.md#naming-and-the-scirust-discovery-collision).

---

## 8. Governance & stability

To become infrastructure ("the Git of discovery"), SDE must have a boring,
predictable stability contract:

- **The stability surface** is (a) the object envelope and the core object
  schemas in `sde-core`, (b) the extension traits in `sde-core`/`sde-registry`,
  and (c) the on-disk/wire serialization. These change only through a numbered
  **SDE-RFC** (this document is RFC-0001), with a migration note and a schema
  version bump.
- **Everything else** — individual plugins, the `sde-scirust` mappings, the CLI
  porcelain — evolves freely under semantic versioning.
- **Determinism is a compatibility property.** A change that alters the hash of
  an unchanged object is a breaking change and must bump the object's schema
  version, because it breaks every downstream ID.
- **Deprecation is append-only too.** Removing a plugin does not remove the
  objects it produced; they remain valid, replayable nodes pinned to the
  plugin version that made them.

---

## 9. What success looks like

SDE succeeds if, five years on, three sentences are ordinary:

1. *"Send me the discovery graph"* has replaced *"send me the code and the data
   and good luck."*
2. *"The planner says the tumor-growth assay buys 2.3 bits more than the
   RNA-seq run at a third of the cost, so we're doing the assay"* is a normal
   thing to say in a lab meeting, with the numbers attached.
3. *"Which experiment should we run next?"* has a **defensible, reproducible,
   quantitative** answer — and the answer, and the reasoning behind it, are
   themselves objects you can cite.

---

> [← README](./README.md) · [Architecture Overview →](./02-architecture.md)
