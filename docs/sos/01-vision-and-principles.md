# 01 · Vision & Principles

> [← README](./README.md) · [System Architecture →](./02-system-architecture.md)

---

## 1. Vision

Scientific software executes calculations. SOS executes the **scientific
method**. The difference is the whole project.

A solver answers "what is the solution to this equation?" SOS answers a
different class of question: *"Given everything we know, which hypothesis is
least supported, what experiment would most reduce our uncertainty about it, what
does the evidence then force us to believe, and can every step be reproduced and
explained?"* That is not a numerical task; it is a **reasoning** task over a body
of knowledge — and it is the task no current tool treats as first-class.

SOS's bet is that the scientific method is a **computation**: a set of
transformations over immutable, content-addressed objects, related by an
append-only DAG, executed by cooperating engines, and pinned to a reproducible
environment. Treat it as one, and reproducibility, auditability, composability,
and eventually *automated discovery* stop being aspirations and become
structural properties — the way Git made history reproducible and LLVM made
compilation retargetable.

> **SOS is to scientific reasoning what an operating system is to computation:**
> a kernel of trusted abstractions (objects, hashing, determinism, capabilities),
> a scheduler (workflows), a filesystem (the object store), long-term memory
> (the cognitive backend), drivers (plugins), and userland (CLI, agents,
> publications) — with the scientific method running on top as its processes.

---

## 2. What SOS is not

Stated sharply, because the surrounding software culture will assume otherwise:

- **Not an AI framework and not an ML framework.** SOS uses machine learning
  *as a wrapped backend capability* (SciRust) and cognition *as a proposer*
  (CCOS), but its own reasoning core is deterministic and LLM-free. It adds no
  modelling library of its own.
- **Not a notebook.** Notebooks record what a human did, mutably and
  irreproducibly. SOS records what *reasoning* did, immutably and reproducibly.
- **Not a numerical library.** It never re-implements a solver, a statistic, or
  an embedding. It wraps SciRust and CCOS and *orchestrates* them.
- **Not an oracle.** It cannot certify that a domain model is true — only that
  the reasoning over it was consistent, reproducible, auditable, and explained.

---

## 3. The invariants

SOS inherits the [seven invariants of the SDE RFC](../sde/01-vision-and-philosophy.md#4-design-philosophy--the-seven-invariants)
— everything is an explicit object; immutable; content-addressed and
deterministically hashed; reproducible and honest about *which kind*; versioned;
no opaque heuristics; a thin explicit effect boundary — and adds four that are
specific to being an *operating system* with two backends.

### VIII. Backend independence is structural
SciRust (computational) and CCOS (cognitive) are the *preferred* backends, not
required ones. The kernel and every engine depend only on `sos-core` traits;
`scirust-*` and CCOS appear in exactly two adapter crates (`sos-scirust`,
`sos-ccos`). A build with neither backend still compiles and runs — the engines
fall back to deterministic defaults. This is enforced by a dependency-lint in
CI, not merely intended ([11 §5](./11-workspace-and-crate-graph.md#5-dependency-invariants-enforced-in-ci)).

### IX. Propose with cognition, verify with deterministic reasoning
This is the load-bearing separation that lets SOS use an LLM-backed cognitive
layer *and* keep an LLM-free reasoning core. Cognitive backends (CCOS,
`scirust-sciagent`) may **propose** — questions, candidate hypotheses, analogies
— but may **never** be the reasoner, ranker, or judge. Every cognitive proposal
(a) is **attested** by a hash-chained record (CCOS's `CcosLog`:
`input_hash → output_hash → chain_hash`, [10 §3](./10-plugins-backends-interfaces.md#3-the-cognitive-backend-sos-ccos)),
and (b) enters the graph as an *untrusted* object that the deterministic
Reasoning Engine must independently validate before it influences any
conclusion. Cognition supplies candidates; determinism supplies verdicts.

### X. Pure Rust, no FFI, no gratuitous `unsafe`
The entire system is pure, stable Rust. **No FFI** (no C/C++/Fortran/Python in
the trusted path) — because FFI is the classic source of hidden nondeterminism,
memory unsafety, and un-reproducible builds, all of which are existential threats
to SOS's guarantees. **No `unsafe`** except where it is *mathematically
justified* (a documented, tested, bounded invariant — e.g. a proven-in-bounds
SIMD kernel) and never for convenience. This is why SciRust (pure-Rust,
`forbid(unsafe)` in its numeric leaf crates) is the natural backend and why a
NumPy/C++ stack is relegated to an out-of-process, L0/L1-declared plugin behind
the effect boundary rather than the trusted core.

### XI. No unfinished work crosses a phase boundary
**No TODO. No placeholder. No stub. No mock presented as production.** Every
merged increment is production-ready: documented, tested, `clippy -D warnings`
clean, deterministic. A subsystem that is not ready is *not merged* — it lives on
a branch. "Production-ready each phase" is a release gate, not a slogan
([12 §2](./12-engineering-and-roadmap.md#2-coding-standards-the-gate)).

---

## 4. Why these constraints buy more than they cost

The invariants are demanding; each pays for itself in a property SOS could not
otherwise have:

| Constraint | The property it buys |
|---|---|
| Immutable, content-addressed objects | Reproducibility, dedup, tamper-evidence, and `git`-style clone/diff/merge of *reasoning*. |
| Append-only DAG, nothing destroyed | Total provenance; the file-drawer of discarded hypotheses becomes visible, not lost. |
| Deterministic + determinism-level taxonomy | Reproducibility claims that are *honest* (L0–L3), propagated, and certified — never binary and aspirational. |
| Pure Rust / no FFI | Hermetic, bit-stable builds; the reproducibility guarantee is actually attainable. |
| Propose/verify separation | The power of LLM ideation *without* importing its irreproducibility into conclusions. |
| No opaque heuristics | Every conclusion carries an explanation (a derivation/proof object) — auditability by construction. |

---

## 5. Governance & the RFC process

To be twenty-year infrastructure, SOS needs a boring, predictable evolution
contract — the same one the SDE RFC established, lifted to SOS scope:

- **The stability surface** is (a) the object envelope and core schemas in
  `sos-core`, (b) the engine trait ABI (the "syscalls"), and (c) the on-disk /
  wire serialization. These change only through a numbered **SOS-RFC** (this is
  RFC-0002; the SDE architecture is RFC-0001) with a migration note and a schema
  version bump.
- **Determinism is a compatibility property.** Any change that alters the hash of
  an unchanged object is breaking and bumps that object's schema version, because
  it invalidates every downstream content address.
- **Evolution is append-only, including deprecation.** Removing an engine or a
  plugin never removes the objects it produced; they remain valid, replayable
  nodes pinned to the version that made them. The promotion of `sde-core` into
  `sos-core` described throughout this RFC is itself an instance of this process —
  recorded, versioned, non-destructive.

---

## 6. Success criteria

SOS succeeds when three things are ordinary:

1. **"Send me the reasoning graph"** replaces "send me the code, the data, and
   good luck" — and the recipient re-runs it to identical results at every
   deterministic node.
2. **The system asks its own questions.** The Curiosity Engine surfaces "these
   two disconnected domains share a mathematical structure" or "this theory's
   parameter is unconstrained," and a researcher acts on a machine-generated,
   provenance-bound research lead.
3. **Every conclusion explains itself.** "Why does SOS believe this?" resolves to
   a deterministic derivation — a proof object — not a person's recollection or a
   model's confidence.

---

> [← README](./README.md) · [System Architecture →](./02-system-architecture.md)
