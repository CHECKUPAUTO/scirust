# 05 · Reasoning Engine

> [← Knowledge Engine](./04-knowledge-engine.md) · [Curiosity Engine →](./06-curiosity-engine.md)

The Reasoning Engine (`sos-reasoning`) is the **compiler of the scientific OS**:
it transforms knowledge into conclusions deterministically, and — like a compiler
emitting debug symbols — every conclusion ships with a **derivation** that
explains it. It is the *verifier* in the propose/verify split: cognition may
suggest, but only the Reasoning Engine may conclude.

Rust is illustrative sketch.

---

## 1. The mandate: deterministic, LLM-free, always explained

Three hard requirements define this engine:

1. **No LLM.** No neural language model participates in a conclusion. Reasoning is
   symbolic, logical, graph-theoretic, and algebraic — reproducible bit-for-bit
   where the technique allows.
2. **Deterministic.** Same knowledge + same goal ⇒ same conclusion + same
   derivation, on any machine. All backing crates are deterministic by
   construction (`neuro-symbolic` logic engines, `scirust-symbolic`,
   `scirust-units`, `scirust-modalg` — the last three `#![forbid(unsafe_code)]`,
   no RNG, no wall-clock).
3. **Every conclusion carries a `Derivation`** ([§4](#4-every-conclusion-carries-a-derivation)).
   A result with no explanation is a bug, not an output.

```rust
pub trait Reason {                       // a "syscall" from sos-core
    fn derive(&self, goal: &Goal, kb: &KnowledgeView) -> Conclusion;
}
pub struct Conclusion {
    pub verdict: Verdict,                // Proven | Refuted | Undetermined
    pub soundness: Soundness,            // Proof | Check  (see §5 — honesty about strength)
    pub derivation: Derivation,          // the explanation — always present
    pub level: DeterminismLevel,         // L3 for exact/symbolic; lower only if numeric
}
```

---

## 2. The deterministic toolbox

The mandate lists nine reasoning techniques. Each maps to a concrete, existing
SciRust capability — SOS wraps, never re-implements (Invariant: no duplication).

| Technique | What it does | Backing API |
|---|---|---|
| **Logic** | forward/backward chaining, satisfiability | `neuro-symbolic::{DatalogEngine, DatalogRule, Atom, Term, Fact, SatSolver, SmtInterface}` |
| **Graph traversal** | derivation paths, reachability in the knowledge graph | `scirust-graph::{bfs, dfs, shortest_path}` + `neuro-symbolic::GraphReasoning` |
| **Constraint propagation** | narrow admissible parameter/state regions | `neuro-symbolic::CspSolver` |
| **Causal inference** | intervention/counterfactual reasoning over causal structure | `neuro-symbolic::CausalEngine` |
| **Symbolic manipulation** | simplify, differentiate, solve, rewrite | `scirust-symbolic::{simplify, diff, eval, solve_*}` + `neuro-symbolic::{EGraph, ENode, EClass}` (equality saturation) |
| **Dimensional analysis** | reject dimensionally inconsistent claims *before* data | `scirust-units::{Dimension, Quantity}` checked arithmetic |
| **Dependency analysis** | what a conclusion rests on; what breaks if a node is retracted | `scirust-graph` `dag` submodule (topological) over the provenance DAG |
| **Proof propagation** | chain lemmas/rules to establish a goal; exact algebra | `neuro-symbolic::{NeuralTheoremProver, RuleEngine, Rule}` + `scirust-modalg` (exact `BigInt`/`BigRational`/finite-field steps) |
| **Contradiction detection** | find incompatibilities across the graph | §3 |

`scirust-modalg` deserves emphasis: because it is **exact integer / modular /
finite-field algebra with no floating point** ("identical inputs give identical
outputs, bit-for-bit on every platform"), algebraic proof steps it performs are
**L3 sound**, not numerically approximate — the gold standard for a reasoning
kernel.

---

## 3. Contradiction detection (four levels, cheapest first)

Detecting incompatibility is central (the mandate names it, and the object model
has first-class `Contradiction`/`Refutation`). SOS checks at four levels, in
increasing cost:

1. **Dimensional (structural, pre-data).** A claim whose dimensions don't balance
   is refuted before any experiment — `scirust-units` returns a checked error on
   `try_add`/`try_sub` of mismatched `Dimension`s. Cheapest possible refutation.
2. **Symbolic.** `scirust-symbolic::prove_equal` and e-graph equality saturation
   flag when two laws cannot both hold, or when a derivation contradicts an
   established equation.
3. **Logical.** Datalog/SAT: assert both claims plus the rule base; an
   unsatisfiable core is a contradiction with a witness (`neuro-symbolic::SatSolver`).
4. **Causal.** Two theories predicting opposite responses to the same
   intervention conflict — `neuro-symbolic::CausalEngine`.

Each detected conflict is recorded as a `Contradiction` object linking the
offending nodes (never a silent deletion), and surfaces to the Curiosity Engine
as a research lead ([06](./06-curiosity-engine.md)).

---

## 4. Every conclusion carries a `Derivation`

The explanation is not a nicety; it is the output's justification and the reason
SOS is auditable. A `Derivation` is an immutable object — a proof DAG:

```rust
pub struct Derivation {
    pub goal: Goal,
    pub steps: Vec<Step>,          // each: (rule/technique, premises: Vec<ObjectId>, result)
    pub premises: Vec<ObjectId>,   // the knowledge nodes consumed (leaves)
    pub soundness: Soundness,      // Proof | Check
}
```

Because each `Step` cites the `ObjectId`s of its premises and the rule applied,
"why does SOS conclude X?" is answered by walking the derivation to its knowledge
leaves — a complete, reproducible chain, not a narrative. The derivation is
itself content-addressed and stored, so an explanation can be cited, diffed, and
re-verified independently (`sos verify <derivation>` re-runs the steps).

---

## 5. Honesty about strength: `Proof` vs. `Check`

A reasoning kernel that overstates its certainty is worse than none. SOS labels
every conclusion's `Soundness`, mirroring the workspace's certificate culture:

| Soundness | Meaning | Examples |
|---|---|---|
| **`Proof`** | sound and (within its logic) complete — if it says proven, it is | e-graph equality saturation; SAT-refutation with an unsat core; exact `scirust-modalg` algebra; dimensional *inconsistency* (a mismatch is a real refutation) |
| **`Check`** | deterministic but **incomplete** — evidence, not proof | `prove_equal`'s fixed 200-point numerical agreement grid (can miss where samples don't separate); dimensional *consistency* (necessary, not sufficient); a bounded-depth proof search that timed out |

This distinction is disclosed on the object, propagated to any `Confidence` that
consumes it, and never elided. A `Check`-level equality is a strong lead the
planner may act on; it is not licensed to be stated as a theorem. Turning a
`Check` into a `Proof` (e.g. replacing sampled equality with an exact `modalg`
argument) is itself recordable progress.

---

## 6. The Reasoning Engine as verifier (Invariant IX in practice)

The propose/verify split ([02 §4](./02-system-architecture.md#4-the-two-backends-and-the-proposeverify-data-flow))
lives here. When the cognitive backend proposes — say CCOS suggests "these two
domains obey the same law" — the Reasoning Engine does not trust it. It attempts:

1. a **dimensional** check (do the quantities even match?),
2. a **symbolic** check (`prove_equal` / e-graph — are the equations equivalent?),
3. a **logical** check (is the analogy consistent with the rule base?).

Only if the claim **survives** does it become a trusted knowledge node, carrying
its derivation and its `Soundness`. If it fails, a `Refutation` is recorded and
the proposer's attestation (`CcosLog` chain) is linked, so a pattern of bad
proposals is itself visible data. Cognition widens the search; determinism keeps
it honest.

---

## 7. Scope and non-goals of the engine

- It is **not** a general automated theorem prover for open mathematics; it is a
  practical reasoner over the SOS knowledge graph, strong where the backing
  engines are strong (Datalog, CSP, e-graphs, exact algebra, dimensional
  analysis) and honest (`Check`) where they are heuristic.
- It does **not** learn. Any learning lives in the (untrusted, proposer)
  cognitive backend; the reasoner's rules are explicit, versioned knowledge nodes.
- Its determinism is **bounded by its inputs**: reasoning over an L2 numeric
  premise yields at best an L2 conclusion, and the level is propagated, never
  laundered.

---

> [← Knowledge Engine](./04-knowledge-engine.md) · [Curiosity Engine →](./06-curiosity-engine.md)
