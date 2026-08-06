# Measurement — Propagation crux (belief revision across the causal graph)

*A claim with no evidence of its own should still take on a belief from the causes it depends on.*

Reproduce: `cargo run --release --example propagation_crux`

Slice 3 of the Q-Page contested-knowledge work, building on `qbelief` (slice 1). A static evidence
store is *local*: a claim's belief comes only from its own incoming `Supports`/`Contradicts` edges,
so a claim with no direct evidence stays neutral (`belief 0`) forever — even when the thing it
*depends on* has been firmly resolved. `MemoryGraph::propagate_beliefs(resolve_threshold, damping)`
runs **one deterministic hop** over the `Causes` edges: every *resolved* cause (`|belief| ≥
resolve_threshold`) emits a **derived, attenuated** edge on its effect — a `Supports` from a believed
cause, a `Contradicts` from a refuted one (`weight = edge.weight · damping · |belief|`). It is a
read-only-then-add pass (collect, sort, dedup), so it is deterministic and idempotent.

## Fixture

Three causes, each with an effect that has **no evidence of its own**:
- **A** resolved-true (`belief +0.75`) `Causes` **B**, which in turn `Causes` **C** (a 2-hop chain).
- **D** resolved-false (`belief −0.75`) `Causes` **E**.
- **F** unresolved / balanced (`belief 0`) `Causes` **G**.

`propagate_beliefs(threshold 0.7, damping 0.6)`.

## Result (measured)

```
  effect                                 belief before   after one hop
  B — effect of A (resolved-true)        +0.00        +0.31
  C — effect of B (2 hops from A)        +0.00        +0.00
  E — effect of D (resolved-false)       +0.00        -0.31
  G — effect of F (unresolved cause)     +0.00        +0.00
                                            (2 derived edges added)
```

## Reading

**B and E inherit a belief from their causes** — despite having no direct evidence. A *true* cause
lends support (`B → +0.31`); a *refuted* cause lends contradiction (`E → −0.31`). This is belief
revision across the causal graph: the kind of inference a vector index or a flat evidence store
cannot perform, because the link is structural (a `Causes` edge), not lexical, and the effect shares
no evidence with its cause.

Two properties keep it safe:
- **Attenuated.** The induced belief is strictly weaker than its cause's: `|0.31| < |0.75|`. The
  effect is a *hint* from its cause, never as strong as direct evidence.
- **Bounded — the wavefront stops.** Because `0.31` is below the resolve threshold `0.7`, **C** (two
  hops from A) stays at `0`: a single resolved source cannot cascade into a storm. **G** stays at `0`
  too, because its cause **F** is unresolved (balanced) — there is nothing to propagate.

So propagation reaches dependent claims a static store can't, while damping + the threshold give it a
natural, deterministic stopping condition rather than an unbounded cascade.

## What this does and does not claim

- **Does:** show that belief can be *derived* across `Causes` edges — an evidence-less effect takes on
  a weaker, correctly-signed belief from a resolved cause, recorded as ordinary (derived)
  `Supports`/`Contradicts` evidence so it stays auditable and is folded into `qbelief` like any other
  surface; and that attenuation + the resolve threshold bound the spread to (here) a single hop.
- **Does not:** model **multi-hop** accumulation (a chain only advances one link per call; convergent
  multi-hop propagation, where many weak hops can add up, with a **scheduler** to bound the cascade
  and an `Op` so it survives replay, is the next slice), nor claim a specific causal-inference
  semantics beyond "a resolved cause is (attenuated) evidence about its effect."
