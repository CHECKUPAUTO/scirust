# Temporal tensor — the "fever curve" of a conflict-resolution engine

> Reproduce: `cargo run --release --example temporal_tensor_crux`

CCOS is not a social network, so *structural centrality over time* is the wrong signal — the
eigencentrality and spectral-region measurements (`docs/MEASUREMENT_eigencentrality.md`,
`docs/MEASUREMENT_spectral_regions.md`) found that direction flat / weak on CCOS's own small,
densely-coupled graph. CCOS is a **conflict-resolution engine**, so the quantity worth watching over
time is the **thermodynamics of belief**: how a claim's **Belief** (`B`, signed ∈ [−1, 1]) and
**Tension** (`T = QBelief.conflict` ∈ [0, 1], the geometric evidence balance) evolve when a
contradiction is injected, propagated, and decayed.

This records the **dynamic-profile tensor** `Θ[node, component, t]`, `component ∈ {Belief, Tension}`,
across a scripted **Conflict-of-Origins** crisis.

## Method

Two origins each *cause* the same three decisions: `origin_A` (a believed source) and `origin_B` (the
conflicting origin). Fully deterministic — logical clock (`tick`), sorted propagation, no RNG — so
`replay == live` holds:

1. **consensus** — assert support on A; `propagate_beliefs` lends a derived support to each decision
   (calm, believed, one-sided).
2. **injection (t₄)** — the conflicting origin B arrives and is refuted (the "false info").
3. **propagation (t₅)** — B's refutation propagates one causal hop onto the shared decisions, which now
   carry **both** a supporting and a contradicting surface → tension spikes.
4. **relaxation (t₇₊)** — no reinforcement; the clock advances, the knowledge half-life
   (`qbelief_decayed`) fades every surface, and belief + tension relax back toward neutral.

## Result

```
── TENSION  Θ[·, Tension, t]   (▁ calm … █ contested) ──
  origin   origin_A      ▁▁▁▁▁▁▁▁▁▁▁   peak 0.00
  origin   origin_B      ▁▁▁▁▁▁▁▁▁▁▁   peak 0.00
  decision set_timeout   ▁▁▁▁▁▄▄▄▃▃▂   peak 0.49
  decision set_retries   ▁▁▁▁▁▄▄▄▃▃▂   peak 0.49
  decision set_pool_size ▁▁▁▁▁▄▄▄▃▃▂   peak 0.49

── SYSTEM TEMPERATURE  mean tension over the 3 decisions ──
  ▁▁▁▁▁▄▄▄▃▃▂   0.00 0.00 0.00 0.00 0.00 0.49 0.49 0.41 0.33 0.26 0.20
                            └ t4 inject  └ t5 fever        └ decay relaxes
```

## Verdict

The signal is **sharp and legible** — flat → spike → relax — so `Θ[node, {Belief, Tension}, t]` is a
real primitive (where temporal *centrality* would have been flat on this graph):

- **The origins stay cool.** Each is internally one-sided (A believed, B refuted), tension ~0. The heat
  emerges **only at the decisions that depend on both** — the thermodynamic signature of a *conflict of
  origins*: a source isn't contested, a *decision that rests on conflicting sources* is.
- **A synchronized fever.** When B's refutation propagates (t₅), all three shared decisions spike
  together (0 → 0.49) — a client can watch *exactly which decisions a conflicting source contaminates*.
- **The fever breaks on its own.** With no reaffirmation the half-life decays every surface and the
  system relaxes (0.49 → 0.20): the decay mechanism *resolves* the crisis, no operator action needed.
- **No cascade.** Once contested, a decision is unresolved, so the propagation wavefront stops — the
  engine **localizes** a conflict instead of spreading a storm (the safety property, seen
  thermodynamically).

This is the client-facing signal the centrality/spectral bricks lacked: a real-time **fever chart** of
the knowledge base facing injected misinformation, plus the decay-driven recovery. The next slice
productionizes it as a `spectral::temporal_profile` primitive + a CLI/MCP surface; this note records
that the measurement justifies building it.
