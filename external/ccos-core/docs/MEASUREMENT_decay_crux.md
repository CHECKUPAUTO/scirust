# Measurement — Decay crux (knowledge half-life)

*Why a stale, never-reaffirmed dissent should stop deadlocking a claim.*

Reproduce: `cargo run --release --example decay_crux`

Slice 2 of the Q-Page contested-knowledge work, building on `qbelief` (slice 1). `MemoryGraph::qbelief`
weighs every assertion **equally and forever**, so a single old objection that was never revisited
keeps a claim looking *eternally contested* even after fresh evidence has arrived on the other side.
`MemoryGraph::qbelief_decayed(claim, half_life)` fades each evidence edge by `0.5^(age / half_life)`
— `age` is the clock ticks since the edge was asserted — so a fresh (re-)assertion outweighs an
ageing one. It is **lazy and pure** (computed on demand from each edge's `created_at` vs the current
`clock`, no stored decay state), so it is deterministic and `replay == live` holds.

## Fixture

A claim is **contested at `t = 0`** (one `Supports`, one `Contradicts` edge). The support is then
re-affirmed by a *fresh* assertion at time `T`, while the contradiction is **never revisited** — a
one-off objection that turned out to be wrong. We sweep `T` (the objection's age) at `half_life = 10`
ticks and compare plain vs decayed belief.

## Result (measured)

```
    T      PLAIN qbelief         DECAYED qbelief
         belief  conflict      belief  conflict   (objection age = T)
    0      0.00     0.67       0.00     0.67
    5      0.00     0.67       0.11     0.62
   10      0.00     0.67       0.20     0.57
   20      0.00     0.67       0.33     0.44
   40      0.00     0.67       0.45     0.24
   80      0.00     0.67       0.50     0.06
```

## Reading

**Plain `qbelief` is frozen.** One un-reaffirmed objection counts as much as the fresh support
forever, so the claim reads as a *permanent deadlock* — `conflict 0.67`, `belief 0` — no matter how
stale the objection is. An agent relying on it would keep flagging this claim for resolution
indefinitely, even though the only thing keeping it contested is a single ancient objection nobody
stood behind.

**With decay, the claim resolves on its own.** The objection's weight halves every 10 ticks, so as
its age `T` grows the fresh support wins: `conflict` collapses `0.67 → 0.06` and `belief` climbs
`0 → +0.50`. This is the **knowledge half-life** — recent evidence outweighs stale, unrefreshed
evidence, so memory is not held hostage by old objections. Reinforcement is just re-assertion: a
fresh edge restores full weight.

With the signed belief and the geometric/`ε`-prior tension, both axes move the right way: `belief`
swings from the deadlocked `0` toward the fresh support, and `conflict` falls as the stale surface
loses mass (faint evidence is both neutral *and* uncontested). Here it is the *freshness asymmetry*
that drives it — the support is fresh, the lone objection old — exactly the case where an
append-only store that cannot forget stays stuck.

## What this does and does not claim

- **Does:** show that an un-decayed belief over an append-only evidence log cannot forget — a stale
  minority objection deadlocks a claim forever — and that an `0.5^(age/half_life)` decay, computed
  lazily and deterministically from edge timestamps, lets fresh evidence resolve it without
  mutating or deleting history (the `Contradicts` edge is still in the log and the audit chain; only
  its *current weight* fades).
- **Does not:** prescribe a half-life. The right `half_life` is domain-dependent (code/logic: long;
  prices/logs: short) and is a caller parameter, not a fixed constant. Choosing or learning it per
  evidence class, and decaying on the *retrieval* path, are follow-ups.
