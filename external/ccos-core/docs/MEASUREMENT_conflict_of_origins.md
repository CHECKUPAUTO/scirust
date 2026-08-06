# Measurement — Conflict of Origins

*When two sources disagree about the same claim, how does authority weighting resolve it?*

Reproduce: `cargo run --release --example conflict_of_origins`

A flat or vector store has no notion of *who* said something, so when two documents disagree it can
only pick one, average them, or take the most recent. The Q-Page resolves the disagreement by
**source authority**: the more credible origin sets the *direction* of `belief`, `conflict` reports
how contested the claim remains, and [`QBelief::is_validated`](../src/memory.rs) decides whether the
claim is safe to act on. This is the payoff of the per-source **authority** weight (each assertion's
edge weight, clamped to `[0, 1]`) on top of the signed belief and geometric tension.

The assertions are produced through the `Extractor` pipeline (a deterministic `MockExtractor` here;
the `llm`-backed `LlmExtractor` distills the *same* `{claim, source, stance, authority}` shape from
raw text) and recorded via the normal `assert_support` / `assert_contradiction` path — so this is
exactly what an ingested pair of conflicting documents yields.

## Fixture

One claim, *"the payment API is thread-safe"*, asserted:
- **+** by source **A** = official docs, authority **0.90** (Supports),
- **−** by source **B** = an incident report, authority **β** (Contradicts) — swept from 0 to 1.

Validation gate: `belief ≥ 0.30` **and** `conflict ≤ 0.50`.

## Result (measured)

```
    β      belief    conflict   validated?
  0.00     +0.47      0.00      yes
  0.20     +0.33      0.40      yes
  0.30     +0.27      0.47      NO
  0.50     +0.17      0.56      NO
  0.70     +0.08      0.61      NO
  0.90     +0.00      0.64      NO
  1.00     -0.03      0.65      NO
```

## Reading

The credible source A holds the claim **positive** while the dissent is weak: a low-authority
incident report does **not** overturn the official docs, so at `β ≤ 0.2` the claim stays *validated*.
As B gains authority the two surfaces approach parity — `belief` slides toward `0` and, once B
out-weighs A (`β > 0.9`), past it: **the more credible origin now wins the direction**. Meanwhile
`conflict` climbs monotonically (`0 → 0.65`) because the disagreement is increasingly *real*, not
noise. The validation gate flips to **NO** at `β = 0.3` — precisely when B is credible enough that
the claim is no longer a confident fact and an agent *should* stop acting on it and seek resolution.

A flat or majority store cannot express any of this: with no model of *who* asserted what, it can
neither let a trustworthy source outweigh a dubious one nor report that a leaning claim is still
contested. Authority weighting + signed belief + geometric tension turn the **conflict of origins**
into a *computation* — a defensible, inspectable resolution — rather than a coin flip.

## What this does and does not claim

- **Does:** show that a per-source authority weight makes disagreeing origins resolve by credibility
  (not majority/recency), with `conflict` exposing the residual dispute and `is_validated` gating
  action. Demonstrated on a controlled two-source sweep (constructed to isolate the mechanism).
- **Does not:** specify *where* authority comes from (a trust policy — source reputation, signing,
  operator config — is a separate concern; the `LlmExtractor` merely proposes an authority that an
  operator can override). Nor does it benchmark end-to-end retrieval. The extraction step (text →
  assertions) is the only non-deterministic part and runs once at ingest; its output is recorded as
  replayable `Op::Assert` events, so a replay never re-calls the model (`replay == live`).
