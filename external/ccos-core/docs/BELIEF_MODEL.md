# CCOS Core — Belief Model (Q-Page primitive)

A belief is **not a retrieved text fragment**. It is the state currently
accepted after folding evidence, authority, polarity and policy
(`src/memory.rs`, `QBelief`).

## Dual evidence surfaces

Every claim accumulates two typed surfaces:

- **affirmative surface** `S_A` — incoming `EdgeType::Supports` edges;
- **negative surface** `S_¬A` — incoming `EdgeType::Contradicts` edges.

Each edge weight is the **authority** of its source in `[0, 1]`.

## Derived axes (orthogonal by construction)

With prior strength `ε = BELIEF_EPS` (unit prior):

- `support` / `contradiction` — authority-weighted sums of the two surfaces;
- `belief = (s − c) / (s + c + ε)` ∈ `[−1, 1]` — signed direction and strength:
  `0` at no/balanced evidence, `→ +1` believed, `→ −1` refuted;
- `conflict = 2·√(s·c) / (s + c + ε)` ∈ `[0, 1]` — geometric evidence balance:
  `0` one-sided (consensus), `→ 1` strong *and* matched opposition (genuine
  cognitive tension).

A claim is **validated** only when `belief ≥ min_belief` **and**
`conflict ≤ max_conflict` — a strongly-believed but contested claim is
(correctly) not actionable.

## Why not a similarity score

Vector relatedness has no polarity: a refutation is lexically "about" its
claim. The Q-Page stores polarity as *structure* (two edge types), so
dissent is reported by construction (see `examples/contradiction_crux.rs`
and `tests/cognitive.rs::contradiction_detection`).
