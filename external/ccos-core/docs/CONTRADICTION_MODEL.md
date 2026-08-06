# CCOS Core — Contradiction Model

A contradiction is an **explicit, inspectable object** — never a silent LLM
merge (§11.5).

## Representation

- Two claims are incompatible when contradictory evidence is asserted against
  a shared claim (`EdgeType::Contradicts`) or when two claims assert opposed
  current-state values for one subject (temporal displacement, §33.1).
- `QBelief.conflict` measures tension: it is high **only** when support and
  contradiction are both strong and matched.
- Both surfaces are preserved with provenance:
  `evidence_of(claim, Supports)` / `evidence_of(claim, Contradicts)`.

## What Core guarantees

1. Detection is deterministic and offline (a fold over edges).
2. Both sources survive with their polarity, authority and timestamps.
3. The tension is *measurable* and *enumerable* (`qbelief`, `qbeliefs`,
   `claim_beliefs`, the `tensions` CLI command).
4. Resolution is a separate, traced step — see RESOLUTION_MODEL.md.

## What Core deliberately does not do

- It never averages incompatible claims into a blur.
- It never drops the minority source.
- It never lets an LLM "pick one" without a recorded resolution surface.
