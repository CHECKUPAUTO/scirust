# CCOS Core — Cognitive Replay

Replay answers: *how did the state evolve, and can that evolution be
reproduced exactly?* (§7.16–7.18)

## Levels

| Level | Mechanism | Guarantee |
|---|---|---|
| event replay | `replay_events(from, to)` | ordered retrieval of the journal |
| integrity replay | `verify_integrity()` | tamper evidence over the whole chain |
| state replay | re-fold of evidence under a fixed policy | identical inputs → identical derived state |
| snapshot replay | CCPS envelope → restore → re-derive | bit-for-bit snapshot equality |

## The invariant

```
identical events + identical policy + identical software version
    ⇒ identical final state (snapshot equality)
```

Tested by `tests/cognitive.rs::replay_equivalence`,
`tests/replay_equivalence_property.rs`, `tests/replay_vectors.rs` and the
canonical replay-vector target (imported from the security-hardening series).

## Documented relaxations

Replay equality holds for the deterministic build. Feature-gated relaxations
are **visible in the type system and docs**, never silent: the quarantined
neural embedder (`neural_embed`, weights/server/hardware-dependent) — see
DETERMINISM.md. Core ships **no** full-kernel SIMD relax and **no** evaluator
subprocess (those belong to Research Lab).

## Benchmark

§33.9 measures `replay_state_equivalence` across runs, machines and provider
rotations; divergence causes are reported, never hidden (§34).
