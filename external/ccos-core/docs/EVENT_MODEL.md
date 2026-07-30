# CCOS Core — Event Model

The journal (`src/event_log.rs`) is an append-only, hash-chained sequence of
`TraceEvent`s.

## TraceEvent

| Field | Meaning |
|---|---|
| `id` | stable event identifier |
| `timestamp` | logical/sequence time of the run |
| `event_type` | `EventType` (LlmCall, LlmResponse, Parsing, GraphMutation, FailureDetection, GuardCheck, CycleStart/End, ReplayStart/End, Snapshot, AgentAction, RegionCreated/Activated/Merged/Evicted, ContextWindowGenerated) |
| `payload` | typed `EventPayload` (LLM request/response with model identity + usage, parsing, guard, custom key/value, …) |
| `sequence_number` | position in the chain (0-based) |
| `prev_hash` | hash of the previous event (`GENESIS_HASH` for the first) |
| `hash` | SHA-256 over `(prev_hash, sequence_number, event_type, payload)` — the non-deterministic `id`/`timestamp` are deliberately excluded so the chain is reproducible |

## Integrity

`verify_integrity()` replays the chain link-by-link and reports
`LogIntegrity { valid, verified_events, errors }`. Tampering with any recorded
event breaks the chain at that position.

## Replay

`replay_events(from, to)` returns events in order; `replay_deterministic`
re-folds state under a fixed policy. Identical inputs produce identical
derived state (see COGNITIVE_REPLAY.md, DETERMINISM.md).

## Schema evolution

Event payloads are versioned at the envelope level (CCPS: magic + version +
size + digest + payload — see PERSISTENCE_FORMATS.md). New event types are
additive only; removing or reinterpreting a type requires a schema-version
bump and a migration note in RELEASE_POLICY.md.
