# CCOS Core — Cognitive State Model

The cognitive state is the smallest set of structures from which every answer
can be re-derived (§11 minimum model):

| Concept | Representation in Core |
|---|---|
| Observation | ingested evidence node (`GraphNode`) + journal event |
| Event | `TraceEvent` — sequenced, typed, hash-linked (`EventLog`) |
| Source | evidence node identity (label, provenance trust) |
| Claim | claim node — a normalised statement with a stable `NodeId` |
| Belief | **derived** `QBelief` fold over typed evidence edges (never stored) |
| Contradiction | `EdgeType::Contradicts` surface + `QBelief.conflict` measure |
| Resolution | authority-weighted fold (`belief` sign) + inspectable evidence set |
| Decision / Action / Outcome | journaled `AgentAction` / custom payloads (§11.8 chain) |
| Policy | context policy, scoring weights, eviction policy (versioned in snapshots) |
| Snapshot | serialised `MemoryGraph` in a CCPS envelope |
| Replay | `EventLog::replay_events` / `replay_deterministic` |
| Invalidation | source removal / trust demotion → derived belief returns to neutral |

## Rules

1. **Beliefs are derived, not stored.** A belief is recomputed from the edge
   set on every query. Snapshots therefore cannot disagree with the evidence
   they contain.
2. **No silent merging.** Support and contradiction are distinct edge types;
   both are preserved and enumerable (`evidence_of`).
3. **No provenance hallucination.** A claim with no evidence is neutral
   (`support = 0`, `belief = 0`) — never backed by an invented source.
4. **State survives providers.** All structures are provider-independent;
   model identity is recorded only as event metadata.
