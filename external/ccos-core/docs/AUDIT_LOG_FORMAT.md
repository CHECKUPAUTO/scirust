# CCOS Core — Audit Log Format

The audit trail is the hash-chained journal itself (EVENT_MODEL.md) plus any
persisted CCPS snapshot envelopes (PERSISTENCE_FORMATS.md).

## Canonical event line (conceptual)

```
seq=<u64> prev=<sha256:hex> type=<EventType> payload=<canonical-json> hash=<sha256:hex>
```

- `hash = SHA-256(prev_hash ‖ sequence_number ‖ event_type ‖ canonical(payload))`
- canonical payload: serde JSON with struct-field ordering (serde derives
  field order from declaration) — the same construction the journal verifies
  in `verify_integrity()`.
- `id`/`timestamp` are excluded from the chain hash by design (they are
  non-deterministic across runs); the chain therefore verifies identically on
  every replay of the same logical run.

## Querying

- `replay_events(from, to)` — ordered slice;
- `verify_integrity()` — full-chain validation with per-event error detail;
- MCP: `audit.query`-class read tools under the `ccos.*` namespace;
- CLI: `ccos audit` / `ccos tensions` / `ccos trace`.

## Retention & rotation

Journals are append-only. Rotation creates a new segment whose first event
links to the last hash of the previous segment (prev_hash continuity), so a
rotated trail remains verifiable end-to-end. Deletion/retention **policies**
(enterprise-grade) are a CCOS Enterprise feature and are out of Core scope.
