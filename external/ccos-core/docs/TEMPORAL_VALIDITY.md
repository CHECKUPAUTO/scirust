# CCOS Core — Temporal Validity

CCOS distinguishes **current** facts from **historical** facts (§7.2).

## Mechanisms

1. **Temporal displacement** — a new current-state claim contradicts the old
   one (T1 PostgreSQL → T3 FoundationDB in `tests/cognitive.rs::temporal_update`).
   The old claim remains addressable: it is *historically valid*, not current.
2. **Decay** — `qbelief_decayed(claim, half_life)` weighs evidence by age:
   knowledge half-life is a policy, not an accident (`examples/decay_crux.rs`).
3. **Recency fields** — every node carries `created_at`, `last_accessed`,
   `recency`; the journal carries `sequence_number` ordering.
4. **Eviction policies** — context regions age out; eviction is journaled
   (`RegionEvicted` events), so state ageing is auditable.

## Query contract

Answering "what is true now" must consult the current fold, not the historical
record: invalidated/displaced claims keep negative `belief` and therefore
never surface as current truth, while remaining fully inspectable as history.

## Scenario coverage

The benchmark scenario §33.1 (T1–T4 migration sequence) is implemented as a
deterministic test; the metric is `current_state_accuracy` +
`temporal_order_accuracy` + `stale_memory_reuse_rate` (docs/benchmarks/METRICS.md).
