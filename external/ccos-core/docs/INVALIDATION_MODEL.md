# CCOS Core — Invalidation Model

Information can become: **obsolete · replaced · false · unreliable · expired ·
historically valid only** (§11.7).

## Rules

1. **Invalidated ≠ deleted.** The journal is append-only: what was asserted
   stays asserted *in history*. Invalidation changes the *current fold*,
   not the record.
2. **Invalidated information never counts as current truth.**
   - source removal (`remove_node`) drops the node *and* its incident edges,
     so its authority no longer weighs in any `QBelief`
     (`tests/cognitive.rs::invalidation`);
   - direct refutation (`Contradicts`) drives `belief` negative while
     preserving both surfaces.
3. **Auditability.** Provenance trust (`node_trust`) can demote an unreliable
   source without erasing it; every graph mutation is a journaled event.
4. **Measured, not assumed.** The benchmark tracks
   `stale_memory_reuse_rate` and invalidation delay (§33.3) — Core must
   demonstrate low reuse of invalidated memory, not merely claim it.
