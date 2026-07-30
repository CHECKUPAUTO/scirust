# CCOS Cognitive Persistence Benchmark — Metrics (§34)

Provisional name: **CCOS Cognitive Persistence Benchmark**. Not an accepted
scientific standard until publication and external evaluation (§32).

## Compared systems

| Label | System |
|---|---|
| Baseline A | LLM only |
| Baseline B | LLM + RAG |
| Baseline C | LLM + simple agent memory |
| System D | LLM + CCOS Core |
| System E | LLM + CCOS Enterprise (later) |

Research Lab is evaluated separately; its results never evidence Core
stability or certifiability (§6).

## Metric definitions

| Metric | Definition |
|---|---|
| `current_state_accuracy` | fraction of final-state answers matching the *current* (not historical) fact |
| `temporal_order_accuracy` | fraction of T1..Tn orderings correctly reconstructed |
| `stale_memory_reuse_rate` | fraction of answers citing invalidated facts as current |
| `contradiction_detection_rate` | detected explicit conflicts / planted conflicts |
| `contradiction_resolution_accuracy` | resolutions matching the authority-correct outcome |
| `unresolved_conflict_honesty` | unresolved cases reported as unresolved / truly unresolved cases |
| `provenance_accuracy` | answers citing the correct source / all sourced answers |
| `causal_link_accuracy` | decision→outcome links correctly recovered |
| `repeated_failure_rate` | repeated prior failures / similar later tasks |
| `strategy_adaptation_rate` | strategy changes after recorded failure / recorded failures |
| `cross_task_transfer_rate` | successful principle transfers A→B (no verbatim copying) |
| `long_horizon_completion_rate` | objectives completed with step tracking after interruptions |
| `replay_state_equivalence` | bit-equal final snapshots / replay runs (per configuration) |
| `audit_completeness` | state changes with a complete journal explanation / all changes |
| `human_correction_count` | manual corrections required per scenario |
| `token_usage`, `latency`, `model_cost`, `storage_cost`, `retrieval_count` | efficiency accounting per scenario |

## Publication rule (§34)

Never publish a score without: sample count, calculation method, model
identity, model version, test date, configuration, dataset, and an
uncertainty interval when relevant. Thresholds are defined **only after**
reproducible baselines exist (§50) — none are asserted here.
