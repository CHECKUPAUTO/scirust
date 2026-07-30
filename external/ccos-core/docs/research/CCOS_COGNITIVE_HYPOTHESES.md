# CCOS — Cognitive Hypotheses (§35)

Each hypothesis is falsifiable and ends in exactly one classification:
**confirmed · partially confirmed · not confirmed · refuted**.
Evaluation criteria are versioned in this file BEFORE results are observed;
any later change requires a protocol-version bump with rationale.

Protocol version: **0.1 (2026-07-26)** — initial, pre-results.

| ID | Hypothesis | Primary metrics | Scenario(s) |
|---|---|---|---|
| H1 | CCOS reduces invalidated-information reuse compared with standard RAG | `stale_memory_reuse_rate`, invalidation delay | §33.3 |
| H2 | CCOS improves explicit contradiction detection and management | `contradiction_detection_rate`, `contradiction_resolution_accuracy`, `unresolved_conflict_honesty` | §33.2 |
| H3 | CCOS improves long-horizon task continuity | `long_horizon_completion_rate` | §33.6 |
| H4 | CCOS enables more complete and auditable replay than RAG memory | `replay_state_equivalence`, `audit_completeness` | §33.9 |
| H5 | CCOS reduces repetition of previously observed failures | `repeated_failure_rate`, `strategy_adaptation_rate` | §33.4 |
| H6 | CCOS allows model replacement without losing persistent cognitive state | state preservation, `replay_state_equivalence` across providers | §33.8 |
| H7 | CCOS improves structured reuse of experience across tasks | `cross_task_transfer_rate` | §33.5 |

## Current status

| ID | Status | Evidence |
|---|---|---|
| H1 | *unevaluated* | deterministic unit coverage exists (`tests/cognitive.rs::invalidation`); comparative baseline pending |
| H2 | *unevaluated* | deterministic unit coverage (`contradiction_detection`, `contradiction_resolution`); crux examples exist |
| H3 | *unevaluated* | — |
| H4 | *unevaluated* | deterministic unit coverage (`replay_equivalence`, replay vectors) |
| H5 | *unevaluated* | deterministic unit coverage (`repeated_failure`) |
| H6 | *unevaluated* | deterministic unit coverage (`model_switching`); live-provider runs opt-in |
| H7 | *unevaluated* | — |

No hypothesis may be marked confirmed on the basis of Research Lab results
(§6). Negative results must be preserved (NEGATIVE_RESULTS_POLICY).
