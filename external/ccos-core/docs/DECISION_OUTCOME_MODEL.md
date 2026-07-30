# CCOS Core — Decision–Outcome Model

Learning from experience requires the full chain (§11.8):

```
initial state → decision → action → observation → outcome → evaluation → state update
```

## Representation

- **decision / action** — `EventType::AgentAction` with a custom payload
  (choice, inputs hash, state reference);
- **observation** — ingested evidence node + journal event;
- **outcome / evaluation** — subsequent `AgentAction` / custom payloads
  (result, latency, success flag);
- **failure** — first-class `EventType::FailureDetection` / `FailurePropagation`;
  nodes carry `failure_relevance` used by recall and eviction.

## Guarantees

- The chain is **ordered and tamper-evident** (hash-linked journal).
- Prior failures are **exactly retrievable** before a new decision
  (`tests/cognitive.rs::repeated_failure`) — the basis of
  `repeated_failure_rate` and `strategy_adaptation_rate` metrics (§33.4).
- State updates are **policy-driven folds**; the state after N events is a
  pure function of the journal + policy + version (replay equivalence).

## Boundary

"Learning" here means **state-level** learning: revised beliefs, invalidated
strategies, reused episodes. It does **not** modify LLM weights (§7.11) and
never triggers self-modification of the software itself (§4.1).
