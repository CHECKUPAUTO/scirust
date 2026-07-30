# CCOS Core — Model Provider Boundary

The LLM is a replaceable component. CCOS Core is model-provider independent
(§9, §30).

## ModelProvider (conceptual)

```rust
trait ModelProvider {
    fn generate(...);
    fn stream(...);
    fn call_tools(...);
    fn model_identity(...);
    fn usage(...);
}
```

Current implementations: `src/llm.rs` (Ollama HTTP + OpenAI-compatible +
Anthropic Messages paths, behind the `llm` feature), usage/latency journaled
as `LlmCallRequest`/`LlmCallResponse` events with explicit `model` identity.

## Rules

1. **No provider is hardcoded into Core logic.** Provider semantics are not
   merged into one approximate adapter when they differ (OpenAI vs Anthropic
   message models stay distinct).
2. **State survives switching.** Events, beliefs, resolutions, decisions,
   outcomes, snapshots and audit logs are provider-independent
   (`tests/cognitive.rs::model_switching`; benchmark §33.8).
3. **Effects are separated:** model effect ≠ retrieval effect ≠ CCOS effect ≠
   policy effect ≠ agent-framework effect. Benchmarks attribute improvements
   to the right layer (§30).
4. **Egress is policy-gated** (`src/egress.rs`, allowlist validation in
   `llm.rs`/`eval.rs`/`neural_embed.rs`): no silent network calls; the neural
   embedder is quarantined behind a feature flag because it cannot promise
   bit-exact replay (DETERMINISM.md).
5. **Live-provider tests are opt-in** (`tests/model_integration/`, feature
   `llm` + `#[ignore]`). Deterministic tests never touch the network.
