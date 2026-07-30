# The sufficient condition, key-free — isolated models on a synthetic causal chain

> Reproduce the windows: `cargo run --release --example sufficient_condition`
> (writes `/tmp/window_rag.txt`, `/tmp/window_ccos.txt`; the grading hands each window to fresh
> isolated models — here, subagents.)

The **necessary** condition (is the causally-needed file *retrievable*?) is LLM-free and already
measured (`docs/MEASUREMENT_rag_crux.md`). The **sufficient** condition asks more: given the
window a strategy assembles, does a model produce the **correct answer**? The paper lists this as
*not yet run in full* — because it needs a model in the loop. This runs it **without an API key**.

## Why no key is needed (and why a *separate* model is)

The task is a *synthetic* arithmetic causal chain, so its answer cannot be known in advance — it
can only be **computed from the window**. A fresh model that sees ONLY the window is therefore a
clean, isolated judge. It must be *separate* from the agent that built the scenario: that agent
(and this session) already knows the answer, so it would contaminate the test. Fresh subagents,
given only the window, have no such leakage — which is exactly the isolation an external/local
model would otherwise provide.

## Setup

Six one-line files; the chain `result_value = (seed_constant 42 + 8) · 2 = 100` is split so the
distant cause (`seed_constant`) shares **no vocabulary** with the query `result_value`, while three
decoys (`result_value_legacy`, `compute_result_value_old`, `result_value_fallback`) share it and
are causally irrelevant. Two strategies select **3 of 6 files** under one budget; the queried file
`api.rs` is pinned in both (the agent is editing it), a fair lexical baseline:

- **RAG** = pure TF-IDF top-k → `{api.rs, legacy.rs, cache.rs}` — has `result_value()` but **not**
  `transform_step`/`seed_constant`. The value is not computable from this window.
- **CCOS** = causal region (BFS over import edges from `api.rs`) → `{api.rs, step.rs, seed.rs}` —
  the whole chain.

## Result

Each window handed to **3 fresh isolated subagents** (Sonnet), asked for the integer, told to
answer `UNKNOWN` rather than guess:

| window | correct (= 100) | answers |
|--------|:---------------:|---------|
| **RAG (lexical)** | **0 / 3** | all `UNKNOWN` — correctly noted `transform_step` is absent |
| **CCOS (causal region)** | **3 / 3** | all `100` — `(42+8)·2` |

Unanimous: the causal region's window lets the model **solve**; the lexical window does not. Notably
the RAG models were **honest** — they answered `UNKNOWN` rather than hallucinating a decoy value
(7/13/99) — so even a good-faith model fails on the lexical window for lack of the cause, not for
lack of care.

## Honest scope

This confirms the **mechanism** end-to-end, with a strong model, key-free: when a cause is
lexically dissimilar from its symptom, only the causal region admits the answer. It is **not**
evidence that CCOS beats RAG on real code — this is the *favorable synthetic regime*, and the paper
(§9) is explicit that it **does not occur on most real bug-fixes**, where a fix's files share
vocabulary so lexical similarity finds them too. What is new here is the **methodology**: synthetic
tasks + isolated subagents run the sufficient condition the paper left open, with **no API key and
no contamination** — and the next step, the genuinely open one, is the same protocol on a model
that is also isolated from *real* code (a local/Ollama model, since this session's agent is not).
