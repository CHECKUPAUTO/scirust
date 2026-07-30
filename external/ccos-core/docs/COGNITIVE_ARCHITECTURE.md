# CCOS Core — Cognitive Architecture

CCOS Core is a **cognitive state layer above language models** — not a RAG
variant. A LLM + RAG system improves access to information; CCOS maintains a
persistent, structured, temporal, contradiction-aware, revisable, traceable
and reusable **cognitive state** (working hypothesis, §6 — to be validated
experimentally, never claimed as proven).

## Layers

```
agents / tools / MCP clients (Hermes, OpenClaw, …)
        │  stable MCP namespace: ccos.*
┌───────▼────────────────────────────────────────────┐
│ CCOS Core (this repository)                        │
│                                                    │
│  ┌─────────────┐   ┌───────────────────────────┐   │
│  │ EventLog    │   │ MemoryGraph (Q-Pages)     │   │
│  │ hash-chained│──►│ claims, evidence surfaces │   │
│  │ journal     │   │ Supports / Contradicts    │   │
│  └─────────────┘   └───────────────────────────┘   │
│  ┌─────────────┐   ┌───────────────────────────┐   │
│  │ Snapshots   │   │ Context Region Engine     │   │
│  │ CCPS        │   │ temporal decay, eviction  │   │
│  │ envelopes   │   │                           │   │
│  └─────────────┘   └───────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │ Providers (never the source of truth):      │   │
│  │   ModelProvider (OpenAI/Anthropic/local)    │   │
│  │   RetrievalProvider (vector/kw/graph/db)    │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

## Source of truth

The **hash-chained event journal** (`src/event_log.rs`) is the source of
truth. The memory graph and all derived beliefs are **folds** over recorded
evidence — recomputed, never asserted. `replay == live` is the Core invariant:
identical events + identical policy + identical version produce identical
state (see DETERMINISM.md).

## What the LLM is — and is not

The LLM provides language interpretation, extraction, local reasoning and tool
selection. It is **not** the memory, **not** the state, **not** the policy.
Replacing the provider (GPT ↔ Claude ↔ local) must preserve events, beliefs,
decisions, outcomes, snapshots and audit logs (§9; tested in
`tests/cognitive.rs::model_switching`).

## Product boundary

Core contains **no** RSI, **no** Forge, **no** self-modification, **no**
generated-code execution, **no** process execution on untrusted input
(enforced by `scripts/check-no-research-components.sh`). Experimental
extensions live in CCOS Research Lab; governance/tenancy in CCOS Enterprise.
