# The quarantined neural embedder (`neural-embed`)

> Run: `cargo run --release --features neural-embed --example neural_vs_lsa`
> (needs a local embedding server; without one the example prints setup steps and exits cleanly)

This lands the paper's future-work item 1 — *"an optional neural embedder behind a feature flag,
quarantined so the default build stays deterministic and dependency-free"* — as exactly that: a
quarantine, with the contract stated rather than implied.

## Why a quarantine, not a default

CCOS's identity claims are `replay == live` (bit-for-bit), zero default dependencies, and an
air-gappable posture. A neural embedder structurally cannot join that set: its vectors are a
function of model weights, server build, and hardware — re-run next month against a re-quantized
model and the "same" text embeds differently, so nothing downstream is replay-exact. The honest
resolution is not to pretend otherwise but to draw the boundary in the build system:

| | default build | `--features neural-embed` |
|---|---|---|
| compiled in | no (`cfg`-gated module) | yes |
| new crates | — | **none** (reuses the tree's `reqwest`, adds its `blocking` client) |
| network | none | **local endpoint only** (e.g. Ollama on `127.0.0.1`) |
| leaves the host | nothing | nothing, if the server is local (that is your call, not CCOS's) |
| `replay == live` | bit-for-bit | **not guaranteed** — the point of the flag |

## The design (`src/neural_embed.rs`)

`NeuralEncoder` implements the same [`retrieval::Encoder`] trait as the deterministic
`CcosEncoder`/`LsaEncoder`, so it drops into any `SemanticRetriever`/`HybridRetriever` unchanged.

- **Fail fast.** `try_new(endpoint, model)` probes once and returns an error if the endpoint is
  unreachable or answers with no usable vector — there is deliberately **no silent fallback** to a
  deterministic encoder, which would fake semantics and hide the degradation.
- **Degrade visibly.** A *transient* failure mid-run yields a zero vector (which ranks last under
  cosine) and increments `errors()`; a run that ends with `errors() > 0` is degraded and should be
  reported as such.
- **Speak both dialects.** The response parser accepts classic Ollama `{"embedding": […]}` and the
  batched `{"embeddings": [[…]]}` shape; it is a pure function with offline unit tests — the only
  part of the module that *can* be tested without a server, so it is.
- Dimension is learned from the probe (no hard-coded model table); a mid-run model swap cannot
  corrupt the index (vectors are resized to the probed dimension).

## What to measure with it

`examples/neural_vs_lsa.rs` runs the synonym crux — query and answer share **zero** vocabulary —
with three encoders side by side. The two deterministic rows reproduce bit-for-bit
(lexical 0 % / LSA 17 % Recall@1, MRR 0.185 / 0.458); the neural row exists to answer, *on your
machine*: how much of the transformer's semantic-recall advantage does the zero-dependency LSA
already capture, and what does closing the rest cost in replayability? For a standard-benchmark
view, `NeuralEncoder` plugs into the BEIR harness the same way (it is just an `Encoder`) — indexing
a few thousand documents through a local model takes minutes, not seconds; budget accordingly.

No measured neural numbers are committed here **on purpose**: this repository's measurement docs
report only numbers reproducible from the repo alone, and a neural row is a function of whichever
model build you run. Measure locally; trust the deterministic rows.

## Setup (Ollama example)

```bash
ollama pull nomic-embed-text
ollama serve &
cargo run --release --features neural-embed --example neural_vs_lsa
# env overrides: CCOS_EMBED_ENDPOINT (default http://127.0.0.1:11434)
#                CCOS_EMBED_MODEL    (default nomic-embed-text)
```
