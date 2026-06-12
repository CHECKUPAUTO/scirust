# SOM — SciRust Ownership Model

A vertical slice of an ownership-prediction pipeline, end to end and
oracle-validated: toy programs → deterministic ownership analysis (ground
truth) → token encoding → Transformer encoder → per-token predictions →
evaluation against the oracle.

Everything below is implemented and tested in this directory; nothing is
claimed beyond what the tests exercise.

## Pipeline

```
scirust-som-pcg        toy AST + Place Capability Graph (PCG) builder
        │
scirust-som-symbolic   ORACLE: abstract interpreter — emits the token
        │              stream AND its ground-truth labels + diagnostics
        │              (use-after-move, borrow conflicts, escaping borrow…)
scirust-som-tokenizer  same linearization (pinned by a cross-crate test)
        │              + closed deterministic vocab (names → slots)
scirust-som-dataset    seeded program generator → oracle-labelled samples
        │
scirust-som-model      Embedding + PositionalEncoding + TransformerEncoder
        │              (real multi-head attention from scirust-core)
        │              + 3 per-token heads: ownership / borrow / fault
scirust-som-trainer    tape-per-sample training, Adam, CE×2 + MSE loss
        │
scirust-som-inference  evaluation vs oracle, majority baseline,
        │              oracle-checked program prediction
scirust-som-visualizer markdown rendering of analyses
```

## Toy-language semantics (the labelled contract)

Documented in `scirust-som-symbolic`; the highlights:

- every value has move semantics (any variable use is a move);
- `&x` / `&mut x` borrows obey "N shared XOR 1 mutable";
- borrows taken in `let r = &x` are held by `r` and released when `r`
  drops, moves or is reassigned;
- bindings drop in reverse declaration order at scope end; moved-out
  bindings do not drop (their `Drop` token is labelled `Moved`);
- assignment re-initializes a moved variable (Rust re-initialization);
- `return &local` is flagged as an escaping borrow.

Per-token labels: ownership ∈ {NA, Owned, Borrowed, Moved, Dropped},
borrow ∈ {NA, None, Shared, Mut}, fault ∈ {0, 1}.

## Measured results (reproducible)

Train on 200 generated programs (seed 42), evaluate on 50 held-out
programs (seed 9042), model d_model=32 / 2 layers / 2 heads, 8 epochs —
runs in under a second in release mode:

| metric | value |
|---|---|
| ownership accuracy (850 tokens) | **0.8365** |
| — majority-class baseline | 0.3141 |
| borrow accuracy | 0.9388 |
| fault-detection accuracy | 0.8682 |

Reproduce with:

```bash
cargo test -p scirust-som-inference --release -- --ignored --nocapture
```

Determinism is tested, not assumed: same seeds ⇒ bit-identical model
logits, bit-identical training losses, identical datasets
(`forward_is_bit_deterministic_across_fresh_models`,
`training_is_bit_deterministic`, `generation_is_deterministic`).

## What this is NOT yet

- **Not real Rust.** The input is the toy AST of `scirust-som-pcg`. The
  bridge from real Rust code is `scirust-rustc-driver` (HIR/MIR via
  `rustc_private`), kept outside the default workspace; wiring it to the
  oracle is the next milestone.
- **Sequence attention, not graph attention.** The backbone is a real
  Transformer encoder over the linearized token stream. Attention masked
  or biased by PCG edges is future work — until then we deliberately do
  not call this a "graph transformer".
- **No persistence.** Models train in-memory; SRT1-style serialization
  (as in `scirust-runtime`) is not yet hooked up.

## Test inventory

| crate | tests | what they pin |
|---|---|---|
| pcg | 3 | PCG edges for move / borrow / scope-drop |
| tokenizer | 4 | stream order, drops, vocab determinism + UNK overflow |
| symbolic | 9 | every fault kind, drop labels, healing, tokenizer alignment, determinism |
| dataset | 4 | generator determinism, class coverage, vocab range |
| model | 3 | shapes, bit-determinism, seed sensitivity |
| trainer | 2 | loss decreases, bit-deterministic training |
| inference | 3 (+1 probe) | deterministic eval, beats majority baseline, oracle-checked prediction |
| visualizer | 2 | fault rendering, clean-program rendering |
