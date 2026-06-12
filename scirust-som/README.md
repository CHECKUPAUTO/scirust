# SOM — SciRust Ownership Model

An ownership-prediction pipeline, end to end and oracle-validated:
**real Rust source** (parsed with `syn`) → deterministic ownership
analysis (ground truth) → token encoding → Transformer encoder →
per-token predictions → evaluation against the oracle.

Everything below is implemented and tested in this directory; nothing is
claimed beyond what the tests exercise.

## Analyze a real Rust file

```bash
cargo run -p scirust-som-cli -- scirust-som/examples/use_after_move.rs
```

`som-analyze` parses the file with the real Rust grammar, lowers the
supported subset, runs the oracle, prints a per-token
ownership/borrow/fault table, and exits non-zero when it finds a fault —
e.g. on the bundled example it reports the `use of moved value` (E0382)
that rustc would. See [`examples/`](examples/).

## Pipeline

```
real Rust source (.rs)
        │
scirust-som-frontend   syn-based parser → lowers a Rust subset to the IR,
        │              reporting skipped/approximated constructs honestly
scirust-som-pcg        ownership IR (toy AST) + Place Capability Graph
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
scirust-som-inference  evaluation vs oracle, majority baseline, oracle-
        │              checked prediction on real Rust (predict_rust_source)
scirust-som-cli        `som-analyze <file.rs>` — oracle analysis of real code
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

## Scope and honest limits of the real-Rust frontend

The input is **genuine Rust** (real grammar, real `.rs` files, parsed on
stable by `syn`), but the analysis covers a deliberate subset and is
transparent about its boundaries — `som-analyze` prints what it skipped or
approximated for every file:

- **Lexical borrows, not NLL.** A borrow is held until its binding drops,
  moves or is reassigned (the documented oracle contract), which is
  *conservative* relative to rustc's non-lexical lifetimes: a borrow whose
  holder is never used again still counts as live. Borrow-conflict reports
  therefore match rustc when the borrow is genuinely used across the
  conflict, and may over-report otherwise.
- **Copy types are over-approximated.** The oracle models uniform move
  semantics — exact for `String`/`Vec`/`Box`, but `let b = a; let c = a;`
  on `i32` is legal in Rust yet flagged as use-after-move here.
  Distinguishing `Copy` needs type information (the `rustc`-driver path).
  Use non-`Copy` types for faithful results.
- **Straight-line code only.** `if`/`match`/loops/closures/macros are
  recorded as *unsupported* and skipped rather than lowered with invented
  branch-join semantics, so labels stay correct on what is analyzed.
- **Method receivers** are treated as shared borrows (reported as an
  approximation), since `&self` vs by-value `self` is not syntactic.

The deeper precision upgrade — `Copy`-awareness, NLL, and branch joins —
is the `scirust-rustc-driver` (HIR/MIR) path, kept outside the default
workspace; this `syn` frontend is the pragmatic real-Rust entry point that
works today.

Other limits unchanged from the model itself:

- **Sequence attention, not graph attention.** The backbone is a real
  Transformer encoder over the linearized token stream; PCG-edge-biased
  attention is future work, so we deliberately do not call it a "graph
  transformer".
- **No persistence.** Models train in-memory; SRT1-style serialization
  (as in `scirust-runtime`) is not yet hooked up.

## Test inventory

| crate | tests | what they pin |
|---|---|---|
| pcg | 3 | PCG edges for move / borrow / scope-drop |
| tokenizer | 4 | stream order, drops, vocab determinism + UNK overflow |
| symbolic | 9 | every fault kind, drop labels, healing, tokenizer alignment, determinism |
| frontend | 6 | real-Rust lowering: move, borrows, methods, impl/scopes, unsupported, determinism, syntax errors |
| dataset | 4 | generator determinism, class coverage, vocab range |
| model | 3 | shapes, bit-determinism, seed sensitivity |
| trainer | 2 | loss decreases, bit-deterministic training |
| inference | 4 (+1 probe) | deterministic eval, beats baseline, oracle-checked + **real-Rust** prediction |
| cli (integration) | 4 | real `.rs` → oracle faults (use-after-move, borrow conflict), determinism |
| visualizer | 2 | fault rendering, clean-program rendering |
