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

## Typed semantics (the labelled contract)

Documented in `scirust-som-symbolic`; the highlights:

- **type-aware Copy/move**: `i32`/`f64`/`bool`/raw pointers/`&T` copy on
  use (double use is legal, exactly like rustc); `String`/`Vec`/unknown
  owner types/`&mut T` move on use. Unannotated `let` bindings infer
  Copy-ness from their initializer; reading a Copy value under an
  outstanding `&mut` borrow is flagged (E0503-style);
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
| ownership accuracy (850 tokens) | **0.8729** |
| — majority-class baseline | 0.3306 |
| borrow accuracy | 0.9400 |
| fault-detection accuracy | 0.8859 |

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
- **Copy/move is type-aware** (fixed): scalar, pointer and `&T` uses
  copy; owner types and `&mut T` move; unannotated bindings infer from
  their initializer. Remaining over-approximation: `let x = f();` with no
  annotation defaults to move semantics (conservative) — full resolution
  needs the type-resolved `rustc`-driver path.
- **Straight-line code only.** `if`/`match`/loops/closures/macros are
  recorded as *unsupported* and skipped rather than lowered with invented
  branch-join semantics, so labels stay correct on what is analyzed.
- **Method receivers** are treated as shared borrows (reported as an
  approximation), since `&self` vs by-value `self` is not syntactic.

The deeper precision upgrade — NLL borrows, branch joins, and call-return
types — requires a future, fully tested HIR/MIR implementation. The former
analysis-only driver was removed; this `syn` frontend is the real-Rust entry
point that works today.

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
| symbolic | 12 | every fault kind incl. Copy semantics (legal double-use, E0503-style read under &mut, copy under & legal), drop labels, healing, tokenizer alignment, determinism |
| frontend | 6 | real-Rust lowering: move, borrows, methods, impl/scopes, unsupported, determinism, syntax errors |
| dataset | 4 | generator determinism, class coverage, vocab range |
| model | 3 | shapes, bit-determinism, seed sensitivity |
| trainer | 2 | loss decreases, bit-deterministic training |
| inference | 4 (+1 probe) | deterministic eval, beats baseline, oracle-checked + **real-Rust** prediction |
| cli (integration) | 7 | real `.rs` → oracle faults (use-after-move, borrow conflict, Copy legality, Copy inference, read-under-&mut), determinism |
| visualizer | 2 | fault rendering, clean-program rendering |
