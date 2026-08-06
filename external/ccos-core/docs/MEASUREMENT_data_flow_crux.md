# Data-flow semantic edges — what `DataFlow` edges recover that lexical retrieval misses

> Reproduce: `cargo run --release --example data_flow_crux`

The call crux (`docs/MEASUREMENT_call_crux.md`) showed a vector retriever misses the *transitive*
`caller → callee` structure once a chain runs deeper than the names a function spells out. Data flow
goes through a different channel entirely — **shared global state**. Two functions never call each
other and never import the same module, yet both read the same `static`/`const`; they are causally
coupled through that datum (change it, both behaviours move). This is **Slice 1** of the data-flow
graph (ROADMAP P1.3 "semantic edges"): extract `SCREAMING_SNAKE` value references from the `syn` AST
and resolve them to `static`/`const` definitions, adding `EdgeType::DataFlow` edges.

## How (precision-first, deterministic)

- **Extract** (`parser.rs::syn_ast`): a `syn::visit::Visit` over function/impl-method/trait-default
  bodies captures each **bare single-segment** `SCREAMING_SNAKE_CASE` value reference — the Rust
  convention for a `static`/`const` — **unless** the name is locally bound (a param, `let`, or
  fn-local `const`; the scope guard that closes the false "reads the global" edge). Qualified
  `m::CONST` is a later slice. Empty on the heuristic path — a build can only *omit* data edges.
- **Resolve** (`MemoryGraph::resolve_data_flow`, a whole-graph pass after call resolution):
  **global-unique, resolve-uniquely-or-skip** — a reference to `X` links only when exactly one
  resident `static`/`const` named `X` exists graph-wide, so an ambiguous name is dropped rather than
  mislinked. A `static`/`const` is the only valid target (the parser `mark_data_symbol`s it; the
  graph node stores `NodeType`, not `SymbolKind`). Self-edges dropped. Deterministic: indices over
  the **sorted** `data_symbols`/`pending_data_refs`, candidate edges sorted+deduped before insertion.

A `DataFlow` edge is **reader → data**: `source` is the reader function's symbol node, `target` is
the `static`/`const` it references. Confirmed from `g.edges()` on the fixture, e.g.
`sym:src/payment.rs:charge_invoice → sym:src/limits.rs:MAX_RETRIES`.

## The measurement — and the honest subtlety

A reader *names the const it reads* (`MAX_RETRIES` in the body tokenizes to `max`/`retries`, which
also appear in the const's own definition), so a **reader → data** link keeps *some* lexical signal.
The real value of the data-flow graph is the **co-reader** link: two functions that both read the
same global are causally related *through the shared datum*, yet they need share **no** domain
vocabulary — and across a realistic body that one shared concept is swamped by each function's own
disjoint words. On a fixture of three globally-unique consts/statics, six reader functions (some
*pairs* reading the same const but describing unrelated domains — billing, orbital control, baking;
irrigation, audio) and three decoys, ranking the target among all 12 symbols by per-symbol TF-IDF
cosine to the source:

| relation | lexical recall@1 | lexical MRR | data-flow graph |
|----------|:----------------:|:-----------:|:---------------:|
| **reader → data** | **33 %** | 0.62 | recovers (the edge) |
| **co-reader ↔ co-reader** | **25 %** | 0.49 | recovers (the closure) |

(6 reader→data edges, 4 co-reader pairs; raw run below.)

**Lexical similarity keeps only a weak grip on reader → data and all but loses the co-reader link.**
Even reader → data is rank-1 only a third of the time — the const's two subwords compete with the
boilerplate (`pub`/`fn`/`let`/`i64`) every symbol shares and with the tiny body of the data symbol
itself, so the reader's true target is usually *not* its nearest neighbour. Co-readers fare worse:
they share only the single const concept, swamped by disjoint domain vocabulary, so a true co-reader
typically ranks **below** an unrelated decoy (recall@1 25 %, MRR 0.49 — the lone @1 hit is a
coincidence, two readers that happen to share loop boilerplate `while`/`mut`, not shared meaning).
This is exactly the cross-vocabulary, causally-distant relationship the data-flow graph links by
construction: both readers point at the same data node, a 2-hop `reader → data ← reader` path.

As in `call_crux`/`rag_crux`, the data-flow graph reaching these is **by construction** — the ground
truth *is* the edge set the resolver produced — so the measured quantity is **lexical similarity's
shared-state blind spot** (co-reader recall@1 25 %, MRR 0.49), not a tautological "structure beats
RAG" contest. What the data-flow graph adds is the *traversal* a vector index cannot do: from one
reader, reach every other reader of the same global with zero shared vocabulary.

```text
# Data-flow crux — what DataFlow edges recover that lexical retrieval misses

fixture: 12 symbols, 6 resolved DataFlow (reader→data) edges, 4 co-reader pairs

LEXICAL TF-IDF (per-symbol) — rank of the target among all symbols, by cosine to source:
  READER -> DATA               lexical recall@1  33%   MRR 0.62   (n=6)
  CO-READER <-> CO-READER      lexical recall@1  25%   MRR 0.49   (n=4)

DATA-FLOW GRAPH — recovers both by construction (reader→data = the edge; co-readers
= the two readers sharing one data target, a 2-hop reader→data←reader path).
```

## Scope (Slice 1)

Bare `SCREAMING_SNAKE` references only; qualified `m::CONST` paths and non-conventional casings are
deferred — a precision-over-recall trade-off, stated so low recall on such code is not read as a bug.
A bare reference is taken as a global *unless* locally bound, the conservative guard that can only
drop a data-edge, never invent one. The resolver is whole-graph (re-runs each ingest, self-healing as
later files arrive) and feature-gated to the `syn` AST. The `DataFlow` edges are traversed type-blind
by recall/regions/propagation, so they enrich those for free while staying filterable later.
