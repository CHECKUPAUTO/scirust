# Call-graph semantic edges — what `Calls` edges recover that lexical retrieval misses

> Reproduce: `cargo run --release --example call_crux`

The import crux (`docs/MEASUREMENT_rag_crux.md`) showed a vector retriever misses ~half of the
file→file **import** structure. Calls go finer — `caller → callee` at the *function* level —
and they're the link the agent actually follows when chasing a cause. This is **Slice 1** of the
call graph (ROADMAP P1.3 "semantic edges"): extract call-sites from the `syn` AST and resolve them
to definition symbols, adding `EdgeType::Calls` edges.

## How (precision-first, deterministic)

- **Extract** (`parser.rs::syn_ast`): a `syn::visit::Visit` over function/impl-method/trait-default
  bodies captures **bare single-segment** calls `foo()` (qualified `a::b::foo()` and methods
  `x.bar()` are Slices 2/3). Empty on the heuristic path — a build can only *omit* call edges.
- **Resolve** (`MemoryGraph::resolve_symbol_calls`, a whole-graph pass after import resolution): a
  strict ladder, **resolve-uniquely-or-skip** at each tier so a wrong edge is never invented —
  (A) import-scoped (`use …::foo` pins the defining module), (B) same-module, (C) global-unique.
  Self-edges dropped. Deterministic: indices over **sorted** node ids, candidate edges
  sorted+deduped — the same property `link_module_imports` and replay rely on.

## The measurement — and the honest subtlety

A call site *names its callee* (`foo()` contains the token `foo`), so **direct** calls are
lexically visible. The call graph's real value is **transitive** dependencies: a root that reaches
a deep callee it never names. On a fixture of two 3-deep chains (names share no vocabulary across
hops) + decoys, ranking the callee among all symbols by per-symbol TF-IDF cosine to the caller:

| relation | lexical recall@1 | lexical MRR | call graph |
|----------|:----------------:|:-----------:|:----------:|
| **direct calls** (1 hop) | **75 %** | 0.88 | recovers (the edge) |
| **transitive calls** (2 hop) | **0 %** | 0.32 | recovers (the closure) |

**Lexical similarity finds direct calls and collapses on transitive ones** — exactly the
causally-distant dependency the call graph reaches by traversal. This is the call-level analogue of
the import crux: structure recovers the cross-vocabulary links a vector retriever cannot see, and
the deeper the chain, the larger the gap (a vector retriever has no notion of "two hops away").

As in `rag_crux`, the call graph reaching these is **by construction** — the ground truth *is* the
edge set — so the measured quantity is **lexical similarity's transitive blind spot** (recall@1 0 %,
MRR 0.32), not a tautological "structure beats RAG" contest. What the call graph adds is the
*traversal* a vector index cannot do.

## Scope (Slice 1)

Bare free-function calls only; qualified paths (Slice 2: `use` longest-prefix + `self`/`Self`) and
methods (Slice 3: `x.bar()` receiver-type inference, unique-or-skip — now landed, see
`docs/MEASUREMENT_method_crux.md`) extend it — a precision-over-recall trade-off throughout, stated so
low recall on macro-heavy code is not read as a bug. The resolver is whole-graph
(re-runs each ingest, self-healing as later files arrive) and feature-gated to the syn AST. The
`Calls` edges are traversed type-blind by recall/regions/propagation, so they enrich those for free
while staying filterable later.
