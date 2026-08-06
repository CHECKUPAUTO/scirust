# Heuristic vs real-AST ingestion — how much "garbage in"?

> Reproduce (the AST is now the default; `--no-default-features` selects the heuristic):
> ```bash
> cargo run --release --no-default-features --example parse_accuracy > /tmp/heuristic.txt
> cargo run --release                        --example parse_accuracy > /tmp/ast.txt
> diff /tmp/heuristic.txt /tmp/ast.txt
> ```

CCOS's causal graph is only as good as the ingestion that builds it — *garbage in, garbage
out*. The original default parser was a zero-dependency **line-based heuristic**; a **real
Rust AST** (`syn`) was available behind a feature. This measurement quantified how wrong the
heuristic is and is **why the AST is now the default** (heuristic kept as the
`--no-default-features` fallback).

Method: parse CCOS's *own* `src/` tree (41 files) with each backend and emit a canonical,
file-scoped, sorted dump of every symbol / `use` / module. The AST is a real parser, so on
valid Rust it is **ground truth**; every diff line is a heuristic error.

## Result

| | symbols | uses (imports) | modules | total items |
|---|------:|------:|------:|------:|
| **AST (`syn`) — ground truth** | 1254 | 432 | 81 | **1767** |
| heuristic (default) | 1322 | 289 | 83 | — |
| heuristic **missed** (false negatives) | 77 | **282** | 0 | 359 |
| heuristic **hallucinated** (false positives) | 145 | 139 | 2 | 286 |

**The heuristic disagrees with the real AST on 645 of 1767 structural items — a 36.5%
error rate.** Two failure modes dominate:

1. **Imports: 66.9% recall (289 / 432).** A grouped `use std::collections::{BTreeMap,
   BTreeSet, BinaryHeap, HashMap}` is **one** line to the heuristic but **four** imports to
   the AST, so 282 real imports are invisible — and 139 of the 289 it *does* report are
   hallucinated (mis-split paths, `use` seen inside strings / macros / comments). Imports
   are the backbone of the cross-file dependency edges (`external_memory.rs` resolves
   intra-crate imports into file→file edges), so **a third of the causal structure that is
   supposed to beat lexical RAG is missing or wrong** under the heuristic.
2. **Symbols: 145 hallucinated, 77 missed.** The line parser promotes *local* `const`s /
   items inside function bodies and tests to top-level API symbols (e.g. `const CENTRALITY`,
   `const PREFIXES`), inflating the node set with structure that does not exist; `syn`
   scopes them out correctly.

## Why it matters (the RAG argument)

The paper's honest negative result (§9) is that causal selection *ties* a lexical TF-IDF
retriever on real bug-fix commits, because a fix's files share vocabulary. The lever to
**beat** lexical retrieval is the dependency edges it cannot see — and those edges are built
from imports and symbols. A parser that is **36.5% wrong** at the structural layer cannot
deliver that advantage: it both misses real edges (a third of imports) and pollutes the
graph with phantom nodes. Accurate AST ingestion is therefore not an optimisation; it is the
**precondition** for structure-aware retrieval to out-perform search.

This measurement does not by itself prove the AST improves end-to-end retrieval (that is the
next measurement — accurate edges → region/causal selection vs the lexical baseline). What it
proves is the *input* quality gap, and that the heuristic is not a faithful basis for the
causal graph on real code.

## Verdict

The real-AST parser already exists and is correct (`src/parser.rs::syn_ast`); it is merely
**optional**. Given a 36.5% structural error in the default path, the AST should become the
**default** ingestion, with the heuristic retained only as the graceful fallback for non-Rust
or unparseable input (where `syn::parse_file` returns `None`). The dependency-light core stays
available via `--no-default-features`.
