# The crux: structure vs lexical RAG on real cross-file dependencies

> Reproduce: `cargo run --release --example rag_crux`

The paper's honest §9 result is that on real bug-fixes, causal selection *ties* a lexical
TF-IDF retriever — a fix's files share vocabulary. Now that the AST parser is the **default**
(so the causal graph is accurate), this tests the retrieval question directly on CCOS's *own*
code, LLM-free: **how much of the real dependency structure can a vector retriever actually
recover?**

## Method

- **Ground truth** = the real cross-file dependencies — the 134 file→file edges the AST resolved
  across CCOS's 41 source files.
- **Lexical RAG** = a TF-IDF embedder (the same one CCOS recall uses) over file contents; for each
  dependency `A → B`, rank every file by cosine-to-`A` and record `B`'s rank.

## Result

| metric (lexical RAG over 134 real dependencies) | value |
|---|---|
| recall@5 | **31 %** |
| recall@10 | **49 %** |
| MRR | 0.226 |
| mean cosine, *dependency* pair | 0.529 |
| mean cosine, *random* pair | 0.427 |

**Lexical RAG recovers only about half the real dependency structure** (recall@10 = 49 %). The
signal is real but weak: dependency pairs are only modestly more similar than random pairs
(0.529 vs 0.427, ≈ 2× better-than-random recall), because **import edges cross vocabulary
boundaries** — a file and the utility it depends on often share almost no identifiers. That gap
— the ~51 % of real dependencies a vector retriever misses in its top-10 — is exactly what a
structural layer fills.

## Honest scope (what this does and does not show)

- **It is not a tautological "structure beats RAG."** The structural side recovers dependencies
  *by construction* (the ground truth *is* the graph's edge set), so its ~100 % is definitional,
  not a contest. The measured, non-trivial quantity is **lexical's blind spot**: half the
  dependency graph is invisible to vocabulary similarity.
- **The AST's contribution is concrete and on this exact axis:** it raises the structural layer's
  ceiling from ~67 % (the heuristic missed a third of imports — `docs/MEASUREMENT_ast.md`) to
  ~100 % of real dependencies. Accurate parsing is what makes the structural recovery complete.
- **This is the retrieval (necessary) condition, LLM-free.** It does not run a model, so it does
  not measure end-task success (the sufficient condition — `ccos eval`, which needs an LLM key).
- **It complements, not refutes, §9.** §9's tie was on bug-fix *cohesion* (the set of files a fix
  touches — which share a feature's vocabulary). Raw import dependencies are a *different,*
  structurally-defined relation, and there the vocabulary signal is much weaker — so structure
  has the most to add precisely where RAG is weakest.

## Bottom line

This is the first measurement in the series where the structural approach has a clear, real-code
case: **a vector retriever alone misses half of a codebase's actual dependency structure**, and
the accurate AST is what lets CCOS recover the other half. It is the retrieval half of the
"CCOS > RAG" thesis, made concrete and honest; the LLM-in-the-loop half remains to be run.
