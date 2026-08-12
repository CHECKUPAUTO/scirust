# Protocol — CCOS co-change benchmark

**Status: frozen.** This document is written and committed *before* any
implementation and *before* having seen any number. It exists so that the
protocol cannot be adjusted after the fact based on the result obtained.

Any subsequent modification of this file must be a separate commit,
made after the results are published, and must explicitly state what
changes and why.

---

## 1. The question

CCOS is sold as a **causal memory**: it claims to select better working
context than a search. The measurable question is therefore:

> Starting from a file one is working on, does CCOS place into the context
> window the other files that will actually need to be modified — better
> than a lexical search, at an equal token budget?

## 2. Why existing benchmarks do not answer

- `examples/pure_retrieval_vs_rag.rs` takes as ground truth "the dependency
  files that the AST resolved", i.e. **CCOS's own graph**.
  A benchmark whose reference is produced by the system being evaluated
  cannot fail. It measures internal consistency, not value.
- `examples/recall_eval.rs` runs on a **synthetic corpus**, built with its
  own ground truth.
- `examples/beir_eval.rs` measures document retrieval on scientific
  abstracts (SciFact). Useful for situating the retrievers, unrelated to
  code-context selection.

None has ground truth **external to CCOS**. That is the gap this benchmark
fills.

## 3. Ground truth — git history

A commit that modifies several `.rs` files is a direct observation that
"these files had to be touched together". This information is produced by
humans solving real tasks, months before CCOS existed. It cannot be
influenced by the system being evaluated.

**Case selection** (deterministic, no tuning):

- walk the full history of `Memorithm/scirust` (2 246 commits);
- keep the commits touching **between 2 and 8 `.rs` files** — at least two
  so there is something to predict, at most eight to rule out
  mass refactorings and automated renames, which represent no
  reasoning task;
- exclude merge commits (no content of their own);
- keep only files **present in the current tree**: a file deleted since
  then can be neither recalled nor searched, expecting it would penalize
  both systems identically but blur the reading;
- after this filter, keep only commits that retain **≥ 2 files**.

**Building a case**: for a retained commit with files `{f₁ … fₙ}`, we
produce `n` cases. Case `i` has **anchor** `fᵢ` and **targets** `{f₁ … fₙ} \ {fᵢ}`.

## 4. What is compared

Both systems receive **the same corpus** (the current tree), **the same
anchor**, and **the same token budget**.

- **CCOS** — `recall` strategy `around`, anchor `file:<path>`, on the
  `.ccos/workspace.ccos` workspace ingested from that same tree.
- **Control — lexical BM25.** The anchor file's content is used as the query,
  all other corpus files are ranked by BM25 and the window is filled up to
  the budget.

The control is **not** a straw man: BM25 is the baseline the IR
literature still uses today, and the implementation comes from
`ccos_core::retrieval`, hence from the same code, with the same tokenizer and
the same deterministic arithmetic. Comparing CCOS to a blind agent would
prove nothing.

## 5. Metrics

For each case, we look at which target files appear in the returned
window:

- **Recall@budget** — fraction of targets present in the window. Primary
  metric: it is literally "did the context contain what I needed".
- **MRR** — inverse of the rank of the first target file. Measures whether
  targets arrive early.
- **Precision@budget** — fraction of the window that was relevant. Measures
  context waste.

Reported at **three budgets**: 1 024, 2 048 and 8 192 tokens. An advantage
that exists at only one budget is a tuning artifact, not a result.

## 6. Success criterion — fixed now

CCOS is declared **better than search** if, and only if:

> CCOS's Recall@budget exceeds BM25's **at all three budgets**, and the gap
> at the 2 048 budget is at least **5 percentage points**.

Any other result is a failure of the benchmark for CCOS, and will be
reported as such. Explicitly failures:

- CCOS wins at one or two budgets out of three;
- CCOS wins everywhere but by less than 5 points at the 2 048 budget;
- CCOS loses.

The 5-point threshold is set before measurement. It corresponds to what
would be visible on a real task; a one-point gap would not justify selling
an architecture.

## 7. What this benchmark does not prove

To be stated with the results, without waiting to be asked:

- It measures **context selection**, not agent success. Better context
  should help; this benchmark does not demonstrate it.
- Co-modification is a **proxy** for need. Two files changed together have
  sometimes been changed for unrelated reasons (version bump,
  reformatting).
- One repository, one language. Nothing here transfers to a Python
  repository or a ten-million-line codebase without new measurement.
- The anchor is a file, not a natural-language intent. The real scenario of
  an agent often starts from a task description.

## 8. Execution

The benchmark is run **exactly once** after implementation, and its result
is reported whatever it is. If it must be re-run (harness bug, poorly
built corpus), the reason is recorded in the report, along with the result
of the first execution.
