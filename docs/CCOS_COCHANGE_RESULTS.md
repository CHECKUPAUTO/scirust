# Results — CCOS co-change benchmark

Protocol: `docs/CCOS_COCHANGE_PROTOCOL.md`, frozen and committed **before**
any implementation and before having seen any number.

Corpus: `Memorithm/scirust` at `e042861`, 2 785 `.rs` files.
CCOS graph: 76 065 nodes, 135 012 edges.
Ground truth: 518 commits retained out of 2 246, i.e. 1 841 anchor/target cases.
Measured sample: **220 cases**, drawn by content-hash ordering over the whole
history.

## Result

| budget | Recall CCOS | Recall BM25 | gap | MRR CCOS | MRR BM25 | Precision CCOS | Precision BM25 |
|---|---|---|---|---|---|---|---|
| 1 024 | **37.9%** | 16.5% | +21.4 pt | 43.9% | 35.5% | 12.1% | 34.8% |
| 2 048 | **44.6%** | 16.8% | +27.8 pt | 44.2% | 35.6% | 10.1% | 31.9% |
| 8 192 | **69.2%** | 25.4% | +43.8 pt | 44.8% | 41.9% | 5.3% | 20.3% |

The post-hoc split excluding the vendored `external/` tree (n = 218) is
identical to within 0.4 point on every cell.

## Verdict against the frozen criterion

Protocol §6 required, written before measurement:

> CCOS's Recall@budget exceeds BM25's **at all three budgets**, and the gap
> at the 2 048 budget is at least **5 percentage points**.

CCOS leads at all three budgets; the gap at 2 048 is **+27.8 points**.
**The criterion is met.**

This is the first measurement establishing that CCOS does what it claims,
against ground truth it did not produce.

## What CCOS loses

Precision, clearly: 10.1% versus 31.9% at 2 048 tokens. CCOS rakes wider —
it fills the window with granular nodes from the whole causal region,
where BM25 returns few whole files and hits more accurately. At a fixed
budget, a real share of the context goes into files the agent did not
need. The protocol designated recall as the primary metric *before*
measurement, and that is the right choice for "did the agent have what it
needed" — but the waste is real and is paid for.

## Latency

| | before | after |
|---|---|---|
| CCOS, `around` query | 8 812 ms | **1 314 ms** |
| BM25, same corpus | 21 ms | 21 ms |

The cost came from `hop_distances`, whose BFS looked for neighbors by
traversing **all** the graph edges for each popped node — O(V·E). An
adjacency index cached per graph version brings it down to O(V+E): a 6.6×
factor, and **all quality metrics unchanged to a tenth of a point**, which a
test locks in by recomputing the distances the old way.

CCOS remains 63× slower than BM25. The quadratic term is gone; what
remains has not yet been attributed, and no promise is made before
profiling.

This measurement also illuminates the default 5 000-node resident cap: its
real function was not to save RAM but to keep the query from becoming
impractical. It remains that the configuration measured here — cap raised,
entire corpus resident — is not the one that ships.

## What this benchmark does not prove

Recalled from protocol §7, and as true after measurement as before:

- it measures **context selection**, not agent success;
- co-modification is a **proxy** for need;
- one repository, one language;
- the anchor is a file, not a natural-language intent;
- 220 cases out of 1 841, i.e. 12% of the available corpus.

## Two measurement errors, recorded

Protocol §8 requires reporting the discarded runs and why.

**Run 1 — CCOS at 0.0% everywhere.** Harness failure, not a produced
result. `CcosMemory::new()` caps at 5 000 resident nodes; the corpus
produces 76 065, and I had omitted the `ensure_resident` call that the CLI
facade makes before each recall. The anchor was demoted to COLD, the
window came back empty. Fixed with `ensure_resident` plus raising the cap,
so that both systems genuinely see the same corpus as the protocol
requires.

**Run 2 — +12 points at 2 048, on 25 cases.** Flattering and unreliable
result. `git log` outputs in reverse-chronological order: the first 25
cases were the sync commits of CCOS's own code, dense in its own graph
and co-changing because copied wholesale. BM25 dropping to 7.5% there
instead of 14.2% was enough to disqualify the draw. Fixed with a
deterministic sort on the content hash.

Both errors were mine, in opposite directions, and neither was visible in
the number alone. That is the very purpose of the frozen protocol: it
prevents choosing after the fact which of the two to publish.

## Reproduce

```bash
sh scripts/ccos/cochange_cases.sh > cases.tsv
cd external/ccos-core
CASES_LIMIT=220 cargo run --release --example cochange_eval -- ../.. ../../cases.tsv
```
