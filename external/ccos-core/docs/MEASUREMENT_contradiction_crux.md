# Measurement — Contradiction crux

*What does a typed `Contradicts` edge recover that a similarity score cannot represent?*

Reproduce: `cargo run --release --example contradiction_crux`

This is the **Q-Page** analogue of the import/call/data-flow crux measurements. Those showed that
a vector retriever collapses on *structural* relationships it cannot name (transitive calls,
co-readers of a shared global). This one isolates a different blind spot: **polarity**. A vector
retriever ranks by *relatedness*, and relatedness has no notion of *for* vs *against* — a statement
that **refutes** a claim and one that **confirms** it are both "about" the claim, so similarity
cannot tell them apart.

## Fixture

A single claim, four confirmations (each lexically echoing the claim), **one** refutation (the
minority dissent — it shares the claim's subject vocabulary but opposes it), and three unrelated
decoys. The `Supports` / `Contradicts` edges are exactly what an agent records through
`CcosMemory::assert_support` / `assert_contradiction`; the example builds the graph directly so the
lexical baseline has text to embed.

- **claim:** *"the scheduler admits every queued job within the deadline"*
- **refutation:** *"under burst load the scheduler starved a queued job and missed its deadline"*

## Result (measured)

```
LEXICAL TF-IDF — pool ranked by cosine to the claim (similarity is polarity-blind):
  rank  cosine  polarity     text
     1   0.77  support      the scheduler admits every queued job within the deadline
     2   0.51  support      every queued job was admitted within the deadline during t
     3   0.39  support      queued jobs are admitted within the deadline by the schedu
     4   0.31  CONTRADICT   under burst load the scheduler starved a queued job and mi
     5   0.26  support      the scheduler met the deadline for all admitted jobs in th
     6   0.22  decoy        the cache evicts the least recently used entry when full
     7   0.20  decoy        the billing service reconciles outstanding invoices nightl
     8   0.06  decoy        the parser tokenizes identifiers in a single linear pass

The contradiction ranks #4 (cosine 0.31); the support cosines span [0.26, 0.77].
  → its similarity sits INSIDE the support band, so NO cosine threshold separates support
    from refutation: similarity cannot label polarity.
  → top-5 by similarity includes the refutation — and even when present it carries no flag
    that it is a refutation.

Q-PAGE:
  belief 0.50   conflict 0.67   (support 4, contradiction 1)
  → belief is net-positive (support outweighs the lone dissent) yet conflict 0.67: contested.
```

## Reading

The decisive number is not a recall@k gap — a similarity retriever **does** retrieve the
refutation (it ranks #4; it *is* related). The decisive number is that the refutation's cosine
(**0.31**) falls **inside the band of the supporters' cosines (`[0.26, 0.77]`)** — it is literally
interleaved among them (rank 4 of the 5 true evidence items, between a supporter at 0.39 and one at
0.26). So **no cosine threshold exists that puts the supports on one side and the refutation on the
other**: similarity has no axis on which polarity lives. A confirm-only retriever therefore returns
the dissent *without ever marking it as opposition*, and the lone refutation competes for top-k
slots against a crowd of confirmations with no signal of its outsized significance.

The Q-Page stores polarity as **structure**: support and contradiction are distinct edge types, so
the refutation is surfaced *as* a refutation by construction, and `qbelief` turns the two surfaces
into numbers — here **belief +0.50** (net support, signed in `[−1, 1]`) with **conflict 0.67**
(genuinely contested). That high `conflict` on a net-positive claim is precisely the state a
similarity index cannot represent: *leaning true and yet contested*, the claim a reasoning agent
should pause on rather than accept.

## What this does and does not claim

- **Does:** show that a similarity score is a strictly weaker representation than a typed
  support/contradiction edge — it cannot express polarity, so it cannot flag a contested claim, no
  matter how the threshold is tuned. This is a statement about *representation*, demonstrated on a
  controlled fixture (as with the other crux measurements, the fixture is constructed to isolate the
  phenomenon, not sampled from a corpus).
- **Does not:** benchmark an end-to-end RAG pipeline, nor claim the graph "retrieves better" on
  generic relevance. The advantage is narrow and structural: CCOS can *hold and surface contested
  knowledge*; a vector index cannot. Whether a contradiction is **detected** in the first place is a
  separate concern — in this slice contradictions are **explicit assertions** (an agent/tool marks
  them), recorded as replayable `Op::Assert` events; deriving them automatically (rules / NLI) is
  deferred to a later slice.
