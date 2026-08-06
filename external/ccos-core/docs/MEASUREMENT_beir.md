# BEIR — CCOS's deterministic retrievers on standard IR benchmarks

> Reproduce: fetch a dataset, then `cargo run --release --example beir_eval -- <dir>`
>
> ```bash
> curl -o scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
> mkdir -p data/beir && unzip scifact.zip -d data/beir
> cargo run --release --example beir_eval          # defaults to data/beir/scifact
> ```

The retrieval cruxes score CCOS's retrievers on corpora CCOS itself defines — useful for isolating a
mechanism (synonymy, contradiction), but not comparable to anyone else's numbers. This measurement
closes that gap (the paper's future-work item 4): the same four deterministic systems, run on two
standard **BEIR** datasets in their native format, scored with the metrics the IR community reports.
The datasets are **not committed** (size + licensing + the air-gap ethos); the harness reads any
BEIR-format directory. Zero new dependencies — the JSONL/TSV loading is `serde_json` + hand-rolled
TSV, both already in the tree. Output is **bit-for-bit identical across runs** (timings go to
stderr); every number below is the real output of one such run.

## The headline — pure-Rust, zero-dep BM25 matches the published baseline

| dataset | docs | judged queries | published Anserini BM25 (BEIR paper, nDCG@10) | **CCOS BM25** | Δ |
|---|---:|---:|---:|---:|---:|
| SciFact | 5,183 | 300 | 0.665 | **0.662** | −0.003 |
| NFCorpus | 3,633 | 323 | 0.325 | **0.307** | −0.018 |

CCOS's BM25 (`k1 = 1.2`, `b = 0.75`) uses a plain lowercase-alphanumeric tokenizer — **no stemming,
no stopword list, no tuning** — and still lands within a few thousandths of the tuned Lucene/Anserini
baseline on SciFact and within 0.02 on NFCorpus. That is the claim this measurement earns: a
dependency-free, bit-for-bit-reproducible retriever that is *competitive with the standard lexical
baseline on the standard benchmark*, not a toy that only wins on home turf.

## Full tables

**SciFact** (5,183 docs, 300 judged queries, graded qrels):

| system | nDCG@10 | R@10 | R@100 | MRR@10 | MAP |
|---|---:|---:|---:|---:|---:|
| **BM25 (k1=1.2 b=0.75)** | **0.662** | **0.791** | **0.886** | **0.628** | **0.620** |
| TF-IDF dense (hashed, d=512) | 0.313 | 0.417 | 0.671 | 0.286 | 0.287 |
| LSA dense (rank 128) | 0.190 | 0.282 | 0.636 | 0.164 | 0.173 |
| hybrid BM25⊕LSA (RRF k=60) | 0.502 | 0.705 | 0.885 | 0.447 | 0.442 |

**NFCorpus** (3,633 docs, 323 judged queries, graded qrels):

| system | nDCG@10 | R@10 | R@100 | MRR@10 | MAP |
|---|---:|---:|---:|---:|---:|
| **BM25 (k1=1.2 b=0.75)** | **0.307** | **0.149** | **0.237** | **0.515** | **0.139** |
| TF-IDF dense (hashed, d=512) | 0.110 | 0.053 | 0.151 | 0.212 | 0.041 |
| LSA dense (rank 128) | 0.100 | 0.046 | 0.153 | 0.185 | 0.037 |
| hybrid BM25⊕LSA (RRF k=60) | 0.261 | 0.127 | 0.239 | 0.435 | 0.108 |

Cost, single-threaded release on SciFact: BM25 index 0.4 s, TF-IDF dense index 0.8 s, LSA fit+index
8.4 s (a fixed-order 512×512 Jacobi), retrieve+score for 300 queries × 4 systems 1.8 s.

## The honest reading

- **BM25 dominates here, and that is the expected result.** SciFact claims and NFCorpus queries share
  vocabulary with their relevant documents, so exact term matching with IDF weighting is the right
  inductive bias. This is the *mirror* of `semantic_retrieval_crux`, where query and answer share
  **zero** vocabulary and the same LSA encoder beats the same lexical retriever 17 % → 0 % Recall@1.
  Neither result generalises to the other: **the structure of the task decides**, which is exactly
  why CCOS ships both signals and a fusion.
- **Hashed dense TF-IDF pays for its hashing.** Squeezing a ~30 k-term scientific vocabulary into 512
  hashed dimensions collides terms that BM25 keeps distinct — the gap (0.662 vs 0.313 on SciFact) is
  the price of a fixed-width dense vector, not of density per se.
- **Fusion is not free.** RRF with a strong and a weak system (0.502 vs BM25's 0.662 on SciFact)
  *dilutes* the strong one at the top of the ranking, even as it matches R@100. Fusion earns its keep
  when the two systems fail on *different* queries (the synonym crux); when one dominates everywhere,
  use it alone.
- **The remaining ~0.02 vs Anserini** (clearest on NFCorpus) is tokenisation: Anserini stems and
  drops stopwords; CCOS's tokenizer is deliberately minimal and dependency-free. A Porter stemmer in
  pure Rust would be a ~200-line, zero-dep addition if that gap ever matters.

## Why this matters for the moat

Every number above is **deterministic**: rerun the harness and the output is byte-identical (the
example's stdout carries no timings for exactly this reason). A BEIR evaluation of a neural retriever
depends on GPU nondeterminism, batch order, and model-version drift; CCOS's evaluation is a pure
function of the dataset bytes. Combined with the crux results (synonymy, contradiction, self-tuning),
the retrieval story is now: *standard-benchmark-competitive lexical retrieval, semantic recall where
vocabulary diverges, polarity awareness no similarity score carries — all replayable bit for bit.*
