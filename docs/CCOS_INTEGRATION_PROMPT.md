# Prompt — ccos × pure semantic retrieval integration (RAG challenger)

> To paste into a Claude Code session that has **both repos in scope**:
> `ccos` **and** `scirust`. Goal: wire SciRust's *pure* semantic retrieval
> platform (`scirust-retrieval`) onto the embeddings ccos already owns, then
> measure against the existing RAG. Pure Rust, zero FFI,
> deterministic.

---

```text
MISSION
Integrate SciRust's *pure* semantic retrieval platform (crate
`scirust-retrieval`) into the ccos project, to challenge ccos's existing RAG
on relevance — deterministically, pure-Rust, zero FFI. You do not rewrite
embeddings: you implement scirust-retrieval's `Encoder` trait ON TOP OF the
embeddings source ccos already owns, then you measure.

PHILOSOPHY (non-negotiable)
- 100% Rust, zero FFI, no proprietary GPU runtime.
- Bit-for-bit determinism: seeded RNG, f32 accumulation in fixed order.
- Honest oracle tests (hand-derived values), never adjusted to match a
  buggy output. Zero stubs, zero TODOs.
- Clean `cargo clippy --workspace --all-targets -- -D warnings`, clean `cargo fmt`,
  MSRV 1.89.
- Descriptive branch, draft PR. Never push elsewhere without permission.

PREREQUISITES
1. Confirm that the `scirust-retrieval` and `scirust-license` crates are
   reachable (same workspace, or relative path). Add them as `path`
   dependencies of the ccos crate that does retrieval:
     scirust-retrieval = { path = "../scirust/scirust-retrieval" }
     scirust-license   = { path = "../scirust/scirust-license" }   # if gating wanted
   (Adapt the actual path; do NOT duplicate retrieval code inside ccos.)
2. Locate in ccos: (a) the current embeddings source (the component that
   turns text into a dense vector), its dimension D, and whether it is
   deterministic; (b) the current RAG retriever and its evaluation set
   (queries + known relevant documents). You will use it as the comparison
   oracle.

TASK 1 — Adapt ccos's encoder (the bridge)
Create a type, e.g. `CcosEncoder`, that wraps ccos's embeddings source and
implements `scirust_retrieval::Encoder`:
    impl Encoder for CcosEncoder {
        fn embedding_dim(&self) -> usize { /* D of ccos */ }
        fn encode(&mut self, text: &str) -> Vec<f32> { /* ccos embedding of the text */ }
        // encode_batch has a default; override it if ccos can batch efficiently.
    }
Oracle test: the same text encodes to the same vector (determinism); the
returned dimension == D.

TASK 2 — Wire the pure semantic retrieval
- Build a `SemanticRetriever::new(CcosEncoder::…)`, index ccos's corpus
  (`index_text(id, text)`), query (`retrieve(query, k) -> Vec<Scored>`).
- Also add a `HybridRetriever::new(encoder, rrf_k)` (dense + BM25 fused
  by RRF) for the hybrid path.
- Verify the basic invariant: a query identical to a document finds it at
  rank 1 with a cosine of ~1.0.

TASK 3 — RAG CHALLENGER (the core)
On ccos's evaluation set, compute side by side, with
`scirust_retrieval::metrics`, for BOTH ccos's current RAG and the pure
retrieval (dense + hybrid):
    recall_at_k, precision_at_k, mean_reciprocal_rank, average_precision, ndcg_at_k
Produce a comparison table (k = 1, 5, 10) and a short quantified verdict:
where pure retrieval wins/loses, and the determinism/auditability gain
(same query → same result, bit for bit; no generative step that
hallucinates). Do not invent the numbers: run the measurement and report
the actual output.

TASK 4 — (optional) Continuous improvement + premium
- `ImprovementLoop::new(D, dim_out, seed, cfg)`: record the
  (query, relevant doc) pairs confirmed by ccos, `train_cycle()`, and show
  the Recall@k curve rising cycle after cycle.
- Premium gating: protect the commercial entry behind
  `RetrievalAccess::unlock(&entitlements)` (module `Module::Retrieval`). For
  the 1 USD/machine/month model, issue a node-locked license
  (`License::new(..).with_node_lock(machine_id)`) and verify it with
  `verify_license_on_node(.., machine_id)`. In dev, use `demo_vendor()` /
  `demo_root()`.

DELIVERABLES
- Descriptive branch (e.g. `ccos-pure-retrieval`), draft PR.
- The `CcosEncoder` bridge + the retriever wiring, under tests.
- The RAG-vs-pure benchmark with the REAL metrics in the PR description.
- All green: tests, clippy -D warnings, fmt.

REFERENCE API (scirust-retrieval, already on master)
- trait Encoder { fn embedding_dim(&self)->usize; fn encode(&mut self,&str)->Vec<f32>;
  fn encode_batch(&mut self,&[String])->Vec<Vec<f32>> }
- SemanticRetriever::new(E) ; .index_text(u64,&str)->Result<(),RetrievalError> ;
  .retrieve(&str,usize)->Vec<Scored> ; .len()/.is_empty()
- HybridRetriever::new(E, rrf_k:f32) ; .index_text ; .retrieve
- ImprovementLoop::new(dim_in,dim_out,seed,cfg:ContrastiveConfig) ; .record ;
  .train_cycle()->Vec<f32> ; .evaluate_recall_at_k(eval,corpus,k)
- RetrievalAccess::unlock(&Entitlements)->Result<Self,LicenseError> ;
  .semantic_retriever(E) / .hybrid_retriever(E,f32) / .improvement_loop(..)
- metrics::{recall_at_k, precision_at_k, reciprocal_rank, mean_reciprocal_rank,
  average_precision, ndcg_at_k}
- Scored { id:u64, score:f32 }
- scirust_license::{Module::Retrieval, verify_license, verify_license_on_node,
  License, node_fingerprint, demo_vendor, demo_root}

NOTE: Encoder::encode takes &mut self (allows an internal embeddings cache).
If the ccos source is immutable, simply ignore the mutability.
```
