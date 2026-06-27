//! # scirust-retrieval — pure semantic (dense) retrieval
//!
//! A deterministic, pure-Rust dense-retrieval engine, positioned as an
//! **auditable alternative to RAG**. Where Retrieval-Augmented Generation bolts a
//! stochastic LLM generator onto a retriever, *pure semantic retrieval* returns
//! the most relevant passages **directly**: every score is a reproducible inner
//! product, every ranking is explainable, and the same query yields the same
//! result, bit for bit. There is no generation step to hallucinate, and nothing
//! to audit but linear algebra.
//!
//! ## Pipeline
//! 1. An [`Encoder`] turns text into a dense vector. The default is
//!    [`scirust_core::embed::EmbeddingEngine`] — a from-scratch MiniLLM
//!    transformer — but any encoder (e.g. a contrastively fine-tuned one) can be
//!    plugged in.
//! 2. [`DenseIndex`] stores L2-normalised vectors and returns the **exact** top-k
//!    by cosine similarity — no approximation, fully deterministic.
//! 3. [`rerank::maxsim`] optionally refines the candidates with ColBERT-style
//!    late interaction (token-level MaxSim) for higher precision.
//! 4. [`metrics`] scores the ranking (Recall@k, MRR, MAP, nDCG@k) so the quality
//!    claim is a measured number.
//!
//! ## Licensing
//! Retrieval is a **premium add-on** (the "RAG-killer", sold in the *Perception*
//! and *Industrie 4.0* bundles). The commercial entry points hang off a
//! [`RetrievalAccess`] capability token, obtained by unlocking a verified
//! `scirust-license` entitlement that covers
//! [`Module::Retrieval`](scirust_license::Module::Retrieval) — see the
//! [`license`] module. The example below uses the ungated constructors directly,
//! which remain available as primitives.
//!
//! ```
//! use scirust_retrieval::{SemanticRetriever, Encoder};
//! use scirust_core::embed::EmbeddingEngine;
//!
//! let corpus = ["the cat sat on the mat", "rust is a systems language",
//!               "dense retrieval ranks by meaning"];
//! let mut r = SemanticRetriever::new(EmbeddingEngine::new(&corpus));
//! for (i, text) in corpus.iter().enumerate()
//! {
//!     r.index_text(i as u64, text).unwrap();
//! }
//! // A query identical to a document retrieves that document first.
//! let hits = r.retrieve("rust is a systems language", 3);
//! assert_eq!(hits[0].id, 1);
//! ```

pub mod ann;
pub mod contrastive;
pub mod feedback;
pub mod hybrid;
pub mod index;
pub mod license;
pub mod metrics;
pub mod rerank;
pub mod vector;

pub use ann::LshIndex;
pub use contrastive::{ContrastiveConfig, ProjectedEncoder, ProjectionHead};
pub use feedback::ImprovementLoop;
pub use hybrid::{Bm25Index, HybridRetriever, reciprocal_rank_fusion};
pub use index::DenseIndex;
pub use license::RetrievalAccess;

use std::fmt;

/// A document id paired with a similarity score (higher is more relevant).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    /// The document id.
    pub id: u64,
    /// The similarity score.
    pub score: f32,
}

/// Errors from index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalError {
    /// A vector's length did not match the index's dimension.
    DimMismatch {
        /// The dimension the index expects.
        expected: usize,
        /// The dimension that was supplied.
        got: usize,
    },
}

impl fmt::Display for RetrievalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            RetrievalError::DimMismatch { expected, got } =>
            {
                write!(
                    f,
                    "vector dimension {got} does not match index dimension {expected}"
                )
            },
        }
    }
}

impl std::error::Error for RetrievalError {}

/// Anything that turns text into a dense embedding vector.
///
/// Implement this for your own encoder (for example a contrastively fine-tuned
/// model) to drive [`SemanticRetriever`]. The default implementation wraps
/// [`scirust_core::embed::EmbeddingEngine`].
pub trait Encoder {
    /// The dimension of the vectors this encoder produces.
    fn embedding_dim(&self) -> usize;

    /// Encode one text into a dense vector.
    fn encode(&mut self, text: &str) -> Vec<f32>;

    /// Encode a batch of texts. Defaults to encoding each in turn.
    fn encode_batch(&mut self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

impl Encoder for scirust_core::embed::EmbeddingEngine {
    fn embedding_dim(&self) -> usize {
        self.dim()
    }

    fn encode(&mut self, text: &str) -> Vec<f32> {
        self.embed(text)
    }

    fn encode_batch(&mut self, texts: &[String]) -> Vec<Vec<f32>> {
        self.embed_batch(texts)
    }
}

/// End-to-end dense retriever: an [`Encoder`] feeding a [`DenseIndex`].
pub struct SemanticRetriever<E: Encoder> {
    encoder: E,
    index: DenseIndex,
}

impl<E: Encoder> SemanticRetriever<E> {
    /// Build a retriever over `encoder`; the index dimension is taken from it.
    pub fn new(encoder: E) -> Self {
        let dim = encoder.embedding_dim();
        Self {
            encoder,
            index: DenseIndex::new(dim),
        }
    }

    /// Encode `text` and add it to the index under `id`.
    pub fn index_text(&mut self, id: u64, text: &str) -> Result<(), RetrievalError> {
        let v = self.encoder.encode(text);
        self.index.add(id, &v)
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether nothing has been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Encode `query` and return the exact top-`k` documents by similarity.
    pub fn retrieve(&mut self, query: &str, k: usize) -> Vec<Scored> {
        let q = self.encoder.encode(query);
        self.index.search(&q, k)
    }

    /// Borrow the underlying index (for inspection or serialisation).
    pub fn index(&self) -> &DenseIndex {
        &self.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::embed::EmbeddingEngine;

    // A small deterministic encoder built on the bundled MiniLLM. Same vocab and
    // text always yield the same vector, so these assertions are reproducible.
    fn engine() -> EmbeddingEngine {
        EmbeddingEngine::new(&[
            "the cat sat on the mat",
            "rust is a systems language",
            "dense retrieval ranks by meaning",
            "a query finds its document",
        ])
    }

    #[test]
    fn a_query_identical_to_a_document_retrieves_it_first() {
        let corpus = [
            "the cat sat on the mat",
            "rust is a systems language",
            "dense retrieval ranks by meaning",
        ];
        let mut r = SemanticRetriever::new(engine());
        for (i, text) in corpus.iter().enumerate()
        {
            r.index_text(i as u64, text).unwrap();
        }
        assert_eq!(r.len(), 3);
        let hits = r.retrieve("rust is a systems language", 3);
        assert_eq!(hits.len(), 3);
        // Identical text encodes to the same (normalised) vector -> self-cosine 1.
        assert_eq!(
            hits[0].id, 1,
            "self-retrieval must rank the exact doc first"
        );
        assert!(
            (hits[0].score - 1.0).abs() < 1e-4,
            "self-similarity should be ~1.0, got {}",
            hits[0].score
        );
        // Scores are sorted descending.
        assert!(hits[0].score >= hits[1].score && hits[1].score >= hits[2].score);
    }

    #[test]
    fn encoding_is_deterministic() {
        let mut e = engine();
        let a = e.encode("dense retrieval ranks by meaning");
        let b = e.encode("dense retrieval ranks by meaning");
        assert_eq!(a, b, "the same text must encode identically");
        assert_eq!(a.len(), e.embedding_dim());
    }

    #[test]
    fn retriever_dimension_follows_the_encoder() {
        let r = SemanticRetriever::new(engine());
        assert_eq!(r.index().dim(), 128);
        assert!(r.is_empty());
    }
}
