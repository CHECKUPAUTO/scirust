//! # scirust-retrieval — pure semantic (dense) retrieval
//!
//! A deterministic, pure-Rust dense-retrieval engine, positioned as an
//! **auditable alternative to RAG**.

// Pure retrieval core — no scirust-core / scirust-graph dependency.
pub mod forgetting;
pub mod hybrid;
pub mod index;
pub mod ivf;
pub mod license;
pub mod metrics;
pub mod rag;
pub mod rerank;
pub mod vector;
pub mod vsa;

// Fine-grained learned/graph capabilities. The `learned` Cargo feature remains a
// backward-compatible umbrella, but external consumers can now select only the
// capability they need.
#[cfg(feature = "ann")]
pub mod ann;
#[cfg(feature = "causal-rerank")]
pub mod causal_rerank;
#[cfg(feature = "contrastive")]
pub mod contrastive;
#[cfg(feature = "feedback")]
pub mod feedback;

pub use forgetting::{BoundedSemanticMemory, DecaySchedule, DocMeta};
pub use hybrid::{Bm25Index, HybridRetriever, reciprocal_rank_fusion};
pub use index::DenseIndex;
pub use ivf::IvfIndex;
pub use license::RetrievalAccess;
pub use vsa::{Cleaned, CleanupMemory};

#[cfg(feature = "fusion")]
pub use vsa::{FusionStrategy, fuse_observations};

#[cfg(feature = "ann")]
pub use ann::LshIndex;
#[cfg(feature = "contrastive")]
pub use contrastive::{ContrastiveConfig, ProjectedEncoder, ProjectionHead};
#[cfg(feature = "feedback")]
pub use feedback::ImprovementLoop;

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
    /// An operation needed at least one element and got none.
    EmptyInput,
    /// An input contained NaN or ±∞ where finite values are required.
    NonFiniteInput,
    /// An operation required a nonzero vector.
    ZeroVector,
    /// A configuration parameter was out of its valid range.
    InvalidParameter {
        /// What was wrong with the parameter.
        reason: &'static str,
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
            RetrievalError::EmptyInput =>
            {
                write!(f, "operation requires at least one element")
            },
            RetrievalError::NonFiniteInput =>
            {
                write!(f, "input contains NaN or infinite values")
            },
            RetrievalError::ZeroVector =>
            {
                write!(f, "operation requires a nonzero vector")
            },
            RetrievalError::InvalidParameter { reason } =>
            {
                write!(f, "invalid parameter: {reason}")
            },
        }
    }
}

impl std::error::Error for RetrievalError {}

/// Anything that turns text into a dense embedding vector.
pub trait Encoder {
    /// The dimension of the vectors this encoder produces.
    fn embedding_dim(&self) -> usize;

    /// Encode one text into a dense vector.
    fn encode(&mut self, text: &str) -> Vec<f32>;

    /// Encode a batch of texts.
    fn encode_batch(&mut self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

/// Adapter for scirust-core's MiniLLM transformer embedding engine.
///
/// Kept behind its own feature so ANN/projection consumers are not forced to
/// compile a concrete encoder they do not use.
#[cfg(feature = "scirust-encoder")]
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
    /// Build a retriever over `encoder`.
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

    /// Borrow the underlying index.
    pub fn index(&self) -> &DenseIndex {
        &self.index
    }
}

#[cfg(all(test, feature = "scirust-encoder"))]
mod tests {
    use super::*;
    use scirust_core::embed::EmbeddingEngine;

    #[test]
    fn scirust_encoder_drives_the_semantic_retriever() {
        let engine = EmbeddingEngine::new(&["hello world", "rust is fast"]);
        let mut retriever = SemanticRetriever::new(engine);

        retriever.index_text(1, "hello world").unwrap();
        retriever.index_text(2, "rust is fast").unwrap();

        let hits = retriever.retrieve("hello", 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
    }
}
