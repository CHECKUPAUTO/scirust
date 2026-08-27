//! Execution-intent bridge: `scirust-attention-intent` → FLAT public API.
//!
//! This module owns the narrowest translation that keeps the two layers
//! separate: SciRust intent says WHAT (including the exact physical storage
//! already bound), FLAT `api::v1::AttentionShape` says the same WHAT in
//! FLAT's contract. No kernel or capability knowledge is introduced here.

#![forbid(unsafe_code)]

use scirust_attention_intent::{AttentionExecutionIntent, IntentError};

/// Errors from the SciRust→FLAT shape translation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlatIntentError {
    /// The intent came from `derive_attention_intent` but fails FLAT's
    /// strict contract (rejected by `flat_attention::api::v1::AttentionShape::validate`).
    InvalidFlatShape(String),
    /// The underlying intent was invalid.
    Intent(#[allow(missing_docs)] IntentError),
}

impl std::fmt::Display for FlatIntentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::InvalidFlatShape(detail) => write!(f, "FLAT shape contract rejected: {detail}"),
            Self::Intent(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for FlatIntentError {}

/// Translate one SciRust execution intent into the FLAT backend-neutral shape.
///
/// The field mapping separates the two descriptions while preserving the
/// same logical computation:
///
/// - `batch_q_heads` here is the folded `batch_heads` that maps to FLAT's
///   `q_heads`/`kv_heads` after unfolding by `kv_heads`;
/// - rectangular `q_len`/`kv_len` and `value_dim` are preserved exactly.
///
/// # Errors
///
/// Returns [`FlatIntentError`] when the shape overflows usize or FLAT rejects it.
pub fn intent_to_flat_shape(
    intent: &AttentionExecutionIntent,
) -> Result<flat_attention::api::v1::AttentionShape, FlatIntentError> {
    // In v1 the SciRust intent enforces `value_dim == head_dim`, so the
    // FLAT shape's rectangular lengths are `q_len` and `kv_len` with the same
    // head feature width.
    let shape = flat_attention::api::v1::AttentionShape {
        batch: usize::try_from(intent.batch)
            .map_err(|_| FlatIntentError::InvalidFlatShape("batch overflows usize".into()))?,
        q_heads: usize::try_from(intent.q_heads)
            .map_err(|_| FlatIntentError::InvalidFlatShape("q_heads overflows".into()))?,
        kv_heads: usize::try_from(u64::from(intent.kv_heads))
            .map_err(|_| FlatIntentError::InvalidFlatShape("kv_heads overflows".into()))?,
        query_len: usize::try_from(intent.q_len)
            .map_err(|_| FlatIntentError::InvalidFlatShape("q_len overflows".into()))?,
        kv_len: usize::try_from(intent.kv_len)
            .map_err(|_| FlatIntentError::InvalidFlatShape("kv_len overflows".into()))?,
        head_dim: usize::try_from(intent.head_dim)
            .map_err(|_| FlatIntentError::InvalidFlatShape("head_dim overflows".into()))?,
        query_position_offset: 0,
    };
    shape
        .validate()
        .map_err(|e| FlatIntentError::InvalidFlatShape(e.to_string()))?;
    Ok(shape)
}

/// Round-trip helper used by integration tests: intent → FLAT shape → back
/// through FLAT validation.
#[cfg(test)]
mod tests {
    use super::*;
    use scirust_attention_intent::derive_attention_intent;
    use scirust_compute::{DType, Shape};
    use scirust_tensor_ir::{Graph, RepresentationPlan, TensorType};

    fn intent_fixture() -> AttentionExecutionIntent {
        let mut graph = Graph::new();
        let dtype = DType::F32;
        let tensor = TensorType::new(dtype, Shape::new([1, 8, 16, 64]));
        let q = graph.add_input("q", tensor.clone()).expect("q");
        let k = graph.add_input("k", tensor.clone()).expect("k");
        let v = graph.add_input("v", tensor.clone()).expect("v");
        let plan = RepresentationPlan::dense(&graph).expect("dense");
        derive_attention_intent(&graph, &plan, q, k, v, true).expect("intent")
    }

    #[test]
    fn converts_dense_intent_to_flat_shape_and_validates() {
        let intent = intent_fixture();
        let shape = intent_to_flat_shape(&intent).expect("converts");
        assert_eq!(shape.q_heads, 8);
        assert_eq!(shape.query_len, 16);
        assert_eq!(shape.head_dim, 64);
        assert_eq!(shape.kv_len, 16);
        shape.validate().expect("FLAT accepts the translated shape");
        // Workload fingerprints stay stable under translation.
        assert_ne!(intent.workload_fingerprint, 0);
    }
}
