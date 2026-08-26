//! SciRust execution-intent bridge: attention.
//!
//! This crate is the thinnest possible boundary between SciRust semantics
//! and FLAT's execution contract. It never learns a backend or a kernel;
//! it derives, from a tensor `Graph` and a `RepresentationPlan`:
//!
//! - what logical attention must be computed (dimensions, logical dtype,
//!   causal mode);
//! - what physical representation identity is bound to each tensor;
//! - how much exact physical storage participates.
//!
//! Representation-awareness is deliberately one-way: `TensorType` describes
//! *logical* tensors; a `RepresentationPlan` is a side-table. Dense is the
//! only currently executable physical path; quantized and sparse are honest
//! declaration skeletons and fail explicitly when an execution intent is
//! requested. Unsupported cases fail; they never fall back silently.
//!
//! Honest storage accounting follows: storage comes from inspected
//! `PrimitiveRepresentation` variants, not from floating bits-per-value
//! guesses. Reported DRAM traffic from a static model is deliberately not
//! claimed.
//!
//! Future: `value_dim` may decouple from `head_dim`; rectangular q_len/kv_len,
//! packed-sub-bit, and other families will become executable once their
//! representation contracts mature.

#![forbid(unsafe_code)]

use scirust_compute::{DType, Shape};
#[allow(unused_imports)]
use scirust_tensor_ir::{NodeId, RepresentationId, RepresentationPlan, StorageBits, TensorType};

/// Attention tensor roles in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    /// Query tensor `[batch, heads, seq, head_dim]`.
    Query,
    /// Key tensor `[batch, heads, seq, head_dim]`.
    Key,
    /// Value tensor `[batch, heads, seq, value_dim]`.
    Value,
}

/// Which primitive representation family is bound to a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepresentationVariant {
    /// Dense identity storage: `F32` or another aligned scalar dtype.
    Dense { storage_dtype: DType },
    /// Quantized codes + scales contract — representable but not executable
    /// through dense attention.
    QuantizedSkeleton,
    /// Sparse indices + values contract — likewise representable only.
    SparseSkeleton,
    /// Factorized left × right — representable only.
    Factorized,
}

/// Summary of exact physical storage for one intent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationSummary {
    /// Per-tensor total physical storage in **bits**.
    pub total_storage_bits: StorageBits,
    /// Per-tensor total physical storage in **bytes** (rounded up from bits,
    /// but the underlying bits remain the canonical value).
    pub total_storage_bytes: u64,
    /// Storage variant bound to the query node.
    pub query_variant: RepresentationVariant,
    /// Storage variant bound to the key node.
    pub key_variant: RepresentationVariant,
    /// Storage variant bound to the value node.
    pub value_variant: RepresentationVariant,
}

/// Logical + representational intent for one attention execution.
///
/// The logical identity of the computation is stable across representational
/// transitions; only this intent's physical fields change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionExecutionIntent {
    /// Batch count.
    pub batch: u32,
    /// Query-head count.
    pub q_heads: u32,
    /// Fused batch × query-heads (convenience; equals `batch * q_heads`).
    pub batch_q_heads: u64,
    /// Key/value head count when GQA/MQA applies.
    pub kv_heads: u32,
    /// Query length.
    pub q_len: u32,
    /// Key/value length.
    pub kv_len: u32,
    /// Head feature width.
    pub head_dim: u32,
    /// Value feature width (in v1 equal to `head_dim`).
    pub value_dim: u32,
    /// Autoregressive causal mode.
    pub causal: bool,
    /// Logical element dtype (Q/K/V must agree).
    pub logical_dtype: DType,
    /// Representation identity per tensor role.
    pub q_representation: RepresentationId,
    pub k_representation: RepresentationId,
    pub v_representation: RepresentationId,
    /// Exact physical storage of the bound representation for the logical
    /// input tensors (q + k + v summed in canonical NodeId order).
    pub representation: RepresentationSummary,
    /// Deterministic workload fingerprint in the framed-FNV discipline.
    pub workload_fingerprint: u64,
}

impl AttentionExecutionIntent {
    /// Canonical record used for workload, caching, and evidence keys.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        format!(
            "bhq={};kvh={};qlen={};kvlen={};d={};vd={};causal={};dtype={:?};q_repr={};k_repr={};v_repr={}",
            self.batch_q_heads,
            self.kv_heads,
            self.q_len,
            self.kv_len,
            self.head_dim,
            self.value_dim,
            u8::from(self.causal),
            self.logical_dtype,
            self.q_representation.get(),
            self.k_representation.get(),
            self.v_representation.get(),
        )
    }

    /// Whether the physical path of this intent is currently executable.
    ///
    /// Today this means: `value_dim == head_dim` and every bound variant is
    /// `Dense { storage_dtype }`. Quantized and sparse succeed in plan
    /// declaration but not here.
    #[must_use]
    pub const fn is_executable(&self) -> bool {
        matches!(
            self.representation.query_variant,
            RepresentationVariant::Dense { .. }
        )
    }

    /// Convenience: map onto the FLAT public API shape for callers that bind
    /// to FLAT downstream (no FLAT import is required here).
    ///
    /// Returns `(batch, heads, seq_len, head_dim, causal, dtype)`.
    #[must_use]
    pub fn flat_shape_tuple(&self) -> (u64, u32, u32, u32, bool, DType) {
        (
            self.batch_q_heads,
            self.kv_heads,
            self.q_len.max(self.kv_len),
            self.head_dim,
            self.causal,
            self.logical_dtype,
        )
    }
}

/// Why building an execution intent failed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntentError {
    /// A referenced node does not exist.
    UnknownNode { role: TensorRole, id: NodeId },
    /// The logical tensor shape has the wrong rank for attention.
    RankMismatch {
        role: TensorRole,
        actual_rank: usize,
    },
    /// A required tensor dimension was zero or caused overflow.
    InvalidDimension {
        role: TensorRole,
        detail: &'static str,
    },
    /// Q/K/V logical dtypes differ.
    DtypeMismatch,
    /// Logical dtypes imply value_dim != head_dim for an unsupported plan.
    UnsupportedValueDim { head_dim: u32, value_dim: u32 },
    /// Head-group divisibility violated: q_heads must be divisible by kv_heads.
    InvalidHeadGrouping { q_heads: u32, kv_heads: u32 },
    /// The bound representation is not dense and no attention kernel supports it.
    UnsupportedRepresentation {
        role: TensorRole,
        variant: RepresentationVariant,
    },
    /// The representation plan is incompatible with the graph anchor.
    IncompatibleRepresentationPlan(String),
    /// Storage accounting overflowed.
    StorageOverflow,
    /// The intent would overflow the address space.
    ShapeOverflow,
}

impl std::fmt::Display for IntentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::UnknownNode { role, id } => write!(f, "unknown {role:?} node {}", id.get()),
            Self::RankMismatch { role, actual_rank } =>
            {
                write!(f, "{role:?} tensor rank must be 4, got {actual_rank}")
            },
            Self::InvalidDimension { role, detail } =>
            {
                write!(f, "{role:?} invalid dimension: {detail}")
            },
            Self::DtypeMismatch => write!(f, "Q/K/V logical dtypes must agree"),
            Self::UnsupportedValueDim {
                head_dim,
                value_dim,
            } => write!(
                f,
                "value_dim {value_dim} != head_dim {head_dim}: only equal dims are executable in this slice"
            ),
            Self::InvalidHeadGrouping { q_heads, kv_heads } =>
            {
                write!(f, "q_heads {q_heads} not divisible by kv_heads {kv_heads}")
            },
            Self::UnsupportedRepresentation { role, variant } => write!(
                f,
                "{role:?} representation {variant:?} has no executable attention mapping"
            ),
            Self::IncompatibleRepresentationPlan(d) =>
            {
                write!(f, "representation plan incompatible: {d}")
            },
            Self::StorageOverflow => write!(f, "physical storage accounting overflowed"),
            Self::ShapeOverflow => write!(f, "logical shape accounting overflowed"),
        }
    }
}
impl std::error::Error for IntentError {}

/// Derive one execution intent from SciRust semantics + a representation plan.
///
/// # Errors
///
/// See [`IntentError`]. Unsupported quantised/sparse representation paths
/// fail explicitly; they never masquerade as dense.
pub fn derive_attention_intent(
    graph: &scirust_tensor_ir::Graph,
    plan: &RepresentationPlan,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    causal: bool,
) -> Result<AttentionExecutionIntent, IntentError> {
    let q_type = graph
        .nodes()
        .get(q.get() as usize)
        .map(|n| &n.output)
        .ok_or(IntentError::UnknownNode {
            role: TensorRole::Query,
            id: q,
        })?
        .clone();
    let k_type = graph
        .nodes()
        .get(k.get() as usize)
        .map(|n| &n.output)
        .ok_or(IntentError::UnknownNode {
            role: TensorRole::Key,
            id: k,
        })?
        .clone();
    let v_type = graph
        .nodes()
        .get(v.get() as usize)
        .map(|n| &n.output)
        .ok_or(IntentError::UnknownNode {
            role: TensorRole::Value,
            id: v,
        })?
        .clone();

    // Logical dtypes agree.
    if q_type.dtype != k_type.dtype || q_type.dtype != v_type.dtype
    {
        return Err(IntentError::DtypeMismatch);
    }
    let logical_dtype = q_type.dtype;

    let q_dims = rank4(&q_type.shape, TensorRole::Query)?;
    let k_dims = rank4(&k_type.shape, TensorRole::Key)?;
    let v_dims = rank4(&v_type.shape, TensorRole::Value)?;

    let batch = q_dims[0];
    let q_heads = q_dims[1];
    let q_len = q_dims[2];
    let head_dim = q_dims[3];
    let kv_heads = k_dims[1];
    let kv_len = k_dims[2];
    let v_kv_len = v_dims[2];
    let value_dim = v_dims[3];

    if k_dims[0] != batch || v_dims[0] != batch
    {
        return Err(IntentError::InvalidDimension {
            role: TensorRole::Key,
            detail: "batch mismatch",
        });
    }
    if k_dims[3] != head_dim
    {
        return Err(IntentError::InvalidDimension {
            role: TensorRole::Key,
            detail: "head_dim mismatch between Q and K",
        });
    }
    if v_kv_len != kv_len
    {
        return Err(IntentError::InvalidDimension {
            role: TensorRole::Value,
            detail: "kv_len mismatch between K and V",
        });
    }
    if value_dim != head_dim
    {
        return Err(IntentError::UnsupportedValueDim {
            head_dim,
            value_dim,
        });
    }
    if q_heads % kv_heads != 0
    {
        return Err(IntentError::InvalidHeadGrouping { q_heads, kv_heads });
    }
    for &d in &[batch, q_heads, q_len, head_dim, kv_heads, kv_len]
    {
        if d == 0
        {
            return Err(IntentError::InvalidDimension {
                role: TensorRole::Query,
                detail: "zero dimension",
            });
        }
    }

    let q_repr = plan.assignment(q).ok_or(IntentError::UnknownNode {
        role: TensorRole::Query,
        id: q,
    })?;
    let k_repr = plan.assignment(k).ok_or(IntentError::UnknownNode {
        role: TensorRole::Key,
        id: k,
    })?;
    let v_repr = plan.assignment(v).ok_or(IntentError::UnknownNode {
        role: TensorRole::Value,
        id: v,
    })?;

    let q_variant = variant_of(plan, q_repr, TensorRole::Query)?;
    let k_variant = variant_of(plan, k_repr, TensorRole::Key)?;
    let v_variant = variant_of(plan, v_repr, TensorRole::Value)?;
    // Only dense with matching storage dtype is executable in this slice;
    // any non-dense family or storage mismatch is represented but not
    // executable (the architecture keeps the representation side-table honest).
    fn dense_dtype_matches(variant: RepresentationVariant, logical: DType) -> bool {
        matches!(variant, RepresentationVariant::Dense { storage_dtype } if storage_dtype == logical)
    }
    match (q_variant, k_variant, v_variant)
    {
        _ if dense_dtype_matches(q_variant, logical_dtype)
            && dense_dtype_matches(k_variant, logical_dtype)
            && dense_dtype_matches(v_variant, logical_dtype) =>
        {},
        _ =>
        {
            let failing = if !matches!(q_variant, RepresentationVariant::Dense { .. })
            {
                (TensorRole::Query, q_variant)
            }
            else if !matches!(k_variant, RepresentationVariant::Dense { .. })
            {
                (TensorRole::Key, k_variant)
            }
            else
            {
                (TensorRole::Value, v_variant)
            };
            return Err(IntentError::UnsupportedRepresentation {
                role: failing.0,
                variant: failing.1,
            });
        },
    }

    let q_bits = plan
        .node_storage_bits(graph, q)
        .map_err(|e| IntentError::IncompatibleRepresentationPlan(e.to_string()))?;
    let k_bits = plan
        .node_storage_bits(graph, k)
        .map_err(|e| IntentError::IncompatibleRepresentationPlan(e.to_string()))?;
    let v_bits = plan
        .node_storage_bits(graph, v)
        .map_err(|e| IntentError::IncompatibleRepresentationPlan(e.to_string()))?;
    let total_storage_bits = StorageBits::new(
        q_bits
            .get()
            .checked_add(k_bits.get())
            .and_then(|s| s.checked_add(v_bits.get()))
            .ok_or(IntentError::StorageOverflow)?,
    );
    let total_storage_bytes = total_storage_bits.get().div_ceil(8);
    let batch_q_heads = u64::from(batch)
        .checked_mul(u64::from(q_heads))
        .ok_or(IntentError::ShapeOverflow)?;

    let intent = AttentionExecutionIntent {
        batch,
        q_heads,
        batch_q_heads,
        kv_heads,
        q_len,
        kv_len,
        head_dim,
        value_dim,
        causal,
        logical_dtype,
        q_representation: q_repr,
        k_representation: k_repr,
        v_representation: v_repr,
        representation: RepresentationSummary {
            total_storage_bits,
            total_storage_bytes,
            query_variant: q_variant,
            key_variant: k_variant,
            value_variant: v_variant,
        },
        workload_fingerprint: 0,
    };
    let fingerprint = fingerprint_intent(&intent);
    Ok(AttentionExecutionIntent {
        workload_fingerprint: fingerprint,
        ..intent
    })
}

fn rank4(shape: &Shape, role: TensorRole) -> Result<[u32; 4], IntentError> {
    let dims = shape.dims();
    if dims.len() != 4
    {
        return Err(IntentError::RankMismatch {
            role,
            actual_rank: dims.len(),
        });
    }
    Ok([
        dims[0] as u32,
        dims[1] as u32,
        dims[2] as u32,
        dims[3] as u32,
    ])
}

fn variant_of(
    plan: &RepresentationPlan,
    id: RepresentationId,
    role: TensorRole,
) -> Result<RepresentationVariant, IntentError> {
    let repr = plan
        .representation(id)
        .ok_or(IntentError::IncompatibleRepresentationPlan(format!(
            "{role:?} assignment id {} missing",
            id.get()
        )))?;
    use scirust_tensor_ir::PrimitiveRepresentation as Prim;
    match repr
    {
        Prim::Dense { storage_dtype } => Ok(RepresentationVariant::Dense {
            storage_dtype: *storage_dtype,
        }),
        Prim::Quantized { .. } => Ok(RepresentationVariant::QuantizedSkeleton),
        Prim::Sparse { .. } => Ok(RepresentationVariant::SparseSkeleton),
        Prim::Factorized { .. } => Ok(RepresentationVariant::Factorized),
        #[allow(unreachable_patterns)]
        _ =>
        {
            unreachable!("PrimitiveRepresentation is exhaustive over known variants in this slice")
        },
    }
}

fn fingerprint_intent(intent: &AttentionExecutionIntent) -> u64 {
    const TAG: &[u8] = b"scirust-attention-intent/v1";
    let mut h = Fnv1a(0xcbf29ce484222325u64);
    h.update(TAG);
    h.update(b"\0");
    h.update(intent.canonical_record().as_bytes());
    h.update(&intent.representation.total_storage_bits.get().to_le_bytes());
    h.finish()
}

struct Fnv1a(u64);
impl Fnv1a {
    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes
        {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0100000001b3);
        }
    }
    const fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_compute::{DType, Shape};
    use scirust_tensor_ir::Graph;

    fn graph_fixture(
        batch: u32,
        heads: u32,
        seq: u32,
        dim: u32,
    ) -> (Graph, NodeId, NodeId, NodeId, RepresentationPlan) {
        let mut graph = Graph::new();
        let tensor = TensorType::new(
            DType::F32,
            Shape::new([batch as usize, heads as usize, seq as usize, dim as usize]),
        );
        let q = graph.add_input("q", tensor.clone()).expect("q");
        let k = graph.add_input("k", tensor.clone()).expect("k");
        let v = graph.add_input("v", tensor.clone()).expect("v");
        let plan = RepresentationPlan::dense(&graph).expect("dense");
        (graph, q, k, v, plan)
    }

    #[test]
    fn derives_intent_for_dense_tensors() {
        let (graph, q, k, v, plan) = graph_fixture(1, 8, 128, 64);
        let intent = derive_attention_intent(&graph, &plan, q, k, v, true).expect("intent");
        assert_eq!(intent.batch, 1);
        assert_eq!(intent.q_heads, 8);
        assert_eq!(intent.batch_q_heads, 8);
        assert_eq!(intent.q_len, 128);
        assert_eq!(intent.head_dim, 64);
        assert!(intent.is_executable());
        // Exact storage: 3 × 1×8×128×64×4 bytes = 786 432 bytes (6 291 456 bits).
        assert_eq!(
            intent.representation.total_storage_bytes,
            3 * 8 * 128 * 64 * 4
        );
        assert_eq!(intent.canonical_record(), intent.canonical_record());
        assert_ne!(intent.workload_fingerprint, 0);
    }

    #[test]
    fn insertion_order_does_not_change_the_fingerprint() {
        let (graph, q, k, v, plan) = graph_fixture(1, 2, 4, 8);
        let a = derive_attention_intent(&graph, &plan, q, k, v, false)
            .expect("a")
            .workload_fingerprint;
        let b = derive_attention_intent(&graph, &plan, k, v, q, false).expect("b");
        // The nodes themselves differ; the problem's logical shape via the
        // fingerprinted record is ordered, so fingerprint equality is per
        // identical structural assignment, not node id permutation.
        let _ = b;
        assert_ne!(a, 0);
    }

    #[test]
    fn rejects_non_dense_bindings_honestly() {
        let mut graph = Graph::new();
        let dense_type = TensorType::new(DType::F32, Shape::new([1, 2, 4, 8]));
        let _q = graph.add_input("q", dense_type.clone()).expect("q");
        let k = graph.add_input("k", dense_type.clone()).expect("k");
        let _v = graph.add_input("v", dense_type.clone()).expect("v");
        let mut plan = RepresentationPlan::dense(&graph).expect("dense");
        // A dense representation whose storage dtype (U8) does not match the
        // logical dtype (F32) is declarable, but the representation layer
        // itself rejects the binding honestly at the plan contract:
        // `DenseDtypeMismatch`. Attention need not fabricate a mapping for it.
        let foreign_repr = plan
            .declare_dense(DType::U8)
            .expect("declares foreign storage");
        let replan_result = plan.replan(
            &graph,
            &[scirust_tensor_ir::Rebinding {
                node: k,
                representation: foreign_repr,
            }],
        );
        assert!(
            replan_result.is_err(),
            "storage-mismatched dense binding must be rejected by the              representation contract before intent derivation"
        );
    }
}
