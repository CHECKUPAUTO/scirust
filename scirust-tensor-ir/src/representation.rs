//! Backend-neutral representation planning for canonical tensor values.
//!
//! A [`RepresentationPlan`] is deliberately separate from [`crate::TensorType`].
//! `TensorType` describes the logical value (`dtype`, `shape`); this module
//! describes how that value may be represented for storage or execution.
//!
//! This first phase supports only the identity physical representation:
//! dense storage in the tensor's declared logical dtype. Quantization,
//! factorization, composite representations, cost models and backend-specific
//! materialization are intentionally out of scope.

use alloc::vec::Vec;
use core::fmt;

use scirust_compute::DType;

use crate::{Graph, NodeId};

/// Stable identifier of one representation declared in a
/// [`RepresentationPlan`].
///
/// Identifiers are assigned deterministically in first-use order and carry no
/// backend, address, allocation or execution meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepresentationId(u32);

impl RepresentationId {
    /// Construct an identifier from its canonical integer value.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the canonical integer value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Exact number of physical storage bits required by a representation.
///
/// This is an integer count, not an entropy estimate or an average
/// bits-per-value rate. Representation metadata that physically occupies
/// storage must eventually be included in this count by the representation
/// family that owns it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageBits(u64);

impl StorageBits {
    /// Construct an exact storage-bit count.
    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    /// Return the exact number of bits.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Primitive physical representation of one logical tensor value.
///
/// The enum is intentionally non-exhaustive: later phases may add block
/// quantization, codebooks or other representation families without changing
/// the logical tensor type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PrimitiveRepresentation {
    /// Dense scalar storage.
    Dense {
        /// Scalar dtype physically stored for each logical element.
        storage_dtype: DType,
    },
}

impl PrimitiveRepresentation {
    /// Construct the identity dense representation for a logical dtype.
    pub const fn dense(storage_dtype: DType) -> Self {
        Self::Dense { storage_dtype }
    }

    /// Return the dense storage dtype when this is a dense representation.
    pub const fn dense_dtype(&self) -> Option<DType> {
        match self
        {
            Self::Dense { storage_dtype } => Some(*storage_dtype),
        }
    }

    /// Return the exact physical storage required for `logical`.
    ///
    /// Dense storage uses the logical tensor shape and this representation's
    /// physical scalar dtype. The calculation is fully checked and never uses
    /// floating-point arithmetic.
    pub fn storage_bits(
        &self,
        logical: &crate::TensorType,
    ) -> Result<StorageBits, RepresentationError> {
        let storage_dtype = match self
        {
            Self::Dense { storage_dtype } => *storage_dtype,
        };

        if storage_dtype != logical.dtype
        {
            return Err(RepresentationError::DenseDTypeMismatch {
                logical: logical.dtype,
                storage: storage_dtype,
            });
        }

        let elements = logical
            .shape
            .checked_num_elements()
            .map_err(|_| RepresentationError::ShapeOverflow)?;

        let elements =
            u64::try_from(elements).map_err(|_| RepresentationError::StorageSizeOverflow)?;

        let bytes_per_element = u64::try_from(storage_dtype.size_bytes())
            .map_err(|_| RepresentationError::StorageSizeOverflow)?;

        let bits = elements
            .checked_mul(bytes_per_element)
            .and_then(|bytes| bytes.checked_mul(8))
            .ok_or(RepresentationError::StorageSizeOverflow)?;

        Ok(StorageBits::new(bits))
    }
}

/// Failure while constructing a representation plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RepresentationError {
    /// The representation identifier space exceeded `u32`.
    TooManyRepresentations,
    /// The logical tensor shape cannot be represented by `usize`.
    ShapeOverflow,
    /// The exact physical storage-bit count cannot be represented by `u64`.
    StorageSizeOverflow,
    /// Dense physical storage does not match the tensor's logical dtype.
    ///
    /// This initial IR phase models only identity dense storage. Any lossy or
    /// converting representation must be introduced explicitly by a later
    /// representation family with defined semantics.
    DenseDTypeMismatch {
        /// Logical scalar dtype declared by the canonical tensor type.
        logical: DType,
        /// Scalar dtype requested by the dense physical representation.
        storage: DType,
    },
}

impl fmt::Display for RepresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::TooManyRepresentations =>
            {
                write!(formatter, "representation identifier space exhausted")
            },
            Self::ShapeOverflow =>
            {
                write!(formatter, "logical tensor shape overflows usize")
            },
            Self::StorageSizeOverflow =>
            {
                write!(formatter, "representation storage size overflows u64 bits")
            },
            Self::DenseDTypeMismatch { logical, storage } =>
            {
                write!(
                    formatter,
                    "dense storage dtype {storage:?} does not match logical dtype {logical:?}"
                )
            },
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RepresentationError {}

/// Backend-neutral side table assigning one representation to every graph node.
///
/// The canonical [`Graph`] remains unchanged. Node identifiers are used only as
/// stable keys into this table; representation planning does not alter logical
/// dtype, shape, operation identity or graph topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationPlan {
    representations: Vec<PrimitiveRepresentation>,
    assignments: Vec<RepresentationId>,
}

impl RepresentationPlan {
    /// Build the identity representation plan for a canonical graph.
    ///
    /// Every node is assigned dense storage in its declared logical dtype.
    /// Equal dense representations are interned deterministically, so nodes with
    /// the same dtype share one [`RepresentationId`].
    pub fn dense(graph: &Graph) -> Result<Self, RepresentationError> {
        let mut representations = Vec::new();
        let mut assignments = Vec::with_capacity(graph.nodes().len());

        for node in graph.nodes()
        {
            let candidate = PrimitiveRepresentation::dense(node.output.dtype);

            let id = if let Some(index) = representations
                .iter()
                .position(|representation| representation == &candidate)
            {
                RepresentationId::new(
                    u32::try_from(index)
                        .map_err(|_| RepresentationError::TooManyRepresentations)?,
                )
            }
            else
            {
                let index = u32::try_from(representations.len())
                    .map_err(|_| RepresentationError::TooManyRepresentations)?;
                representations.push(candidate);
                RepresentationId::new(index)
            };

            assignments.push(id);
        }

        Ok(Self {
            representations,
            assignments,
        })
    }

    /// Return all interned representation declarations in canonical ID order.
    pub fn representations(&self) -> &[PrimitiveRepresentation] {
        &self.representations
    }

    /// Return the representation identifier assigned to `node`.
    pub fn assignment(&self, node: NodeId) -> Option<RepresentationId> {
        self.assignments.get(node.get() as usize).copied()
    }

    /// Resolve a representation identifier.
    pub fn representation(&self, id: RepresentationId) -> Option<&PrimitiveRepresentation> {
        self.representations.get(id.get() as usize)
    }

    /// Resolve the physical representation assigned to `node`.
    pub fn representation_for(&self, node: NodeId) -> Option<&PrimitiveRepresentation> {
        self.assignment(node).and_then(|id| self.representation(id))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use scirust_compute::{DType, Shape};

    use super::*;
    use crate::TensorType;

    fn tensor_type(dtype: DType) -> TensorType {
        TensorType::new(dtype, Shape::new(vec![2, 2]))
    }

    #[test]
    fn dense_plan_is_a_side_table_and_does_not_change_the_graph() {
        let mut graph = Graph::new();
        let input = graph.add_input("input", tensor_type(DType::F32)).unwrap();
        graph.set_outputs(vec![input]).unwrap();

        let before = graph.clone();
        let plan = RepresentationPlan::dense(&graph).unwrap();

        assert_eq!(graph, before);
        assert_eq!(
            plan.representation_for(input),
            Some(&PrimitiveRepresentation::dense(DType::F32))
        );
    }

    #[test]
    fn dense_plan_preserves_each_nodes_logical_dtype_as_storage_dtype() {
        let mut graph = Graph::new();
        let f32_node = graph.add_input("f32", tensor_type(DType::F32)).unwrap();
        let f16_node = graph.add_input("f16", tensor_type(DType::F16)).unwrap();
        graph.set_outputs(vec![f32_node, f16_node]).unwrap();

        let plan = RepresentationPlan::dense(&graph).unwrap();

        assert_eq!(
            plan.representation_for(f32_node)
                .and_then(PrimitiveRepresentation::dense_dtype),
            Some(DType::F32)
        );
        assert_eq!(
            plan.representation_for(f16_node)
                .and_then(PrimitiveRepresentation::dense_dtype),
            Some(DType::F16)
        );

        assert_eq!(
            graph.nodes()[f32_node.get() as usize].output.dtype,
            DType::F32
        );
        assert_eq!(
            graph.nodes()[f16_node.get() as usize].output.dtype,
            DType::F16
        );
    }

    #[test]
    fn identical_dense_representations_are_interned_deterministically() {
        let mut graph = Graph::new();
        let first = graph.add_input("first", tensor_type(DType::F32)).unwrap();
        let second = graph.add_input("second", tensor_type(DType::F32)).unwrap();
        let third = graph.add_input("third", tensor_type(DType::F64)).unwrap();
        graph.set_outputs(vec![first, second, third]).unwrap();

        let plan = RepresentationPlan::dense(&graph).unwrap();

        assert_eq!(plan.assignment(first), plan.assignment(second));
        assert_ne!(plan.assignment(first), plan.assignment(third));
        assert_eq!(plan.representations().len(), 2);
        assert_eq!(plan.assignment(first), Some(RepresentationId::new(0)));
        assert_eq!(plan.assignment(third), Some(RepresentationId::new(1)));
    }

    #[test]
    fn dense_f32_matrix_has_exact_storage_bits() {
        let logical = TensorType::new(DType::F32, Shape::new(vec![2, 2]));
        let representation = PrimitiveRepresentation::dense(DType::F32);

        assert_eq!(
            representation.storage_bits(&logical),
            Ok(StorageBits::new(128))
        );
    }

    #[test]
    fn dense_storage_rejects_implicit_dtype_conversion() {
        let logical = TensorType::new(DType::F32, Shape::new(vec![2, 2]));
        let representation = PrimitiveRepresentation::dense(DType::F16);

        assert_eq!(
            representation.storage_bits(&logical),
            Err(RepresentationError::DenseDTypeMismatch {
                logical: DType::F32,
                storage: DType::F16,
            })
        );
    }

    #[test]
    fn dense_f64_scalar_has_exact_storage_bits() {
        let logical = TensorType::new(DType::F64, Shape::scalar());
        let representation = PrimitiveRepresentation::dense(DType::F64);

        assert_eq!(
            representation.storage_bits(&logical),
            Ok(StorageBits::new(64))
        );
    }

    #[test]
    fn storage_accounting_preserves_shape_overflow_as_a_structured_error() {
        let logical = TensorType::new(DType::F32, Shape::new(vec![usize::MAX, 2]));
        let representation = PrimitiveRepresentation::dense(DType::F32);

        assert_eq!(
            representation.storage_bits(&logical),
            Err(RepresentationError::ShapeOverflow)
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn storage_accounting_rejects_bit_count_overflow() {
        let logical = TensorType::new(DType::U8, Shape::new(vec![usize::MAX]));
        let representation = PrimitiveRepresentation::dense(DType::U8);

        assert_eq!(
            representation.storage_bits(&logical),
            Err(RepresentationError::StorageSizeOverflow)
        );
    }
}
