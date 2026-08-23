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
}

/// Failure while constructing a representation plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RepresentationError {
    /// The representation identifier space exceeded `u32`.
    TooManyRepresentations,
}

impl fmt::Display for RepresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::TooManyRepresentations =>
            {
                write!(formatter, "representation identifier space exhausted")
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
}
