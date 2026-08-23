//! Backend-neutral representation planning for canonical tensor values.
//!
//! A [`RepresentationPlan`] is deliberately separate from [`crate::TensorType`].
//! `TensorType` describes the logical value (`dtype`, `shape`); this module
//! describes how that value may be represented for storage or execution.
//!
//! Representations are interned in a [`RepresentationPlan`] through a strict
//! declaration order: a representation may only be defined over components
//! referencing strictly earlier representation identifiers. Cycles are
//! therefore impossible by construction, mirroring how the canonical
//! [`Graph`] only permits inputs referring to previously created nodes.
//!
//! This first phase supports only the identity physical representation:
//! dense storage in the tensor's declared logical dtype. Quantization,
//! factorization, composite representations, cost models and backend-specific
//! materialization are intentionally out of scope.

use alloc::vec::Vec;
use core::fmt;

use scirust_compute::DType;

use crate::{Graph, NodeId, TensorType};

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

/// Typed reference to one representation declared in a [`RepresentationPlan`].
///
/// The [`RepresentationId`] identifies the physical representation family,
/// while `tensor_type` describes the tensor value carried by this particular
/// component. A component type is independent of the parent tensor type: future
/// representations may contain factors, packed payloads, scales, indices or
/// other typed tensor components with different shapes and dtypes.
///
/// Instances are constructed through [`RepresentationPlan::component`], which
/// validates that the identifier exists and is compatible with the component's
/// own tensor type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationComponent {
    tensor_type: TensorType,
    representation: RepresentationId,
}

impl RepresentationComponent {
    /// Tensor type carried by this representation component.
    pub const fn tensor_type(&self) -> &TensorType {
        &self.tensor_type
    }

    /// Interned physical representation assigned to this component.
    pub const fn representation(&self) -> RepresentationId {
        self.representation
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

    /// Iterate the validated [`RepresentationComponent`] dependencies this
    /// representation is declared over, in canonical field order.
    ///
    /// Dense storage declares no dependencies. Composite families yield their
    /// named components so generic passes can audit declaration ordering
    /// without matching every variant.
    pub fn components(&self) -> impl Iterator<Item = &RepresentationComponent> {
        let dependencies: &[&RepresentationComponent] = match self
        {
            Self::Dense { .. } => &[],
        };

        dependencies.iter().copied()
    }

    /// Validate that this physical representation can represent `logical`.
    ///
    /// Dense storage is currently an identity representation: changing dtype
    /// would require an explicitly defined conversion/quantization family.
    fn validate_for(&self, logical: &TensorType) -> Result<(), RepresentationError> {
        match self
        {
            Self::Dense { storage_dtype } if *storage_dtype != logical.dtype =>
            {
                Err(RepresentationError::DenseDTypeMismatch {
                    logical: logical.dtype,
                    storage: *storage_dtype,
                })
            },
            Self::Dense { .. } => Ok(()),
        }
    }

    /// Return the exact physical storage required for `logical`.
    ///
    /// Dense storage uses the logical tensor shape and this representation's
    /// physical scalar dtype. The calculation is fully checked and never uses
    /// floating-point arithmetic.
    pub fn storage_bits(&self, logical: &TensorType) -> Result<StorageBits, RepresentationError> {
        self.validate_for(logical)?;

        let storage_dtype = match self
        {
            Self::Dense { storage_dtype } => *storage_dtype,
        };

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
    /// A component referenced a representation not declared by its plan.
    InvalidRepresentationId {
        /// Unknown representation identifier.
        id: RepresentationId,
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
            Self::InvalidRepresentationId { id } =>
            {
                write!(
                    formatter,
                    "representation identifier {} is not declared by this plan",
                    id.get()
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
        let mut plan = Self {
            representations: Vec::new(),
            assignments: Vec::with_capacity(graph.nodes().len()),
        };

        for node in graph.nodes()
        {
            let representation = PrimitiveRepresentation::dense(node.output.dtype);
            let id = plan.declare(representation)?;
            plan.assignments.push(id);
        }

        Ok(plan)
    }

    /// Return all interned representation declarations in canonical ID order.
    pub fn representations(&self) -> &[PrimitiveRepresentation] {
        &self.representations
    }

    /// Declare and intern one representation.
    ///
    /// Every [`RepresentationComponent`] the representation is declared over
    /// must reference a representation already declared by this plan and must
    /// be compatible with its own tensor type. Unknown identifiers, forward
    /// references, self-references and incompatible component types are all
    /// rejected at this single insertion point. Because identifiers are
    /// assigned in increasing declaration order, dependencies always point
    /// strictly backwards and cycles are impossible by construction.
    ///
    /// Redeclaring an equal representation returns the existing identifier, so
    /// interning stays deterministic.
    pub(crate) fn declare(
        &mut self,
        primitive: PrimitiveRepresentation,
    ) -> Result<RepresentationId, RepresentationError> {
        let next = u32::try_from(self.representations.len())
            .map_err(|_| RepresentationError::TooManyRepresentations)?;

        let declared = self.representations.len();
        for dependency in primitive.components()
        {
            let id = dependency.representation();
            if id.get() as usize >= declared
            {
                return Err(RepresentationError::InvalidRepresentationId { id });
            }

            // Bounds checked above; the identifier is declared by this plan.
            let physical = &self.representations[id.get() as usize];
            physical.validate_for(dependency.tensor_type())?;
        }

        if let Some(index) = self
            .representations
            .iter()
            .position(|declared| *declared == primitive)
        {
            return Ok(RepresentationId::new(
                u32::try_from(index).map_err(|_| RepresentationError::TooManyRepresentations)?,
            ));
        }

        self.representations.push(primitive);

        Ok(RepresentationId::new(next))
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

    /// Construct a typed component referencing an interned representation.
    ///
    /// This validates both identifier membership and compatibility between the
    /// physical representation and the component's own tensor type. Valid
    /// components are the only way future composite representations may name
    /// their dependencies when calling [`RepresentationPlan::declare`].
    pub fn component(
        &self,
        tensor_type: TensorType,
        representation: RepresentationId,
    ) -> Result<RepresentationComponent, RepresentationError> {
        let physical = self
            .representation(representation)
            .ok_or(RepresentationError::InvalidRepresentationId { id: representation })?;

        physical.validate_for(&tensor_type)?;

        Ok(RepresentationComponent {
            tensor_type,
            representation,
        })
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
    fn typed_component_preserves_its_own_tensor_type() {
        let mut graph = Graph::new();
        let matrix_type = TensorType::new(DType::F32, Shape::new(vec![2, 2]));
        let vector_type = TensorType::new(DType::F32, Shape::new(vec![4]));

        let matrix = graph.add_input("matrix", matrix_type.clone()).unwrap();
        let vector = graph.add_input("vector", vector_type.clone()).unwrap();
        graph.set_outputs(vec![matrix, vector]).unwrap();

        let plan = RepresentationPlan::dense(&graph).unwrap();

        let matrix_id = plan.assignment(matrix).unwrap();
        let vector_id = plan.assignment(vector).unwrap();

        // Representation identity is reusable across shapes.
        assert_eq!(matrix_id, vector_id);

        let matrix_component = plan.component(matrix_type.clone(), matrix_id).unwrap();
        let vector_component = plan.component(vector_type.clone(), vector_id).unwrap();

        assert_eq!(matrix_component.tensor_type(), &matrix_type);
        assert_eq!(vector_component.tensor_type(), &vector_type);
        assert_eq!(matrix_component.representation(), matrix_id);
        assert_eq!(vector_component.representation(), vector_id);
    }

    #[test]
    fn typed_component_rejects_unknown_representation_id() {
        let graph = Graph::new();
        let plan = RepresentationPlan::dense(&graph).unwrap();
        let unknown = RepresentationId::new(7);

        assert_eq!(
            plan.component(TensorType::new(DType::F32, Shape::new(vec![2, 2])), unknown,),
            Err(RepresentationError::InvalidRepresentationId { id: unknown })
        );
    }

    #[test]
    fn typed_component_rejects_incompatible_dense_dtype() {
        let mut graph = Graph::new();
        let node = graph
            .add_input("f32", TensorType::new(DType::F32, Shape::new(vec![2, 2])))
            .unwrap();
        graph.set_outputs(vec![node]).unwrap();

        let plan = RepresentationPlan::dense(&graph).unwrap();
        let dense_f32 = plan.assignment(node).unwrap();

        assert_eq!(
            plan.component(
                TensorType::new(DType::F16, Shape::new(vec![2, 2])),
                dense_f32,
            ),
            Err(RepresentationError::DenseDTypeMismatch {
                logical: DType::F16,
                storage: DType::F32,
            })
        );
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

    fn empty_plan() -> RepresentationPlan {
        RepresentationPlan {
            representations: Vec::new(),
            assignments: Vec::new(),
        }
    }

    #[test]
    fn declaration_interning_is_deterministic() {
        let mut plan = empty_plan();

        let dense_f32 = plan
            .declare(PrimitiveRepresentation::dense(DType::F32))
            .unwrap();
        assert_eq!(dense_f32, RepresentationId::new(0));

        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        assert_eq!(dense_u8, RepresentationId::new(1));

        // Redeclaring an equal representation returns the interned identifier.
        assert_eq!(
            plan.declare(PrimitiveRepresentation::dense(DType::F32)),
            Ok(dense_f32)
        );
        assert_eq!(
            plan.declare(PrimitiveRepresentation::dense(DType::U8)),
            Ok(dense_u8)
        );

        // An independently built plan assigns identical identifiers in
        // first-use order.
        let mut twin = empty_plan();
        assert_eq!(
            twin.declare(PrimitiveRepresentation::dense(DType::F32)),
            Ok(dense_f32)
        );
        assert_eq!(
            twin.declare(PrimitiveRepresentation::dense(DType::U8)),
            Ok(dense_u8)
        );
    }

    #[test]
    fn typed_component_rejects_forward_and_self_references() {
        let mut plan = empty_plan();
        let dense_f32 = plan
            .declare(PrimitiveRepresentation::dense(DType::F32))
            .unwrap();

        // The next identifier to be assigned is exactly the identifier a
        // declaration would receive: referencing it now would be a cycle.
        let next = RepresentationId::new(dense_f32.get() + 1);
        assert_eq!(
            plan.component(tensor_type(DType::F32), next),
            Err(RepresentationError::InvalidRepresentationId { id: next })
        );

        let forward = RepresentationId::new(dense_f32.get() + 42);
        assert_eq!(
            plan.component(tensor_type(DType::F32), forward),
            Err(RepresentationError::InvalidRepresentationId { id: forward })
        );
    }

    #[test]
    fn declaring_representations_does_not_disturb_existing_assignments() {
        let mut graph = Graph::new();
        let f32_node = graph.add_input("f32", tensor_type(DType::F32)).unwrap();
        let f16_node = graph.add_input("f16", tensor_type(DType::F16)).unwrap();
        graph.set_outputs(vec![f32_node, f16_node]).unwrap();

        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let f32_id = plan.assignment(f32_node).unwrap();
        let f16_id = plan.assignment(f16_node).unwrap();
        let before = plan.representations().to_vec();

        plan.declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();

        assert_eq!(plan.assignment(f32_node), Some(f32_id));
        assert_eq!(plan.assignment(f16_node), Some(f16_id));
        assert_eq!(&plan.representations()[..before.len()], &before[..]);
        assert_eq!(plan.representations().len(), before.len() + 1);
    }
}
