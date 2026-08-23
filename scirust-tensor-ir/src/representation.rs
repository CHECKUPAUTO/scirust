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
//! Dense storage remains the identity representation. The composite families
//! are [`PrimitiveRepresentation::Factorized`] (two contracted matrix factors)
//! and [`PrimitiveRepresentation::Quantized`] (discrete codes corrected by
//! continuous scales). Storage is accounted as the exact sum of declared
//! components. Codebook layouts, block geometry, packed/sub-bit payloads,
//! sparse representations, cost models and backend-specific materialization
//! remain intentionally out of scope.

use alloc::vec::Vec;
use core::fmt;

use scirust_compute::{DType, Shape};

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
/// the logical tensor type. Composite variants own their components as named,
/// typed fields; every referenced [`RepresentationComponent`] must point to a
/// strictly earlier declaration in the same plan.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrimitiveRepresentation {
    /// Dense scalar storage.
    Dense {
        /// Scalar dtype physically stored for each logical element.
        storage_dtype: DType,
    },
    /// Matrix-factorized storage: the represented value is defined by two
    /// contracted factors (`left × right`), e.g. low-rank or adapter-style
    /// decompositions.
    ///
    /// Factor dtypes are unconstrained because a factorized representation is
    /// inherently converting; only the contraction structure binds the factors
    /// to the logical tensor type: `left [m, r]`, `right [r, n]` for a logical
    /// `[m, n]` value.
    Factorized {
        /// Left factor with shape `[m, r]`.
        left: RepresentationComponent,
        /// Right factor with shape `[r, n]`.
        right: RepresentationComponent,
    },
    /// Affine quantization skeleton: discrete `codes` corrected by continuous
    /// `scales`.
    ///
    /// The family contract constrains component dtypes only, and is enforced
    /// once at declaration time: codes must be integer-valued (discrete
    /// payload) and scales floating-point (continuous correction). Block
    /// layouts, group sizes, zero-points and codebook geometry belong to
    /// concrete schemes and stay out of scope; no shape relationship binds the
    /// components to each other or to the logical tensor type.
    Quantized {
        /// Discrete code payload with an integer dtype.
        codes: RepresentationComponent,
        /// Continuous dequantization factors with a floating dtype.
        scales: RepresentationComponent,
    },
}

impl PrimitiveRepresentation {
    /// Construct the identity dense representation for a logical dtype.
    pub const fn dense(storage_dtype: DType) -> Self {
        Self::Dense { storage_dtype }
    }

    /// Construct a factorized representation from two contracted factors.
    pub const fn factorized(left: RepresentationComponent, right: RepresentationComponent) -> Self {
        Self::Factorized { left, right }
    }

    /// Construct a quantized representation from codes and scales.
    pub const fn quantized(
        codes: RepresentationComponent,
        scales: RepresentationComponent,
    ) -> Self {
        Self::Quantized { codes, scales }
    }

    /// Return the dense storage dtype when this is a dense representation.
    pub const fn dense_dtype(&self) -> Option<DType> {
        match self
        {
            Self::Dense { storage_dtype } => Some(*storage_dtype),
            Self::Factorized { .. } | Self::Quantized { .. } => None,
        }
    }

    /// Iterate the validated [`RepresentationComponent`] dependencies this
    /// representation is declared over, in canonical field order.
    ///
    /// Dense storage declares no dependencies. Composite families yield their
    /// named components so generic passes can audit declaration ordering
    /// without matching every variant.
    pub fn components(&self) -> impl Iterator<Item = &RepresentationComponent> {
        let (first, second, third): (
            Option<&RepresentationComponent>,
            Option<&RepresentationComponent>,
            Option<&RepresentationComponent>,
        ) = match self
        {
            Self::Dense { .. } => (None, None, None),
            Self::Factorized { left, right } => (Some(left), Some(right), None),
            Self::Quantized { codes, scales } => (Some(codes), Some(scales), None),
        };

        first.into_iter().chain(second).chain(third)
    }

    /// Validate that this physical representation can represent `logical`.
    ///
    /// Dense storage is currently an identity representation: changing dtype
    /// would require an explicitly defined conversion/quantization family.
    /// Factorized storage is inherently converting, so only its contraction
    /// structure is validated against the logical tensor type. Quantized
    /// storage carries no logical binding yet; its family contract is enforced
    /// once at declaration time by [`PrimitiveRepresentation::validate_declaration`].
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
            Self::Factorized { left, right } =>
            {
                match (
                    matrix_dims(&logical.shape),
                    matrix_dims(&left.tensor_type().shape),
                    matrix_dims(&right.tensor_type().shape),
                )
                {
                    (
                        Some((rows, columns)),
                        Some((left_rows, inner)),
                        Some((right_inner, right_columns)),
                    ) if left_rows == rows && right_inner == inner && right_columns == columns =>
                    {
                        Ok(())
                    },
                    _ => Err(RepresentationError::FactorizedIncompatibleShapes {
                        left: left.tensor_type().shape.clone(),
                        right: right.tensor_type().shape.clone(),
                        logical: logical.shape.clone(),
                    }),
                }
            },
            Self::Quantized { .. } => Ok(()),
        }
    }

    /// Validate contracts internal to a declaration, independent of any
    /// logical tensor binding.
    ///
    /// Runs once when a representation enters a plan so that invalid
    /// declarations can never be interned. Structure that only makes sense
    /// against a logical value (dense identity, factor contraction) stays in
    /// [`PrimitiveRepresentation::validate_for`].
    fn validate_declaration(&self) -> Result<(), RepresentationError> {
        match self
        {
            Self::Dense { .. } | Self::Factorized { .. } => Ok(()),
            Self::Quantized { codes, scales } =>
            {
                let codes_dtype = codes.tensor_type().dtype;
                let scales_dtype = scales.tensor_type().dtype;

                if !is_integer_dtype(codes_dtype) || !is_float_dtype(scales_dtype)
                {
                    Err(RepresentationError::QuantizedInvalidComponentDTypes {
                        codes: codes_dtype,
                        scales: scales_dtype,
                    })
                }
                else
                {
                    Ok(())
                }
            },
        }
    }
}

/// Return `(rows, columns)` when `shape` is exactly a matrix shape.
fn matrix_dims(shape: &Shape) -> Option<(usize, usize)> {
    match shape.dims()
    {
        [rows, columns] => Some((*rows, *columns)),
        _ => None,
    }
}

/// Whether `dtype` carries discrete integer values.
///
/// `Bool` is excluded: it is a logical flag, not a code value.
fn is_integer_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::U8
            | DType::I8
            | DType::U16
            | DType::I16
            | DType::U32
            | DType::I32
            | DType::U64
            | DType::I64
    )
}

/// Whether `dtype` carries continuous floating-point values.
fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64)
}

/// Exact bit count of dense scalar storage for `logical`.
///
/// Fully checked integer arithmetic; floating-point sizes are never used for
/// exact physical accounting.
fn dense_storage_bits(
    storage_dtype: DType,
    logical: &TensorType,
) -> Result<StorageBits, RepresentationError> {
    let elements = logical
        .shape
        .checked_num_elements()
        .map_err(|_| RepresentationError::ShapeOverflow)?;

    let elements = u64::try_from(elements).map_err(|_| RepresentationError::StorageSizeOverflow)?;

    let bytes_per_element = u64::try_from(storage_dtype.size_bytes())
        .map_err(|_| RepresentationError::StorageSizeOverflow)?;

    let bits = elements
        .checked_mul(bytes_per_element)
        .and_then(|bytes| bytes.checked_mul(8))
        .ok_or(RepresentationError::StorageSizeOverflow)?;

    Ok(StorageBits::new(bits))
}

/// Failure while constructing a representation plan.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Factorized factors do not contract into the logical tensor type.
    ///
    /// A factorized representation requires matrix factors `left [m, r]` and
    /// `right [r, n]` to represent a logical `[m, n]` tensor type.
    FactorizedIncompatibleShapes {
        /// Left factor shape.
        left: Shape,
        /// Right factor shape.
        right: Shape,
        /// Logical tensor shape.
        logical: Shape,
    },
    /// Quantized component dtypes violate the family contract.
    ///
    /// Codes must carry discrete integer values and scales continuous
    /// floating-point values.
    QuantizedInvalidComponentDTypes {
        /// Dtype carried by the codes component.
        codes: DType,
        /// Dtype carried by the scales component.
        scales: DType,
    },
    /// The node is outside the plan's assignment scope.
    ///
    /// Assignments are indexed by canonical node identifiers; the plan only
    /// covers the nodes of the graph it was seeded from.
    UnknownAssignmentNode {
        /// Node identifier outside the assignment scope.
        node: NodeId,
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
            Self::FactorizedIncompatibleShapes {
                left,
                right,
                logical,
            } =>
            {
                write!(
                    formatter,
                    "factorized shapes left {left:?} x right {right:?} do not compose into logical shape {logical:?}"
                )
            },
            Self::QuantizedInvalidComponentDTypes { codes, scales } =>
            {
                write!(
                    formatter,
                    "quantized codes dtype {codes:?} must be integer-valued and scales dtype {scales:?} floating-point"
                )
            },
            Self::UnknownAssignmentNode { node } =>
            {
                write!(
                    formatter,
                    "node {} is outside this plan's assignment scope",
                    node.get()
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

        primitive.validate_declaration()?;

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

    /// Return the exact physical storage required to represent `logical` with
    /// the declared representation `id`.
    ///
    /// Dense storage counts `elements × dtype size × 8` bits. Composite
    /// families validate their structure against `logical`, then sum the exact
    /// storage of their declared components, resolving nested references
    /// recursively (references point strictly backwards, so recursion
    /// terminates). All arithmetic is checked and never uses floating-point.
    pub fn storage_bits(
        &self,
        id: RepresentationId,
        logical: &TensorType,
    ) -> Result<StorageBits, RepresentationError> {
        let physical = self
            .representation(id)
            .ok_or(RepresentationError::InvalidRepresentationId { id })?;

        physical.validate_for(logical)?;

        match physical
        {
            PrimitiveRepresentation::Dense { storage_dtype } =>
            {
                dense_storage_bits(*storage_dtype, logical)
            },
            PrimitiveRepresentation::Factorized { left, right } =>
            {
                let left_bits = self.storage_bits(left.representation(), left.tensor_type())?;
                let right_bits = self.storage_bits(right.representation(), right.tensor_type())?;

                left_bits
                    .get()
                    .checked_add(right_bits.get())
                    .map(StorageBits::new)
                    .ok_or(RepresentationError::StorageSizeOverflow)
            },
            PrimitiveRepresentation::Quantized { codes, scales } =>
            {
                let codes_bits = self.storage_bits(codes.representation(), codes.tensor_type())?;
                let scales_bits =
                    self.storage_bits(scales.representation(), scales.tensor_type())?;

                codes_bits
                    .get()
                    .checked_add(scales_bits.get())
                    .map(StorageBits::new)
                    .ok_or(RepresentationError::StorageSizeOverflow)
            },
        }
    }

    /// Construct a typed component referencing an interned representation.
    ///
    /// This validates both identifier membership and compatibility between the
    /// physical representation and the component's own tensor type. Composite
    /// representations name their dependencies through such validated
    /// components when calling [`RepresentationPlan::declare`].
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

    /// Bind `node` to the declared representation `id`.
    ///
    /// The node must lie inside this plan's assignment scope (the node set the
    /// plan was seeded from) and the representation must be able to represent
    /// the node's canonical tensor type. Rejected bindings leave the table
    /// unchanged; successful ones override any previous assignment of that
    /// node without touching other nodes or the graph itself.
    pub fn assign(
        &mut self,
        graph: &Graph,
        node: NodeId,
        id: RepresentationId,
    ) -> Result<(), RepresentationError> {
        let index = node.get() as usize;
        if index >= self.assignments.len()
        {
            return Err(RepresentationError::UnknownAssignmentNode { node });
        }

        // The plan scope guarantees the index, but a mismatched or truncated
        // graph is rejected instead of trusted.
        let logical = &graph
            .nodes()
            .get(index)
            .ok_or(RepresentationError::UnknownAssignmentNode { node })?
            .output;

        self.representation(id)
            .ok_or(RepresentationError::InvalidRepresentationId { id })?
            .validate_for(logical)?;

        self.assignments[index] = id;

        Ok(())
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
        let mut plan = empty_plan();
        let dense_f32 = plan
            .declare(PrimitiveRepresentation::dense(DType::F32))
            .unwrap();
        let logical = TensorType::new(DType::F32, Shape::new(vec![2, 2]));

        assert_eq!(
            plan.storage_bits(dense_f32, &logical),
            Ok(StorageBits::new(128))
        );
    }

    #[test]
    fn dense_storage_rejects_implicit_dtype_conversion() {
        let mut plan = empty_plan();
        let dense_f16 = plan
            .declare(PrimitiveRepresentation::dense(DType::F16))
            .unwrap();
        let logical = TensorType::new(DType::F32, Shape::new(vec![2, 2]));

        assert_eq!(
            plan.storage_bits(dense_f16, &logical),
            Err(RepresentationError::DenseDTypeMismatch {
                logical: DType::F32,
                storage: DType::F16,
            })
        );
    }

    #[test]
    fn dense_f64_scalar_has_exact_storage_bits() {
        let mut plan = empty_plan();
        let dense_f64 = plan
            .declare(PrimitiveRepresentation::dense(DType::F64))
            .unwrap();
        let logical = TensorType::new(DType::F64, Shape::scalar());

        assert_eq!(
            plan.storage_bits(dense_f64, &logical),
            Ok(StorageBits::new(64))
        );
    }

    #[test]
    fn storage_accounting_preserves_shape_overflow_as_a_structured_error() {
        let mut plan = empty_plan();
        let dense_f32 = plan
            .declare(PrimitiveRepresentation::dense(DType::F32))
            .unwrap();
        let logical = TensorType::new(DType::F32, Shape::new(vec![usize::MAX, 2]));

        assert_eq!(
            plan.storage_bits(dense_f32, &logical),
            Err(RepresentationError::ShapeOverflow)
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn storage_accounting_rejects_bit_count_overflow() {
        let mut plan = empty_plan();
        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        let logical = TensorType::new(DType::U8, Shape::new(vec![usize::MAX]));

        assert_eq!(
            plan.storage_bits(dense_u8, &logical),
            Err(RepresentationError::StorageSizeOverflow)
        );
    }

    fn empty_plan() -> RepresentationPlan {
        RepresentationPlan {
            representations: Vec::new(),
            assignments: Vec::new(),
        }
    }

    /// Seed a plan with one dense factor representation and build contracted
    /// matrix components `[rows, inner]` and `[inner, columns]` over it.
    fn factorized_components(
        plan: &mut RepresentationPlan,
        factor_dtype: DType,
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> (RepresentationComponent, RepresentationComponent) {
        let dense_factor = plan
            .declare(PrimitiveRepresentation::dense(factor_dtype))
            .unwrap();

        let left = plan
            .component(
                TensorType::new(factor_dtype, Shape::new(vec![rows, inner])),
                dense_factor,
            )
            .unwrap();
        let right = plan
            .component(
                TensorType::new(factor_dtype, Shape::new(vec![inner, columns])),
                dense_factor,
            )
            .unwrap();

        (left, right)
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

    #[test]
    fn factorized_binds_only_to_contracting_matrix_types() {
        let mut plan = empty_plan();
        let (left, right) = factorized_components(&mut plan, DType::F16, 4, 8, 2);
        let factored = plan
            .declare(PrimitiveRepresentation::factorized(
                left.clone(),
                right.clone(),
            ))
            .unwrap();

        // The contracting logical type binds successfully.
        let logical = TensorType::new(DType::F32, Shape::new(vec![4, 2]));
        assert_eq!(
            plan.component(logical.clone(), factored)
                .unwrap()
                .tensor_type(),
            &logical
        );

        // Mismatched outer dimensions are rejected with structured shapes.
        assert_eq!(
            plan.component(
                TensorType::new(DType::F32, Shape::new(vec![4, 3])),
                factored,
            ),
            Err(RepresentationError::FactorizedIncompatibleShapes {
                left: Shape::new(vec![4, 8]),
                right: Shape::new(vec![8, 2]),
                logical: Shape::new(vec![4, 3]),
            })
        );

        // Non-matrix logical types are rejected as well.
        assert_eq!(
            plan.component(TensorType::new(DType::F32, Shape::new(vec![4])), factored,),
            Err(RepresentationError::FactorizedIncompatibleShapes {
                left: Shape::new(vec![4, 8]),
                right: Shape::new(vec![8, 2]),
                logical: Shape::new(vec![4]),
            })
        );

        // Factor-side violations: a non-matrix left factor cannot contract.
        let dense_f16 = plan
            .declare(PrimitiveRepresentation::dense(DType::F16))
            .unwrap();
        let rank_one_left = plan
            .component(TensorType::new(DType::F16, Shape::new(vec![4])), dense_f16)
            .unwrap();
        let degenerate = PrimitiveRepresentation::factorized(rank_one_left.clone(), right);

        assert_eq!(
            degenerate.validate_for(&TensorType::new(DType::F16, Shape::new(vec![4, 2]))),
            Err(RepresentationError::FactorizedIncompatibleShapes {
                left: Shape::new(vec![4]),
                right: Shape::new(vec![8, 2]),
                logical: Shape::new(vec![4, 2]),
            })
        );

        // Inner contraction mismatch between matrix factors is rejected.
        let wrong_inner_right = plan
            .component(
                TensorType::new(DType::F16, Shape::new(vec![7, 2])),
                dense_f16,
            )
            .unwrap();
        let mismatched = PrimitiveRepresentation::factorized(left, wrong_inner_right);

        assert_eq!(
            mismatched.validate_for(&TensorType::new(DType::F16, Shape::new(vec![4, 2]))),
            Err(RepresentationError::FactorizedIncompatibleShapes {
                left: Shape::new(vec![4, 8]),
                right: Shape::new(vec![7, 2]),
                logical: Shape::new(vec![4, 2]),
            })
        );
    }

    #[test]
    fn factorized_storage_bits_sum_component_storage_exactly() {
        let mut plan = empty_plan();
        let (left, right) = factorized_components(&mut plan, DType::F16, 4, 8, 2);
        let factored = plan
            .declare(PrimitiveRepresentation::factorized(left, right))
            .unwrap();

        // The parent dtype is irrelevant to a converting representation.
        let logical = TensorType::new(DType::F32, Shape::new(vec![4, 2]));

        // 4*8 elements * 2 bytes * 8 bits + 8*2 elements * 2 bytes * 8 bits.
        assert_eq!(
            plan.storage_bits(factored, &logical),
            Ok(StorageBits::new(512 + 256))
        );
    }

    #[test]
    fn factorized_declaration_identity_includes_named_components() {
        let mut plan = empty_plan();
        let (left, right) = factorized_components(&mut plan, DType::F16, 4, 8, 2);
        let first = plan
            .declare(PrimitiveRepresentation::factorized(
                left.clone(),
                right.clone(),
            ))
            .unwrap();

        // Redeclaring an equal composite interns the existing identifier.
        assert_eq!(
            plan.declare(PrimitiveRepresentation::factorized(
                left.clone(),
                right.clone()
            )),
            Ok(first)
        );
        assert_eq!(first, RepresentationId::new(1));

        // Changing one named component changes declaration identity.
        let dense_f16 = plan
            .declare(PrimitiveRepresentation::dense(DType::F16))
            .unwrap();
        let narrower_right = plan
            .component(
                TensorType::new(DType::F16, Shape::new(vec![8, 3])),
                dense_f16,
            )
            .unwrap();
        let second = plan
            .declare(PrimitiveRepresentation::factorized(left, narrower_right))
            .unwrap();

        assert_ne!(second, first);
        assert!(second.get() > first.get());

        // Every declared dependency of every declaration points strictly
        // backwards.
        for (index, declared) in plan.representations().iter().enumerate()
        {
            for dependency in declared.components()
            {
                assert!(dependency.representation().get() < index as u32);
            }
        }
    }

    #[test]
    fn declare_rejects_factorized_referencing_undeclared_representations() {
        let mut source = empty_plan();
        let (left, right) = factorized_components(&mut source, DType::F16, 4, 8, 2);
        let foreign = PrimitiveRepresentation::factorized(left, right);

        // The referenced dense representation exists only in the source plan;
        // the empty target plan must reject the forward dependency.
        let mut target = empty_plan();
        assert_eq!(
            target.declare(foreign.clone()),
            Err(RepresentationError::InvalidRepresentationId {
                id: RepresentationId::new(0)
            })
        );

        // Once the target declares the same dependency state, the identical
        // value becomes a legitimate backward reference.
        target
            .declare(PrimitiveRepresentation::dense(DType::F16))
            .unwrap();
        assert_eq!(target.declare(foreign), Ok(RepresentationId::new(1)));
    }

    #[test]
    fn nested_factorized_compositions_resolve_storage_recursively() {
        let mut plan = empty_plan();
        let (inner_left, inner_right) = factorized_components(&mut plan, DType::F16, 2, 3, 4);
        let inner = plan
            .declare(PrimitiveRepresentation::factorized(inner_left, inner_right))
            .unwrap();

        // The inner composition represents a [2, 4] value and may itself be
        // used as the left factor of an outer composition.
        let outer_left = plan
            .component(TensorType::new(DType::F16, Shape::new(vec![2, 4])), inner)
            .unwrap();
        let dense_f16 = plan
            .declare(PrimitiveRepresentation::dense(DType::F16))
            .unwrap();
        let outer_right = plan
            .component(
                TensorType::new(DType::F16, Shape::new(vec![4, 5])),
                dense_f16,
            )
            .unwrap();
        let outer = plan
            .declare(PrimitiveRepresentation::factorized(outer_left, outer_right))
            .unwrap();

        assert!(outer.get() > inner.get());

        // A[2,3] + B[3,4] + C[4,5], all F16:
        // (6 + 12 + 20) elements * 2 bytes * 8 bits.
        let logical = TensorType::new(DType::F32, Shape::new(vec![2, 5]));
        assert_eq!(
            plan.storage_bits(outer, &logical),
            Ok(StorageBits::new(608))
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn factorized_storage_rejects_checked_sum_overflow() {
        let mut plan = empty_plan();
        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        let two_pow_59 = 576_460_752_303_423_488_usize;

        let left = plan
            .component(
                TensorType::new(DType::U8, Shape::new(vec![two_pow_59, 2])),
                dense_u8,
            )
            .unwrap();
        let right = plan
            .component(
                TensorType::new(DType::U8, Shape::new(vec![2, two_pow_59])),
                dense_u8,
            )
            .unwrap();
        let factored = plan
            .declare(PrimitiveRepresentation::factorized(left, right))
            .unwrap();

        // Each factor accounts exactly 2^63 bits; their sum overflows u64.
        let logical = TensorType::new(DType::U8, Shape::new(vec![two_pow_59, two_pow_59]));
        assert_eq!(
            plan.storage_bits(factored, &logical),
            Err(RepresentationError::StorageSizeOverflow)
        );
    }

    #[test]
    fn factorized_components_keep_their_own_tensor_types_and_representations() {
        let mut plan = empty_plan();
        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        let dense_f32 = plan
            .declare(PrimitiveRepresentation::dense(DType::F32))
            .unwrap();

        let left = plan
            .component(TensorType::new(DType::U8, Shape::new(vec![2, 8])), dense_u8)
            .unwrap();
        let right = plan
            .component(
                TensorType::new(DType::F32, Shape::new(vec![8, 2])),
                dense_f32,
            )
            .unwrap();
        let factored = plan
            .declare(PrimitiveRepresentation::factorized(
                left.clone(),
                right.clone(),
            ))
            .unwrap();

        // Mixed-dtype factors over distinct representations are legitimate;
        // each component keeps its own tensor type and reference.
        assert!(factored.get() > dense_u8.get());
        assert!(factored.get() > dense_f32.get());

        match plan.representation(factored)
        {
            Some(PrimitiveRepresentation::Factorized { left, right }) =>
            {
                assert_eq!(left.tensor_type().dtype, DType::U8);
                assert_eq!(
                    left.tensor_type(),
                    &TensorType::new(DType::U8, Shape::new(vec![2, 8]))
                );
                assert_eq!(left.representation(), dense_u8);
                assert_eq!(
                    right.tensor_type(),
                    &TensorType::new(DType::F32, Shape::new(vec![8, 2]))
                );
                assert_eq!(right.representation(), dense_f32);
            },
            other => panic!("unexpected stored representation: {other:?}"),
        }
    }

    #[test]
    fn quantized_declares_with_integer_codes_and_float_scales() {
        let mut plan = empty_plan();
        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        let dense_f32 = plan
            .declare(PrimitiveRepresentation::dense(DType::F32))
            .unwrap();

        let codes = plan
            .component(TensorType::new(DType::U8, Shape::new(vec![4, 2])), dense_u8)
            .unwrap();
        let scales = plan
            .component(TensorType::new(DType::F32, Shape::new(vec![4])), dense_f32)
            .unwrap();

        let quantized = plan
            .declare(PrimitiveRepresentation::quantized(
                codes.clone(),
                scales.clone(),
            ))
            .unwrap();

        // Redeclaring an equal composite interns the existing identifier.
        assert_eq!(quantized, RepresentationId::new(2));
        assert_eq!(
            plan.declare(PrimitiveRepresentation::quantized(codes, scales)),
            Ok(quantized)
        );

        // Codes and scales keep their named roles, dtypes and references.
        match plan.representation(quantized)
        {
            Some(PrimitiveRepresentation::Quantized { codes, scales }) =>
            {
                assert_eq!(codes.tensor_type().dtype, DType::U8);
                assert_eq!(codes.representation(), dense_u8);
                assert_eq!(scales.tensor_type().dtype, DType::F32);
                assert_eq!(scales.representation(), dense_f32);
            },
            other => panic!("unexpected stored representation: {other:?}"),
        }
    }

    #[test]
    fn quantized_storage_bits_sum_component_storage_exactly() {
        let mut plan = empty_plan();
        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        let dense_f32 = plan
            .declare(PrimitiveRepresentation::dense(DType::F32))
            .unwrap();

        let codes = plan
            .component(TensorType::new(DType::U8, Shape::new(vec![4, 2])), dense_u8)
            .unwrap();
        let scales = plan
            .component(TensorType::new(DType::F32, Shape::new(vec![4])), dense_f32)
            .unwrap();
        let quantized = plan
            .declare(PrimitiveRepresentation::quantized(codes, scales))
            .unwrap();

        // The parent dtype is irrelevant to a converting representation.
        let logical = TensorType::new(DType::F32, Shape::new(vec![4, 2]));

        // 4*2 code bytes * 8 bits + 4 scale words * 4 bytes * 8 bits.
        assert_eq!(
            plan.storage_bits(quantized, &logical),
            Ok(StorageBits::new(64 + 128))
        );
    }

    #[test]
    fn quantized_declaration_rejects_invalid_component_dtypes_atomically() {
        let mut plan = empty_plan();
        let dense_bool = plan
            .declare(PrimitiveRepresentation::dense(DType::Bool))
            .unwrap();
        let dense_f16 = plan
            .declare(PrimitiveRepresentation::dense(DType::F16))
            .unwrap();
        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        let dense_f32 = plan
            .declare(PrimitiveRepresentation::dense(DType::F32))
            .unwrap();

        let component_of = |plan: &RepresentationPlan, dtype: DType, shape: &[usize], id| {
            plan.component(TensorType::new(dtype, Shape::new(shape.to_vec())), id)
                .unwrap()
        };

        // Non-integer codes are rejected before interning.
        assert_eq!(
            plan.declare(PrimitiveRepresentation::quantized(
                component_of(&plan, DType::F16, &[4], dense_f16),
                component_of(&plan, DType::F32, &[4], dense_f32),
            )),
            Err(RepresentationError::QuantizedInvalidComponentDTypes {
                codes: DType::F16,
                scales: DType::F32,
            })
        );

        // Boolean payloads are flags, not code values.
        assert_eq!(
            plan.declare(PrimitiveRepresentation::quantized(
                component_of(&plan, DType::Bool, &[4], dense_bool),
                component_of(&plan, DType::F32, &[4], dense_f32),
            )),
            Err(RepresentationError::QuantizedInvalidComponentDTypes {
                codes: DType::Bool,
                scales: DType::F32,
            })
        );

        // Non-floating scales are rejected as well.
        assert_eq!(
            plan.declare(PrimitiveRepresentation::quantized(
                component_of(&plan, DType::U8, &[4, 2], dense_u8),
                component_of(&plan, DType::U8, &[4], dense_u8),
            )),
            Err(RepresentationError::QuantizedInvalidComponentDTypes {
                codes: DType::U8,
                scales: DType::U8,
            })
        );

        // Rejected declarations never enter the table.
        assert_eq!(plan.representations().len(), 4);
        for (index, declared) in plan.representations().iter().enumerate()
        {
            for dependency in declared.components()
            {
                assert!(dependency.representation().get() < index as u32);
            }
        }
    }

    #[test]
    fn quantized_binds_to_logical_types_without_structural_constraints() {
        let mut plan = empty_plan();
        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        let dense_f16 = plan
            .declare(PrimitiveRepresentation::dense(DType::F16))
            .unwrap();

        let codes = plan
            .component(TensorType::new(DType::U8, Shape::new(vec![64])), dense_u8)
            .unwrap();
        let scales = plan
            .component(TensorType::new(DType::F16, Shape::new(vec![8])), dense_f16)
            .unwrap();
        let quantized = plan
            .declare(PrimitiveRepresentation::quantized(codes, scales))
            .unwrap();

        // Block geometry is scheme-specific and out of IR scope: any logical
        // type binds as long as components were declared validly.
        for dims in [vec![64], vec![8, 8], vec![2, 4, 8]]
        {
            let logical = TensorType::new(DType::F32, Shape::new(dims));
            assert_eq!(
                plan.storage_bits(quantized, &logical),
                Ok(StorageBits::new(64 * 8 + 8 * 2 * 8))
            );
        }
    }

    #[test]
    fn assign_overrides_dense_defaults_after_validation() {
        let mut graph = Graph::new();
        let weight = graph
            .add_input(
                "weight",
                TensorType::new(DType::F32, Shape::new(vec![4, 2])),
            )
            .unwrap();
        let bias = graph
            .add_input("bias", TensorType::new(DType::F32, Shape::new(vec![4])))
            .unwrap();
        graph.set_outputs(vec![weight, bias]).unwrap();

        let before = graph.clone();
        let mut plan = RepresentationPlan::dense(&graph).unwrap();

        let (left, right) = factorized_components(&mut plan, DType::F16, 4, 3, 2);
        let factored = plan
            .declare(PrimitiveRepresentation::factorized(left, right))
            .unwrap();

        plan.assign(&graph, weight, factored).unwrap();

        // The bound node uses the factorized representation; the untouched
        // node keeps its dense default and the graph is never modified.
        assert_eq!(plan.assignment(weight), Some(factored));
        assert_eq!(
            plan.representation_for(weight)
                .and_then(|physical| physical.dense_dtype()),
            None
        );
        assert_eq!(
            plan.representation_for(bias)
                .and_then(PrimitiveRepresentation::dense_dtype),
            Some(DType::F32)
        );
        assert_eq!(graph, before);
    }

    #[test]
    fn assign_rejects_incompatible_bindings_atomically() {
        let mut graph = Graph::new();
        let weight = graph
            .add_input(
                "weight",
                TensorType::new(DType::F32, Shape::new(vec![4, 2])),
            )
            .unwrap();
        graph.set_outputs(vec![weight]).unwrap();

        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let dense_f32 = plan.assignment(weight).unwrap();

        // The factors contract to [2, 5], not to the node's [4, 2].
        let (left, right) = factorized_components(&mut plan, DType::F16, 2, 3, 5);
        let factored = plan
            .declare(PrimitiveRepresentation::factorized(left, right))
            .unwrap();

        assert_eq!(
            plan.assign(&graph, weight, factored),
            Err(RepresentationError::FactorizedIncompatibleShapes {
                left: Shape::new(vec![2, 3]),
                right: Shape::new(vec![3, 5]),
                logical: Shape::new(vec![4, 2]),
            })
        );

        // A rejected binding leaves the previous assignment in place.
        assert_eq!(plan.assignment(weight), Some(dense_f32));
        assert_eq!(plan.assignments.len(), 1);
    }

    #[test]
    fn assign_rejects_undeclared_representations_and_unknown_nodes() {
        let mut graph = Graph::new();
        let node = graph.add_input("x", tensor_type(DType::F32)).unwrap();
        graph.set_outputs(vec![node]).unwrap();

        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let dense_f32 = plan.assignment(node).unwrap();

        let undeclared = RepresentationId::new(9);
        assert_eq!(
            plan.assign(&graph, node, undeclared),
            Err(RepresentationError::InvalidRepresentationId { id: undeclared })
        );

        let stranger = NodeId::new(99);
        assert_eq!(
            plan.assign(&graph, stranger, dense_f32),
            Err(RepresentationError::UnknownAssignmentNode { node: stranger })
        );
    }

    #[test]
    fn quantized_assignments_bind_without_structural_match() {
        let mut graph = Graph::new();
        let weight = graph
            .add_input(
                "weight",
                TensorType::new(DType::F32, Shape::new(vec![8, 8])),
            )
            .unwrap();
        graph.set_outputs(vec![weight]).unwrap();

        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let dense_u8 = plan
            .declare(PrimitiveRepresentation::dense(DType::U8))
            .unwrap();
        let dense_f16 = plan
            .declare(PrimitiveRepresentation::dense(DType::F16))
            .unwrap();

        let codes = plan
            .component(TensorType::new(DType::U8, Shape::new(vec![8, 8])), dense_u8)
            .unwrap();
        let scales = plan
            .component(TensorType::new(DType::F16, Shape::new(vec![8])), dense_f16)
            .unwrap();
        let quantized = plan
            .declare(PrimitiveRepresentation::quantized(codes, scales))
            .unwrap();

        plan.assign(&graph, weight, quantized).unwrap();
        assert_eq!(plan.assignment(weight), Some(quantized));
        assert_eq!(
            plan.storage_bits(
                quantized,
                &TensorType::new(DType::F32, Shape::new(vec![8, 8]))
            ),
            Ok(StorageBits::new(8 * 8 * 8 + 8 * 2 * 8))
        );
    }
}
