//! The user-facing canonical façade: build a computation, prepare it once, run
//! it many times, in `TensorND` terms only.
//!
//! # What it hides
//!
//! Everything below it. A caller of this module never names a
//! `scirust_tensor_ir::Graph`, a `NodeId`, a `TensorType`, a `DType`, a
//! `ConstantId`, a `LogicalBindingId`, a `GraphInputs`, a `GraphConstants` or a
//! `PlanOutputs`, and never sees a backend buffer, stream or event:
//!
//! ```text
//! CanonicalProgram          build with opaque handles
//!   -> prepare(runtime)     compile, lower, prepare — once
//!   -> CanonicalSession
//!   -> CanonicalInputs      TensorND keyed by handle
//!   -> execute
//!   -> CanonicalOutputs     owned TensorND, in the declared output order
//! ```
//!
//! # Which tensor
//!
//! [`TensorND`] — `scirust_tensor_core`'s dense row-major `f32` tensor, the type
//! this crate's stack already shares. It carries no device and no dtype: it is
//! always host `f32`, which is exactly what the Reference path executes. The
//! eager 2-D `scirust_core::autodiff::reverse::Tensor` is untouched, unrelated,
//! and not usable here — it cannot express a scalar, a rank-3 value or a zero
//! dimension.
//!
//! Conversions are free in both directions: an input lends `&tensor.data[..]`,
//! and an output moves its `Vec<f32>` straight into a new `TensorND`. No byte
//! reinterpretation happens anywhere, and this crate contains no `unsafe`.
//!
//! # Operations
//!
//! `add`, `sub`, `mul`, `div`, `relu`, `scale`, `reshape`, `permute` — the eight
//! the Reference interpreter genuinely executes. `exp`, `log` and `matmul` are
//! **not exposed**: the runtime rejects the first two as non-reproducible and
//! the lowerer rejects the third, so offering them would advertise a capability
//! that does not exist.
//!
//! Binary operations require **identical shapes**. There is no broadcasting: the
//! canonical IR compares whole tensor types, and pretending otherwise would be a
//! silent lie about the semantics.
//!
//! # Determinism
//!
//! No `HashMap`, no clock, no random identifier, no global state. `ConstantId`s
//! are assigned from a program-local counter, in declaration order. Handles are
//! graph positions. Lookups are linear scans over ordered `Vec`s. The order in
//! which a caller binds inputs has no effect.

use scirust_compute::{ComputeBackend, DType, Shape};
use scirust_tensor_core::TensorND;
use scirust_tensor_ir::{ConstantId, Graph, GraphError, NodeId, Operation, Scalar, TensorType};

use core::fmt;

use crate::error::{CanonicalBuildError, CanonicalExecutionError, CanonicalPreparationError};
use crate::graph_session::{GraphConstants, GraphInputs, ReferenceGraphSession};
use crate::reference_plan::ReferencePlanRuntime;

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

/// An opaque value produced by a [`CanonicalProgram`].
///
/// Copyable, comparable, and meaningful only to the program that created it.
/// Its internal identity is private and is not part of this API.
///
/// # Handles from another program
///
/// A handle carries a position in its own program. Using one in a *different*
/// program is detected when that position does not exist there
/// ([`CanonicalBuildError::ForeignValue`]), which covers every handle from a
/// larger program and every handle built before the values it names. It cannot
/// be detected when the position happens to exist in both programs: preventing
/// that statically would need lifetime branding, and distinguishing programs at
/// run time would need a global counter — this crate has neither a
/// closure-shaped API nor global mutable state. Use one program's handles with
/// that program.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalValue {
    node: NodeId,
}

impl fmt::Debug for CanonicalValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CanonicalValue")
            .field(&self.node.get())
            .finish()
    }
}

/// An opaque input of a [`CanonicalProgram`], and the key an execution binds.
///
/// Usable directly wherever a [`CanonicalValue`] is expected. The same
/// cross-program caveat as [`CanonicalValue`] applies.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalInput {
    node: NodeId,
}

impl fmt::Debug for CanonicalInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CanonicalInput")
            .field(&self.node.get())
            .finish()
    }
}

impl From<CanonicalInput> for CanonicalValue {
    fn from(input: CanonicalInput) -> Self {
        Self { node: input.node }
    }
}

impl CanonicalInput {
    /// The input, seen as a value an operation can consume.
    pub const fn value(self) -> CanonicalValue {
        CanonicalValue { node: self.node }
    }
}

// ---------------------------------------------------------------------------
// Public metadata
// ---------------------------------------------------------------------------

/// One input a prepared session requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalInputSpec {
    pub input: CanonicalInput,
    pub name: String,
    pub shape: Vec<usize>,
}

/// One value a prepared session produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalOutputSpec {
    pub value: CanonicalValue,
    pub shape: Vec<usize>,
}

/// The input values one execution needs, addressed by handle.
///
/// Insertion order has no functional effect. A duplicate handle is accepted
/// while building and rejected by [`CanonicalSession::execute`], before any
/// backend call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CanonicalInputs<'a> {
    entries: Vec<(CanonicalInput, &'a TensorND)>,
}

impl<'a> CanonicalInputs<'a> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Supplies one input value. Chainable.
    pub fn bind(&mut self, input: CanonicalInput, tensor: &'a TensorND) -> &mut Self {
        self.entries.push((input, tensor));
        self
    }

    pub fn entries(&self) -> &[(CanonicalInput, &'a TensorND)] {
        &self.entries
    }
}

/// The values one execution produced, in the program's declared output order.
///
/// Owned outright: no borrow of any backend buffer survives here.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalOutputs {
    values: Vec<TensorND>,
}

impl CanonicalOutputs {
    pub fn values(&self) -> &[TensorND] {
        &self.values
    }

    pub fn into_values(self) -> Vec<TensorND> {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

/// One declared input, kept in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DeclaredInput {
    input: CanonicalInput,
    name: String,
}

/// A computation under construction.
///
/// Every operation validates its operands and computes its result shape before
/// touching the graph, so a program that builds is structurally sound. The
/// canonical compiler and the kernel lowerer remain the second line of
/// validation; this layer does not duplicate their algorithms.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CanonicalProgram {
    graph: Graph,
    /// Shape of every value, indexed by handle position.
    shapes: Vec<Vec<usize>>,
    /// Declared inputs, in declaration order.
    inputs: Vec<DeclaredInput>,
    /// Constant payloads; the position is the value of the `ConstantId`.
    constants: Vec<TensorND>,
    /// Declared outputs, in declaration order.
    outputs: Vec<CanonicalValue>,
}

impl CanonicalProgram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares an `f32` input of `shape`.
    ///
    /// Names are a diagnostic aid, never an execution key: the returned handle
    /// is what an execution binds. Duplicates are rejected so a diagnostic
    /// always names exactly one input.
    pub fn input(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
    ) -> Result<CanonicalInput, CanonicalBuildError> {
        let name = name.into();
        if self.inputs.iter().any(|declared| declared.name == name)
        {
            return Err(CanonicalBuildError::DuplicateInputName { name });
        }

        let shape = shape.to_vec();
        checked_elements(&shape).ok_or_else(|| CanonicalBuildError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        let node = self
            .graph
            .add_input(name.clone(), f32_type(&shape))
            .map_err(|source| CanonicalBuildError::GraphConstruction { source })?;

        let input = CanonicalInput { node };
        self.shapes.push(shape);
        self.inputs.push(DeclaredInput { input, name });
        Ok(input)
    }

    /// Adds a constant, taking ownership of its values.
    ///
    /// The tensor must be dense row-major — `TensorND`'s fields are public, so
    /// that invariant is checked rather than assumed. Bits are kept exactly as
    /// given: signed zeros, NaN payloads, infinities and subnormals all survive,
    /// because nothing here computes with them.
    pub fn constant(&mut self, tensor: TensorND) -> Result<CanonicalValue, CanonicalBuildError> {
        dense_layout(&tensor).map_err(|fault| match fault
        {
            LayoutFault::Overflow => CanonicalBuildError::ShapeOverflow {
                shape: tensor.shape.clone(),
            },
            LayoutFault::NotDense => CanonicalBuildError::NonContiguousTensor {
                shape: tensor.shape.clone(),
                elements: tensor.data.len(),
            },
        })?;

        // Every constant is also a graph node, and `Graph::add_node` caps the
        // node count at `u32::MAX`, so this conversion is always exact.
        let raw = u64::try_from(self.constants.len()).map_err(|_| {
            CanonicalBuildError::GraphConstruction {
                source: GraphError::TooManyNodes,
            }
        })?;

        let shape = tensor.shape.clone();
        let node = self
            .graph
            .add_constant(ConstantId::new(raw), f32_type(&shape))
            .map_err(|source| CanonicalBuildError::GraphConstruction { source })?;

        self.shapes.push(shape);
        self.constants.push(tensor);
        Ok(CanonicalValue { node })
    }

    /// Element-wise sum. Both operands must have the same shape.
    pub fn add(
        &mut self,
        lhs: impl Into<CanonicalValue>,
        rhs: impl Into<CanonicalValue>,
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        self.binary(Operation::Add, lhs.into(), rhs.into())
    }

    /// Element-wise difference. Both operands must have the same shape.
    pub fn sub(
        &mut self,
        lhs: impl Into<CanonicalValue>,
        rhs: impl Into<CanonicalValue>,
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        self.binary(Operation::Sub, lhs.into(), rhs.into())
    }

    /// Element-wise product. Both operands must have the same shape.
    pub fn mul(
        &mut self,
        lhs: impl Into<CanonicalValue>,
        rhs: impl Into<CanonicalValue>,
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        self.binary(Operation::Mul, lhs.into(), rhs.into())
    }

    /// Element-wise quotient. Both operands must have the same shape.
    pub fn div(
        &mut self,
        lhs: impl Into<CanonicalValue>,
        rhs: impl Into<CanonicalValue>,
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        self.binary(Operation::Div, lhs.into(), rhs.into())
    }

    /// Rectified linear unit, element-wise.
    pub fn relu(
        &mut self,
        value: impl Into<CanonicalValue>,
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        let value = value.into();
        let shape = self.shape_of(value)?.to_vec();
        self.push(Operation::Relu, vec![value.node], shape)
    }

    /// Multiplies every element by `factor`.
    ///
    /// The factor is part of the kernel signature and is stored bit-exactly, so
    /// every `f32` is representable — including signed zeros and NaN.
    pub fn scale(
        &mut self,
        value: impl Into<CanonicalValue>,
        factor: f32,
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        let value = value.into();
        let shape = self.shape_of(value)?.to_vec();
        self.push(
            Operation::Scale {
                factor: Scalar::f32(factor),
            },
            vec![value.node],
            shape,
        )
    }

    /// Reinterprets a value under a new shape of the same element count.
    ///
    /// This is a copy preserving the linear element order, not a view: the
    /// memory planner reserves a distinct buffer for the reshaped value.
    pub fn reshape(
        &mut self,
        value: impl Into<CanonicalValue>,
        shape: &[usize],
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        let value = value.into();
        let source = self.shape_of(value)?;
        let source_elements =
            checked_elements(source).ok_or_else(|| CanonicalBuildError::ShapeOverflow {
                shape: source.to_vec(),
            })?;

        let target = shape.to_vec();
        let target_elements =
            checked_elements(&target).ok_or_else(|| CanonicalBuildError::ShapeOverflow {
                shape: target.clone(),
            })?;

        if source_elements != target_elements
        {
            return Err(CanonicalBuildError::ElementCountMismatch {
                expected: source_elements,
                actual: target_elements,
            });
        }

        self.push(
            Operation::Reshape {
                shape: Shape::new(target.clone()),
            },
            vec![value.node],
            target,
        )
    }

    /// Permutes the axes of a value.
    ///
    /// `permutation` lists, in output-axis order, the input axis each output
    /// axis comes from — the NumPy convention. The result shape is therefore
    /// `result[i] == source[permutation[i]]`.
    pub fn permute(
        &mut self,
        value: impl Into<CanonicalValue>,
        permutation: &[usize],
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        let value = value.into();
        let source = self.shape_of(value)?.to_vec();
        let rank = source.len();

        if permutation.len() != rank
        {
            return Err(CanonicalBuildError::InvalidPermutation {
                permutation: permutation.to_vec(),
                rank,
            });
        }

        let mut axes = permutation.to_vec();
        axes.sort_unstable();
        if axes.iter().enumerate().any(|(axis, &value)| value != axis)
        {
            return Err(CanonicalBuildError::InvalidPermutation {
                permutation: permutation.to_vec(),
                rank,
            });
        }

        let mut target = Vec::with_capacity(rank);
        for &axis in permutation
        {
            // The permutation was just validated to cover `0..rank` exactly.
            let dimension = source.get(axis).copied().ok_or_else(|| {
                CanonicalBuildError::InvalidPermutation {
                    permutation: permutation.to_vec(),
                    rank,
                }
            })?;
            target.push(dimension);
        }

        self.push(
            Operation::Transpose {
                permutation: permutation.to_vec(),
            },
            vec![value.node],
            target,
        )
    }

    /// Shape of a value of this program.
    pub fn shape_of(
        &self,
        value: impl Into<CanonicalValue>,
    ) -> Result<&[usize], CanonicalBuildError> {
        let value = value.into();
        self.shapes
            .get(value.node.get() as usize)
            .map(Vec::as_slice)
            .ok_or(CanonicalBuildError::ForeignValue { value })
    }

    /// Inputs declared so far, in declaration order.
    ///
    /// Preparation may drop some of them: an input no output depends on is
    /// eliminated, and [`CanonicalSession::inputs`] is what an execution must
    /// satisfy.
    pub fn declared_inputs(&self) -> Vec<CanonicalInput> {
        self.inputs.iter().map(|declared| declared.input).collect()
    }

    /// Declares what the program produces.
    ///
    /// Order is preserved and duplicates are allowed: naming one value twice
    /// yields two outputs. Calling this again replaces the previous list.
    pub fn set_outputs(
        &mut self,
        outputs: impl IntoIterator<Item = CanonicalValue>,
    ) -> Result<(), CanonicalBuildError> {
        let outputs = outputs.into_iter().collect::<Vec<_>>();
        if outputs.is_empty()
        {
            return Err(CanonicalBuildError::NoOutputs);
        }

        for &value in &outputs
        {
            self.shape_of(value)?;
        }

        self.outputs = outputs;
        Ok(())
    }

    /// Compiles, lowers and prepares the program, exactly once.
    ///
    /// Consuming `self` is deliberate: it makes single preparation structural,
    /// and it lets the constants move straight through without a clone that only
    /// exists to satisfy ownership.
    pub fn prepare<B: ComputeBackend>(
        mut self,
        runtime: ReferencePlanRuntime<B>,
    ) -> Result<CanonicalSession<B>, CanonicalPreparationError> {
        if self.outputs.is_empty()
        {
            return Err(CanonicalPreparationError::NoOutputs);
        }

        let declared_outputs = self.outputs.clone();
        let output_nodes = declared_outputs
            .iter()
            .map(|value| value.node)
            .collect::<Vec<_>>();

        self.graph
            .set_outputs(output_nodes)
            .map_err(|source| CanonicalPreparationError::GraphOutputs { source })?;

        // The payloads stay owned by this program until the session has cloned
        // whatever survives elimination.
        let mut constants = GraphConstants::new();
        for (index, tensor) in self.constants.iter().enumerate()
        {
            let raw =
                u64::try_from(index).map_err(|_| CanonicalPreparationError::GraphOutputs {
                    source: GraphError::TooManyNodes,
                })?;
            constants.bind(ConstantId::new(raw), &tensor.data);
        }

        let session = ReferenceGraphSession::prepare(runtime, &self.graph, &constants)
            .map_err(|source| CanonicalPreparationError::GraphSessionPreparation { source })?;

        // Only the inputs that survived elimination are required, and they are
        // reported in the session's own binding order.
        let mut inputs = Vec::with_capacity(session.inputs().len());
        for specification in session.inputs()
        {
            let input = CanonicalInput {
                node: specification.node,
            };
            let declared = self
                .inputs
                .iter()
                .find(|declared| declared.input == input)
                .ok_or(CanonicalPreparationError::UnknownSessionInput { input })?;

            inputs.push(CanonicalInputSpec {
                input,
                name: declared.name.clone(),
                shape: specification.tensor_type.shape.dims().to_vec(),
            });
        }

        if session.outputs().len() != declared_outputs.len()
        {
            return Err(CanonicalPreparationError::OutputCountMismatch {
                expected: declared_outputs.len(),
                actual: session.outputs().len(),
            });
        }

        let outputs = session
            .outputs()
            .iter()
            .zip(declared_outputs)
            .map(|(specification, value)| CanonicalOutputSpec {
                value,
                shape: specification.tensor_type.shape.dims().to_vec(),
            })
            .collect();

        Ok(CanonicalSession {
            session,
            inputs,
            outputs,
        })
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn binary(
        &mut self,
        operation: Operation,
        lhs: CanonicalValue,
        rhs: CanonicalValue,
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        let left = self.shape_of(lhs)?.to_vec();
        let right = self.shape_of(rhs)?;

        // The canonical IR compares whole tensor types; there is no broadcasting
        // to fall back on, so unequal shapes are an error, not a hint.
        if left != right
        {
            return Err(CanonicalBuildError::ShapeMismatch {
                expected: left,
                actual: right.to_vec(),
            });
        }

        self.push(operation, vec![lhs.node, rhs.node], left)
    }

    fn push(
        &mut self,
        operation: Operation,
        inputs: Vec<NodeId>,
        shape: Vec<usize>,
    ) -> Result<CanonicalValue, CanonicalBuildError> {
        let node = self
            .graph
            .add_node(operation, inputs, f32_type(&shape))
            .map_err(|source| CanonicalBuildError::GraphConstruction { source })?;

        self.shapes.push(shape);
        Ok(CanonicalValue { node })
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A prepared program, ready to run any number of times.
///
/// Immutable: `execute` takes `&self`, and the session holds no buffer, stream,
/// event, previous input, previous output, counter or cache. A failed execution
/// leaves it immediately reusable. `Send` and `Sync` are never imposed; they
/// follow from `B` and `B::Kernel`.
pub struct CanonicalSession<B: ComputeBackend> {
    session: ReferenceGraphSession<B>,
    inputs: Vec<CanonicalInputSpec>,
    outputs: Vec<CanonicalOutputSpec>,
}

/// Written by hand rather than derived so it needs no bound on the backend or
/// its kernel type, and so it shows the façade's own vocabulary instead of the
/// compiled artefacts underneath.
impl<B: ComputeBackend> fmt::Debug for CanonicalSession<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CanonicalSession")
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish_non_exhaustive()
    }
}

impl<B: ComputeBackend> CanonicalSession<B> {
    /// Inputs this session requires.
    ///
    /// An input the program declared but preparation eliminated is absent:
    /// supplying it is an error, not a no-op.
    pub fn inputs(&self) -> &[CanonicalInputSpec] {
        &self.inputs
    }

    /// Outputs this session produces, in the program's declared order.
    pub fn outputs(&self) -> &[CanonicalOutputSpec] {
        &self.outputs
    }

    /// Runs the session over `inputs`.
    ///
    /// Every input is validated — required, unique, complete, exactly shaped,
    /// dense — before a single backend call is issued, so a rejected set costs
    /// nothing. Constants are supplied by the session itself and are never asked
    /// for here.
    pub fn execute(
        &self,
        inputs: &CanonicalInputs<'_>,
    ) -> Result<CanonicalOutputs, CanonicalExecutionError> {
        let mut slots: Vec<Option<&TensorND>> = vec![None; self.inputs.len()];

        for &(input, tensor) in inputs.entries()
        {
            let position = self
                .inputs
                .iter()
                .position(|specification| specification.input == input)
                .ok_or(CanonicalExecutionError::UnexpectedInput { input })?;

            match slots.get_mut(position)
            {
                Some(slot) =>
                {
                    if slot.is_some()
                    {
                        return Err(CanonicalExecutionError::DuplicateInput { input });
                    }
                    *slot = Some(tensor);
                },
                // Defensive: `position` indexes the vector `slots` was sized
                // from.
                None => return Err(CanonicalExecutionError::UnexpectedInput { input }),
            }
        }

        let mut bound = GraphInputs::new();
        for (specification, slot) in self.inputs.iter().zip(slots)
        {
            let tensor = slot.ok_or_else(|| CanonicalExecutionError::MissingInput {
                input: specification.input,
                name: specification.name.clone(),
            })?;

            // A stronger check than the element count the layer below enforces:
            // `[2, 3]` and `[6]` hold the same number of values and are not the
            // same input.
            if tensor.shape != specification.shape
            {
                return Err(CanonicalExecutionError::InputShapeMismatch {
                    input: specification.input,
                    name: specification.name.clone(),
                    expected: specification.shape.clone(),
                    actual: tensor.shape.clone(),
                });
            }

            dense_layout(tensor).map_err(|_| CanonicalExecutionError::NonContiguousInput {
                input: specification.input,
                name: specification.name.clone(),
                shape: tensor.shape.clone(),
                elements: tensor.data.len(),
            })?;

            bound.bind(specification.input.node, &tensor.data);
        }

        let produced = self
            .session
            .execute(&bound)
            .map_err(|source| CanonicalExecutionError::GraphSessionExecution { source })?;

        let mut values = Vec::with_capacity(produced.len());
        for value in produced.into_values()
        {
            let shape = value.tensor_type.shape.dims().to_vec();
            let elements = value.values.len();

            // Moves the `Vec<f32>`; no copy, and no byte reinterpretation.
            let tensor = TensorND::try_new(value.values, shape.clone())
                .map_err(|_| CanonicalExecutionError::OutputConstruction { shape, elements })?;
            values.push(tensor);
        }

        Ok(CanonicalOutputs { values })
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The canonical tensor type of every value this façade builds.
///
/// The dtype is fixed here, once: it never reaches the caller's vocabulary.
fn f32_type(shape: &[usize]) -> TensorType {
    TensorType::new(DType::F32, Shape::new(shape.to_vec()))
}

/// Number of elements of a shape, or `None` on overflow.
fn checked_elements(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
}

/// Row-major strides of a shape, or `None` on overflow.
///
/// `scirust_tensor_core` computes these privately; recomputing them here is what
/// lets this layer verify a `TensorND` without modifying that crate.
fn row_major_strides(shape: &[usize]) -> Option<Vec<usize>> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];

    if rank <= 1
    {
        return Some(strides);
    }

    for axis in (0..rank - 1).rev()
    {
        let next = strides.get(axis + 1).copied()?;
        let dimension = shape.get(axis + 1).copied()?;
        let stride = next.checked_mul(dimension)?;
        *strides.get_mut(axis)? = stride;
    }

    Some(strides)
}

/// Why a `TensorND` is not a dense row-major value.
enum LayoutFault {
    Overflow,
    NotDense,
}

/// Checks the invariant `TensorND` documents but cannot enforce: its three
/// fields are public, so a value can be handed over with a data length or a
/// stride vector that contradicts its shape.
fn dense_layout(tensor: &TensorND) -> Result<(), LayoutFault> {
    let elements = checked_elements(&tensor.shape).ok_or(LayoutFault::Overflow)?;
    let strides = row_major_strides(&tensor.shape).ok_or(LayoutFault::Overflow)?;

    if tensor.data.len() != elements || tensor.strides != strides
    {
        return Err(LayoutFault::NotDense);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Unit tests of the façade, driven by a recording backend.
    //!
    //! The real CPU path is covered end to end in
    //! `tests/canonical_program.rs`. What a real backend cannot show is *when*
    //! the façade touches it: that building and rejecting cost nothing, and that
    //! repeated executions compile nothing new.

    use super::*;

    use core::error::Error as _;
    use std::rc::Rc;

    use scirust_compute::{
        BufferBinding, ComputeResult, DeviceCapabilities, KernelModule, LaunchConfig, MemorySpace,
    };

    use crate::reference_plan::test_backend::{Call, RecordingBackend, RecordingBuffer};

    // -----------------------------------------------------------------------
    // A recording backend the test keeps a handle on
    // -----------------------------------------------------------------------

    /// `CanonicalProgram::prepare` takes the runtime by value and the session
    /// never hands it back, so a shared recorder is the only way to observe the
    /// backend once a program has been prepared — or failed to prepare.
    ///
    /// `graph_session`'s tests define an equivalent fixture; duplicating ~80
    /// lines of delegation here keeps this phase inside its authorised file set
    /// instead of reshaping `reference_plan.rs` for a test's convenience.
    #[derive(Debug, Clone)]
    struct SharedBackend(Rc<RecordingBackend>);

    impl SharedBackend {
        fn new() -> Self {
            Self(Rc::new(RecordingBackend::new()))
        }

        fn failing_at(dispatch_index: usize) -> Self {
            Self(Rc::new(
                RecordingBackend::new().fail_launch_at(dispatch_index),
            ))
        }

        fn clear_launch_failure(&self) {
            self.0.clear_launch_failure();
        }

        fn calls(&self) -> Vec<Call> {
            self.0.calls()
        }

        fn count<F: Fn(&Call) -> bool>(&self, predicate: F) -> usize {
            self.0.count(predicate)
        }
    }

    impl ComputeBackend for SharedBackend {
        type Buffer = RecordingBuffer;
        type Kernel = String;
        type Stream = ();
        type Event = ();

        fn capabilities(&self) -> &DeviceCapabilities {
            self.0.capabilities()
        }

        fn allocate(
            &self,
            bytes: usize,
            alignment: usize,
            memory_space: MemorySpace,
        ) -> ComputeResult<Self::Buffer> {
            self.0.allocate(bytes, alignment, memory_space)
        }

        fn write(
            &self,
            destination: &Self::Buffer,
            offset_bytes: usize,
            data: &[u8],
        ) -> ComputeResult<()> {
            self.0.write(destination, offset_bytes, data)
        }

        fn read(
            &self,
            source: &Self::Buffer,
            offset_bytes: usize,
            destination: &mut [u8],
        ) -> ComputeResult<()> {
            self.0.read(source, offset_bytes, destination)
        }

        fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
            self.0.compile(module)
        }

        fn create_stream(&self) -> ComputeResult<Self::Stream> {
            self.0.create_stream()
        }

        fn launch(
            &self,
            kernel: &Self::Kernel,
            stream: &Self::Stream,
            config: LaunchConfig,
            bindings: &[BufferBinding<'_, Self::Buffer>],
        ) -> ComputeResult<Self::Event> {
            self.0.launch(kernel, stream, config, bindings)
        }

        fn wait(&self, event: &Self::Event) -> ComputeResult<()> {
            self.0.wait(event)
        }

        fn synchronize(&self, stream: &Self::Stream) -> ComputeResult<()> {
            self.0.synchronize(stream)
        }
    }

    fn prepare(
        backend: &SharedBackend,
        program: CanonicalProgram,
    ) -> Result<CanonicalSession<SharedBackend>, CanonicalPreparationError> {
        program.prepare(ReferencePlanRuntime::new(backend.clone()))
    }

    /// `x -> relu -> relu -> output`: two dispatches sharing one kernel.
    fn shared_kernel_program() -> (CanonicalProgram, CanonicalInput) {
        let mut program = CanonicalProgram::new();
        let x = program.input("x", &[2]).expect("input");
        let first = program.relu(x).expect("relu");
        let second = program.relu(first).expect("relu");
        program.set_outputs([second]).expect("outputs");
        (program, x)
    }

    // -----------------------------------------------------------------------
    // Building
    // -----------------------------------------------------------------------

    #[test]
    fn an_empty_program_cannot_be_prepared() {
        let backend = SharedBackend::new();
        assert_eq!(
            prepare(&backend, CanonicalProgram::new()).err(),
            Some(CanonicalPreparationError::NoOutputs)
        );
        assert!(backend.calls().is_empty());
    }

    #[test]
    fn an_empty_output_list_is_rejected() {
        let mut program = CanonicalProgram::new();
        assert_eq!(
            program.set_outputs([]).err(),
            Some(CanonicalBuildError::NoOutputs)
        );
    }

    #[test]
    fn a_duplicate_input_name_is_rejected() {
        let mut program = CanonicalProgram::new();
        program.input("x", &[2]).expect("input");
        assert_eq!(
            program.input("x", &[4]).err(),
            Some(CanonicalBuildError::DuplicateInputName {
                name: "x".to_string(),
            })
        );
    }

    #[test]
    fn a_handle_from_a_larger_program_is_rejected() {
        let mut donor = CanonicalProgram::new();
        donor.input("a", &[2]).expect("input");
        donor.input("b", &[2]).expect("input");
        let stranger = donor.input("c", &[2]).expect("input").value();

        let mut program = CanonicalProgram::new();
        program.input("x", &[2]).expect("input");

        assert_eq!(
            program.relu(stranger).err(),
            Some(CanonicalBuildError::ForeignValue { value: stranger })
        );
        assert_eq!(
            program.set_outputs([stranger]).err(),
            Some(CanonicalBuildError::ForeignValue { value: stranger })
        );
    }

    #[test]
    fn binary_operands_must_have_the_same_shape() {
        let mut program = CanonicalProgram::new();
        let a = program.input("a", &[2, 3]).expect("input");
        let b = program.input("b", &[6]).expect("input");

        assert_eq!(
            program.add(a, b).err(),
            Some(CanonicalBuildError::ShapeMismatch {
                expected: vec![2, 3],
                actual: vec![6],
            })
        );
    }

    #[test]
    fn a_reshape_must_preserve_the_element_count() {
        let mut program = CanonicalProgram::new();
        let x = program.input("x", &[2, 3]).expect("input");

        assert_eq!(
            program.reshape(x, &[4]).err(),
            Some(CanonicalBuildError::ElementCountMismatch {
                expected: 6,
                actual: 4,
            })
        );
        let flat = program.reshape(x, &[6]).expect("same count");
        assert_eq!(program.shape_of(flat).expect("known"), &[6]);
    }

    #[test]
    fn a_permutation_must_cover_every_axis_exactly_once() {
        let mut program = CanonicalProgram::new();
        let x = program.input("x", &[2, 3]).expect("input");

        for permutation in [vec![0], vec![0, 0], vec![0, 2], vec![1, 0, 1]]
        {
            assert_eq!(
                program.permute(x, &permutation).err(),
                Some(CanonicalBuildError::InvalidPermutation {
                    permutation: permutation.clone(),
                    rank: 2,
                }),
                "{permutation:?} must be rejected"
            );
        }

        let transposed = program.permute(x, &[1, 0]).expect("valid permutation");
        assert_eq!(program.shape_of(transposed).expect("known"), &[3, 2]);
    }

    #[test]
    fn a_non_dense_constant_is_rejected() {
        let mut program = CanonicalProgram::new();

        let mut wrong_strides = TensorND::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        wrong_strides.strides[0] = 1;
        assert_eq!(
            program.constant(wrong_strides).err(),
            Some(CanonicalBuildError::NonContiguousTensor {
                shape: vec![2, 2],
                elements: 4,
            })
        );

        let mut wrong_length = TensorND::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        wrong_length.data.push(5.0);
        assert_eq!(
            program.constant(wrong_length).err(),
            Some(CanonicalBuildError::NonContiguousTensor {
                shape: vec![2, 2],
                elements: 5,
            })
        );
    }

    #[test]
    fn a_shape_whose_element_count_overflows_is_rejected() {
        let mut program = CanonicalProgram::new();
        assert_eq!(
            program.input("x", &[usize::MAX, 2]).err(),
            Some(CanonicalBuildError::ShapeOverflow {
                shape: vec![usize::MAX, 2],
            })
        );
    }

    // -----------------------------------------------------------------------
    // Preparation
    // -----------------------------------------------------------------------

    #[test]
    fn preparation_compiles_each_logical_kernel_once_and_nothing_else() {
        let backend = SharedBackend::new();
        let (program, _) = shared_kernel_program();
        let session = prepare(&backend, program).expect("preparable");

        assert_eq!(session.inputs().len(), 1);
        assert_eq!(session.outputs().len(), 1);
        assert_eq!(
            backend.calls(),
            vec![Call::Compile {
                entry_point: "scirust_reference_kernel_0".to_string(),
            }],
            "two dispatches share one kernel, and preparation does nothing else"
        );
    }

    #[test]
    fn repeated_executions_compile_nothing_new_and_keep_the_metadata_stable() {
        let backend = SharedBackend::new();
        let (program, x) = shared_kernel_program();
        let session = prepare(&backend, program).expect("preparable");

        let before_inputs = session.inputs().to_vec();
        let before_outputs = session.outputs().to_vec();

        for values in [[-1.0f32, 2.0], [3.0, -4.0], [0.0, 0.0]]
        {
            let tensor = TensorND::new(values.to_vec(), vec![2]);
            let mut inputs = CanonicalInputs::new();
            inputs.bind(x, &tensor);
            session.execute(&inputs).expect("runs");
        }

        assert_eq!(
            backend.count(|call| matches!(call, Call::Compile { .. })),
            1,
            "three executions must not recompile anything"
        );
        assert_eq!(session.inputs(), before_inputs.as_slice());
        assert_eq!(session.outputs(), before_outputs.as_slice());
    }

    #[test]
    fn preparing_the_same_program_twice_yields_the_same_public_metadata() {
        let describe = || {
            let backend = SharedBackend::new();
            let (program, _) = shared_kernel_program();
            let session = prepare(&backend, program).expect("preparable");
            (session.inputs().to_vec(), session.outputs().to_vec())
        };

        assert_eq!(describe(), describe());
    }

    #[test]
    fn an_eliminated_input_is_not_required_and_an_eliminated_constant_is_not_kept() {
        let backend = SharedBackend::new();
        let mut program = CanonicalProgram::new();

        let live = program.input("live", &[2]).expect("input");
        let dead = program.input("dead", &[2]).expect("input");
        let dead_constant = program
            .constant(TensorND::new(vec![7.0, 9.0], vec![2]))
            .expect("constant");

        let kept = program.relu(live).expect("relu");
        program.relu(dead).expect("dead relu");
        program.relu(dead_constant).expect("dead relu");
        program.set_outputs([kept]).expect("outputs");

        let session = prepare(&backend, program).expect("preparable");

        assert_eq!(session.inputs().len(), 1);
        assert_eq!(session.inputs()[0].input, live);
        assert_eq!(session.inputs()[0].name, "live");

        let tensor = TensorND::new(vec![-1.0, 2.0], vec![2]);
        let mut inputs = CanonicalInputs::new();
        inputs.bind(dead, &tensor);
        assert_eq!(
            session.execute(&inputs).err(),
            Some(CanonicalExecutionError::UnexpectedInput { input: dead })
        );
    }

    #[test]
    fn constants_are_injected_automatically_and_never_asked_of_the_caller() {
        let backend = SharedBackend::new();
        let mut program = CanonicalProgram::new();

        let x = program.input("x", &[2]).expect("input");
        let bias = program
            .constant(TensorND::new(vec![10.0, 20.0], vec![2]))
            .expect("constant");
        let sum = program.add(x, bias).expect("add");
        program.set_outputs([sum]).expect("outputs");

        let session = prepare(&backend, program).expect("preparable");
        assert_eq!(session.inputs().len(), 1, "only the input is required");

        let tensor = TensorND::new(vec![1.0, 2.0], vec![2]);
        for _ in 0..2
        {
            let mut inputs = CanonicalInputs::new();
            inputs.bind(x, &tensor);
            session.execute(&inputs).expect("runs");
        }

        assert_eq!(
            backend.count(|call| matches!(call, Call::Write { bytes: 8 })),
            4,
            "each run imports the input and the constant"
        );
    }

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    #[test]
    fn an_invalid_input_costs_no_backend_call() {
        let backend = SharedBackend::new();
        let (program, x) = shared_kernel_program();
        let session = prepare(&backend, program).expect("preparable");

        let good = TensorND::new(vec![1.0, 2.0], vec![2]);
        let wrong_shape = TensorND::new(vec![1.0, 2.0], vec![1, 2]);
        let mut stray = TensorND::new(vec![1.0, 2.0], vec![2]);
        stray.strides[0] = 7;

        let mut donor = CanonicalProgram::new();
        donor.input("a", &[2]).expect("input");
        donor.input("b", &[2]).expect("input");
        let stranger = donor.input("c", &[2]).expect("input");

        assert_eq!(
            session.execute(&CanonicalInputs::new()).err(),
            Some(CanonicalExecutionError::MissingInput {
                input: x,
                name: "x".to_string(),
            })
        );

        let mut extra = CanonicalInputs::new();
        extra.bind(x, &good).bind(stranger, &good);
        assert_eq!(
            session.execute(&extra).err(),
            Some(CanonicalExecutionError::UnexpectedInput { input: stranger })
        );

        let mut duplicate = CanonicalInputs::new();
        duplicate.bind(x, &good).bind(x, &good);
        assert_eq!(
            session.execute(&duplicate).err(),
            Some(CanonicalExecutionError::DuplicateInput { input: x })
        );

        let mut reshaped = CanonicalInputs::new();
        reshaped.bind(x, &wrong_shape);
        assert_eq!(
            session.execute(&reshaped).err(),
            Some(CanonicalExecutionError::InputShapeMismatch {
                input: x,
                name: "x".to_string(),
                expected: vec![2],
                actual: vec![1, 2],
            })
        );

        let mut sparse = CanonicalInputs::new();
        sparse.bind(x, &stray);
        assert_eq!(
            session.execute(&sparse).err(),
            Some(CanonicalExecutionError::NonContiguousInput {
                input: x,
                name: "x".to_string(),
                shape: vec![2],
                elements: 2,
            })
        );

        assert_eq!(
            backend.count(|call| matches!(call, Call::Allocate { .. })),
            0,
            "five rejections, zero allocation"
        );

        // Five rejections later, the session still runs.
        let mut valid = CanonicalInputs::new();
        valid.bind(x, &good);
        session.execute(&valid).expect("valid run");
    }

    #[test]
    fn the_order_inputs_are_bound_in_has_no_effect() {
        let backend = SharedBackend::new();
        let mut program = CanonicalProgram::new();
        let a = program.input("a", &[2]).expect("input");
        let b = program.input("b", &[2]).expect("input");
        let difference = program.sub(a, b).expect("sub");
        program.set_outputs([difference]).expect("outputs");

        let session = prepare(&backend, program).expect("preparable");

        let left = TensorND::new(vec![5.0, 6.0], vec![2]);
        let right = TensorND::new(vec![1.0, 2.0], vec![2]);

        let mut forward = CanonicalInputs::new();
        forward.bind(a, &left).bind(b, &right);
        let mut backward = CanonicalInputs::new();
        backward.bind(b, &right).bind(a, &left);

        assert_eq!(
            session.execute(&forward).expect("runs"),
            session.execute(&backward).expect("runs")
        );
    }

    #[test]
    fn a_session_survives_a_backend_failure() {
        let backend = SharedBackend::failing_at(0);
        let (program, x) = shared_kernel_program();
        let session = prepare(&backend, program).expect("preparable");

        let before_inputs = session.inputs().to_vec();
        let before_outputs = session.outputs().to_vec();

        let tensor = TensorND::new(vec![1.0, 2.0], vec![2]);
        let mut inputs = CanonicalInputs::new();
        inputs.bind(x, &tensor);

        let error = session.execute(&inputs).expect_err("injected failure");
        assert!(
            matches!(error, CanonicalExecutionError::GraphSessionExecution { .. }),
            "the session error must be kept whole, got {error:?}"
        );
        assert!(error.source().is_some(), "the source must stay reachable");

        assert_eq!(session.inputs(), before_inputs.as_slice());
        assert_eq!(session.outputs(), before_outputs.as_slice());
        assert_eq!(
            tensor.data,
            vec![1.0, 2.0],
            "caller values are never mutated"
        );

        // The recording backend computes nothing, so only the shape of the run
        // is asserted here; numeric behaviour belongs to the real CPU tests.
        backend.clear_launch_failure();
        let outputs = session
            .execute(&inputs)
            .expect("the session is still usable");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs.values()[0].shape, vec![2]);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    #[test]
    fn row_major_strides_match_the_tensor_crate() {
        for shape in [vec![], vec![5], vec![2, 3], vec![2, 3, 4], vec![0, 3]]
        {
            let elements = checked_elements(&shape).expect("no overflow");
            let reference = TensorND::new(vec![0.0; elements], shape.clone());
            assert_eq!(
                row_major_strides(&shape),
                Some(reference.strides.clone()),
                "strides for {shape:?}"
            );
        }
    }

    #[test]
    fn dense_layout_accepts_a_canonical_tensor() {
        let tensor = TensorND::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert!(dense_layout(&tensor).is_ok());
    }

    /// The hand-written `Debug` needs no bound on the backend or its kernel, and
    /// shows the façade's own vocabulary rather than the artefacts underneath.
    #[test]
    fn session_debug_shows_only_the_facade_metadata() {
        let backend = SharedBackend::new();
        let (program, _) = shared_kernel_program();
        let session = prepare(&backend, program).expect("preparable");

        let rendered = format!("{session:?}");
        assert!(rendered.starts_with("CanonicalSession"));
        assert!(rendered.contains("CanonicalInput(0)"));
        assert!(
            !rendered.contains("scirust_reference_kernel"),
            "compiled kernels must not leak into the facade's Debug output"
        );
    }
}
