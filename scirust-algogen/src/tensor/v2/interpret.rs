//! Safe, deterministic interpretation of verified V2 research programs.
//!
//! The interpreter implements the operator semantics documented in
//! `docs/SCIRUST_ALGOGEN_IR_V2_NUMERICAL_SEMANTICS.md`:
//!
//! * every operation computes **in its declared dtype** (an `f32` op never
//!   computes in `f64`);
//! * values are carried in a lossless uniform buffer (`f64` stores every
//!   `f32` bit pattern exactly; the carrier is storage, never a compute
//!   dtype);
//! * accumulation orders are fixed by construction (ascending row-major flat
//!   index), so results are bit-reproducible;
//! * extrema (`Min`, `Max`, `Clamp`, `ReduceMin`, `ReduceMax`) use explicit
//!   deterministic kernels (see `deterministic_min_f32` and siblings below)
//!   whose signed zero and NaN rules are defined by this contract, never by
//!   unspecified native-min/max platform behaviour;
//! * under the default [`FloatPolicy::FiniteOutputs`], NaN intermediates and
//!   non-finite observable outputs abort with precise errors while explicit
//!   infinity identities may flow internally;
//! * no FFI, no threads, no panics on hostile input: malformed programs are
//!   rejected up front, malformed inputs produce structured errors.

use serde::{Deserialize, Serialize};

use super::ir::{Op, Ref, ResearchProgram, Section, ValueId};
use super::types::{DType, ScalarValue, ValueType, shape_elements};
use super::verify::{
    ProgramError, SectionKind, VerificationLimits, VerifiedProgram, verify_program,
};

/// Policy governing floating-point results during execution.
///
/// The default discovery regime implements the three-level contract specified
/// in `SCIRUST_ALGOGEN_IR_V2_NUMERICAL_SEMANTICS.md`:
///
/// 1. any **NaN** intermediate aborts evaluation immediately
///    ([`ExecutionError::NanResult`]);
/// 2. `±Infinity` may flow as an explicit identity (e.g. `exp(-Infinity)`,
///    `-Infinity - x`) because stable streaming algorithms require it;
/// 3. **observable outputs must be entirely finite**
///    ([`ExecutionError::NonFiniteOutput`] otherwise), so fitness comparisons
///    can never see non-finite values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FloatPolicy {
    /// Default discovery regime (see the contract above).
    FiniteOutputs,
    /// Reject every non-finite input, item, intermediate, and output. This is
    /// the execution policy matching [`super::semantics::NumericalSemantics::FiniteOnly`].
    RejectNonFinite,
    /// Research escape hatch: no checks at all; non-finite values may reach
    /// outputs. Never used by the default evaluator.
    AllowNonFinite,
}

/// Execution-time policy knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    pub floats: FloatPolicy,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            floats: FloatPolicy::FiniteOutputs,
        }
    }
}

/// A dense row-major tensor with an explicit dtype.
///
/// Numeric payloads are stored in a lossless `f64` carrier (`f32` values
/// round-trip through `f64` exactly). The carrier is storage, not compute:
/// arithmetic always happens in [`Self::dtype`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// Structural payload failure for an externally constructed tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDataError {
    ShapeOverflow {
        shape: Vec<usize>,
    },
    LengthMismatch {
        shape: Vec<usize>,
        expected: usize,
        found: usize,
    },
    F32NotRepresentable {
        element: usize,
    },
    InvalidBoolEncoding {
        element: usize,
        bits: u64,
    },
    DTypeMismatch {
        expected: DType,
        found: DType,
    },
}

impl std::fmt::Display for TensorDataError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::ShapeOverflow { shape } => write!(formatter, "shape {shape:?} overflows usize"),
            Self::LengthMismatch {
                shape,
                expected,
                found,
            } => write!(
                formatter,
                "data length {found} does not match shape {shape:?} ({expected} elements)"
            ),
            Self::F32NotRepresentable { element } => write!(
                formatter,
                "element {element} is not exactly representable as f32"
            ),
            Self::InvalidBoolEncoding { element, bits } => write!(
                formatter,
                "Boolean element {element} has invalid f64 bit pattern {bits:#018x}"
            ),
            Self::DTypeMismatch { expected, found } => write!(
                formatter,
                "conversion requires dtype {}, found {}",
                expected.name(),
                found.name()
            ),
        }
    }
}

impl std::error::Error for TensorDataError {}

impl ValueTensor {
    /// Build a tensor, rejecting data-length mismatches and, for `f32`,
    /// values that are not exactly representable in binary32.
    pub fn new(dtype: DType, shape: Vec<usize>, data: Vec<f64>) -> Result<Self, TensorDataError> {
        let tensor = Self { dtype, shape, data };
        tensor.validate_layout()?;
        Ok(tensor)
    }

    /// Rank-0 scalar from an `f32`.
    pub fn scalar_f32(value: f32) -> Self {
        Self::from_parts(DType::F32, Vec::new(), vec![value as f64])
    }

    /// Rank-0 scalar from an `f64`.
    pub fn scalar_f64(value: f64) -> Self {
        Self::from_parts(DType::F64, Vec::new(), vec![value])
    }

    /// Rank-0 Boolean scalar.
    pub fn scalar_bool(value: bool) -> Self {
        Self::from_parts(DType::Bool, Vec::new(), vec![f64::from(u8::from(value))])
    }

    /// Unchecked constructor for internal call sites that already guarantee
    /// consistency (every interpreter result matches its verified type).
    pub(crate) fn from_parts(dtype: DType, shape: Vec<usize>, data: Vec<f64>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape_elements(&shape).unwrap_or(usize::MAX),
            "internal tensor construction violated shape/data consistency"
        );
        Self { dtype, shape, data }
    }

    /// Element count.
    pub fn elements(&self) -> usize {
        self.data.len()
    }

    /// The declared type of this tensor.
    pub fn value_type(&self) -> ValueType {
        ValueType::new(self.dtype, self.shape.clone())
    }

    /// Revalidate public fields at the trust boundary. This is intentionally
    /// called for every external tensor because serde and struct literals can
    /// bypass [`Self::new`].
    pub fn validate_layout(&self) -> Result<(), TensorDataError> {
        let expected =
            shape_elements(&self.shape).ok_or_else(|| TensorDataError::ShapeOverflow {
                shape: self.shape.clone(),
            })?;
        if self.data.len() != expected
        {
            return Err(TensorDataError::LengthMismatch {
                shape: self.shape.clone(),
                expected,
                found: self.data.len(),
            });
        }
        match self.dtype
        {
            DType::F32 =>
            {
                for (element, &value) in self.data.iter().enumerate()
                {
                    let round_trip = (value as f32) as f64;
                    let exact = if value.is_nan()
                    {
                        round_trip.is_nan()
                    }
                    else
                    {
                        round_trip.to_bits() == value.to_bits()
                    };
                    if !exact
                    {
                        return Err(TensorDataError::F32NotRepresentable { element });
                    }
                }
            },
            DType::Bool =>
            {
                for (element, &value) in self.data.iter().enumerate()
                {
                    if value.to_bits() != 0.0f64.to_bits() && value.to_bits() != 1.0f64.to_bits()
                    {
                        return Err(TensorDataError::InvalidBoolEncoding {
                            element,
                            bits: value.to_bits(),
                        });
                    }
                }
            },
            DType::F64 =>
            {},
        }
        Ok(())
    }

    /// Compatibility-boundary conversion to a plain `f32` tensor.
    ///
    /// Returns a structured error instead of panicking when this tensor is
    /// not an `f32` tensor (external trust boundary).
    pub fn to_f32_tensor(&self) -> Result<scirust_tensor_core::TensorND, TensorDataError> {
        if self.dtype != DType::F32
        {
            return Err(TensorDataError::DTypeMismatch {
                expected: DType::F32,
                found: self.dtype,
            });
        }
        Ok(scirust_tensor_core::TensorND::new(
            self.data.iter().map(|&value| value as f32).collect(),
            self.shape.clone(),
        ))
    }

    /// Compatibility-boundary conversion from a plain `f32` tensor.
    pub fn from_f32(tensor: &scirust_tensor_core::TensorND) -> Self {
        Self::from_parts(
            DType::F32,
            tensor.shape.clone(),
            tensor.data.iter().map(|&value| value as f64).collect(),
        )
    }
}

/// Successful execution of a research program.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionResult {
    /// Observable outputs in program order.
    pub outputs: Vec<ValueTensor>,

    /// Number of active nodes actually evaluated (init once, then step nodes
    /// times `steps`, then finalize once).
    pub executed_nodes: u64,

    /// Static verification information used by the interpreter.
    pub verified: VerifiedProgram,
}

/// Recoverable execution failure.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionError {
    Verification(ProgramError),

    InputArity {
        expected: usize,
        found: usize,
    },
    ItemArity {
        expected: usize,
        found: usize,
    },
    ItemCountOverflow {
        steps: u32,
        items_per_step: usize,
    },
    InputTypeMismatch {
        input: usize,
        expected: ValueType,
        found: ValueType,
    },
    ItemTypeMismatch {
        item: usize,
        expected: ValueType,
        found: ValueType,
    },
    InvalidInputLayout {
        input: usize,
        error: TensorDataError,
    },
    InvalidItemLayout {
        item: usize,
        error: TensorDataError,
    },
    NonFiniteInput {
        input: usize,
        element: usize,
    },
    NonFiniteItem {
        item: usize,
        element: usize,
    },
    /// A NaN appeared in an intermediate value under [`FloatPolicy::FiniteOutputs`].
    NanResult {
        section: SectionKind,
        node: ValueId,
        element: usize,
    },
    NonFiniteResult {
        section: SectionKind,
        node: ValueId,
        element: usize,
    },
    /// An observable output was not fully finite under
    /// [`FloatPolicy::FiniteOutputs`].
    NonFiniteOutput {
        output: usize,
        element: usize,
    },
    MissingRegister {
        section: SectionKind,
        node: ValueId,
        source: ValueId,
    },
    /// Defensive failure if a verifier-approved root was not materialized.
    /// This should be unreachable, but remains structured rather than a panic.
    MissingBinding {
        section: SectionKind,
        binding: ValueId,
    },
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::Verification(error) =>
            {
                write!(formatter, "program verification failed: {error}")
            },
            Self::InputArity { expected, found } => write!(
                formatter,
                "program expects {expected} inputs, found {found}"
            ),
            Self::ItemArity { expected, found } => write!(
                formatter,
                "program expects {expected} item tensors, found {found}"
            ),
            Self::ItemCountOverflow {
                steps,
                items_per_step,
            } => write!(
                formatter,
                "item count overflows usize: {steps} steps × {items_per_step} items"
            ),
            Self::InputTypeMismatch {
                input,
                expected,
                found,
            } => write!(
                formatter,
                "input {input}: expected type {expected:?}, found {found:?}"
            ),
            Self::ItemTypeMismatch {
                item,
                expected,
                found,
            } => write!(
                formatter,
                "item {item}: expected type {expected:?}, found {found:?}"
            ),
            Self::InvalidInputLayout { input, error } =>
            {
                write!(
                    formatter,
                    "input {input} has invalid tensor layout: {error}"
                )
            },
            Self::InvalidItemLayout { item, error } =>
            {
                write!(formatter, "item {item} has invalid tensor layout: {error}")
            },
            Self::NonFiniteInput { input, element } => write!(
                formatter,
                "input {input} contains a non-finite value at element {element}"
            ),
            Self::NonFiniteItem { item, element } => write!(
                formatter,
                "item {item} contains a non-finite value at element {element}"
            ),
            Self::NanResult {
                section,
                node,
                element,
            } => write!(
                formatter,
                "{section} node {node} produced NaN at element {element}"
            ),
            Self::NonFiniteResult {
                section,
                node,
                element,
            } => write!(
                formatter,
                "{section} node {node} produced a non-finite value at element {element}"
            ),
            Self::NonFiniteOutput { output, element } => write!(
                formatter,
                "output {output} contains a non-finite value at element {element}"
            ),
            Self::MissingRegister {
                section,
                node,
                source,
            } => write!(
                formatter,
                "{section} node {node} requires unavailable local value {source}"
            ),
            Self::MissingBinding { section, binding } => write!(
                formatter,
                "{section} root binding {binding} was not materialized"
            ),
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<ProgramError> for ExecutionError {
    fn from(error: ProgramError) -> Self {
        Self::Verification(error)
    }
}

/// Verify and execute a research program.
///
/// `items` supplies the whole stream: exactly `steps * items.len()` tensors in
/// step-major order (step 0 items first, then step 1, ...).
#[allow(clippy::too_many_lines)]
pub fn execute_program(
    program: &ResearchProgram,
    inputs: &[ValueTensor],
    items: &[ValueTensor],
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> Result<ExecutionResult, ExecutionError> {
    let verified = verify_program(program, limits)?;
    validate_externals(program, inputs, items, policy)?;

    let mut executed_nodes: u64 = 0;

    // ---- init -----------------------------------------------------------------
    let mut state: Vec<ValueTensor> = Vec::with_capacity(program.state.len());
    if !program.state.is_empty()
    {
        let (registers, executed) = eval_section(
            &program.init,
            SectionKind::Init,
            &verified.init_active,
            &verified.init_types,
            &SectionContext {
                inputs,
                items: &[],
                item_offset: 0,
                state: &[],
            },
            policy,
        )?;
        executed_nodes += executed;
        for &binding in &program.init_state
        {
            let value = registers
                .get(binding)
                .and_then(Option::as_ref)
                .ok_or(ExecutionError::MissingBinding {
                    section: SectionKind::Init,
                    binding,
                })?
                .clone();
            state.push(value);
        }
    }

    // ---- scan -------------------------------------------------------------------
    let items_per_step = program.items.len();
    for step_index in 0..program.steps
    {
        let (registers, executed) = eval_section(
            &program.step,
            SectionKind::Step,
            &verified.step_active,
            &verified.step_types,
            &SectionContext {
                inputs: &[],
                items,
                item_offset: (step_index as usize) * items_per_step,
                state: &state,
            },
            policy,
        )?;
        executed_nodes += executed;

        let mut next_state = Vec::with_capacity(program.next_state.len());
        for &binding in &program.next_state
        {
            let value = registers
                .get(binding)
                .and_then(Option::as_ref)
                .ok_or(ExecutionError::MissingBinding {
                    section: SectionKind::Step,
                    binding,
                })?
                .clone();
            next_state.push(value);
        }
        state = next_state;
    }

    // ---- finalize -----------------------------------------------------------------
    let (registers, executed) = eval_section(
        &program.finalize,
        SectionKind::Finalize,
        &verified.finalize_active,
        &verified.finalize_types,
        &SectionContext {
            inputs,
            items: &[],
            item_offset: 0,
            state: &state,
        },
        policy,
    )?;
    executed_nodes += executed;

    let mut outputs = Vec::with_capacity(program.outputs.len());
    for (index, &output) in program.outputs.iter().enumerate()
    {
        let value = registers
            .get(output)
            .and_then(Option::as_ref)
            .ok_or(ExecutionError::MissingBinding {
                section: SectionKind::Finalize,
                binding: output,
            })?
            .clone();
        if policy.floats != FloatPolicy::AllowNonFinite && value.dtype.is_float()
        {
            if let Some(element) = value.data.iter().position(|&v| !v.is_finite())
            {
                return Err(ExecutionError::NonFiniteOutput {
                    output: index,
                    element,
                });
            }
        }
        outputs.push(value);
    }

    Ok(ExecutionResult {
        outputs,
        executed_nodes,
        verified,
    })
}

/// Validate arity, declared types and finiteness of external inputs/items.
fn validate_externals(
    program: &ResearchProgram,
    inputs: &[ValueTensor],
    items: &[ValueTensor],
    policy: ExecutionPolicy,
) -> Result<(), ExecutionError> {
    if inputs.len() != program.inputs.len()
    {
        return Err(ExecutionError::InputArity {
            expected: program.inputs.len(),
            found: inputs.len(),
        });
    }
    for (index, (tensor, declared)) in inputs.iter().zip(&program.inputs).enumerate()
    {
        tensor
            .validate_layout()
            .map_err(|error| ExecutionError::InvalidInputLayout {
                input: index,
                error,
            })?;
        if &tensor.value_type() != declared
        {
            return Err(ExecutionError::InputTypeMismatch {
                input: index,
                expected: declared.clone(),
                found: tensor.value_type(),
            });
        }
        if policy.floats != FloatPolicy::AllowNonFinite && tensor.dtype.is_float()
        {
            if let Some(element) = tensor.data.iter().position(|&value| !value.is_finite())
            {
                return Err(ExecutionError::NonFiniteInput {
                    input: index,
                    element,
                });
            }
        }
    }

    let expected_items = (program.steps as usize)
        .checked_mul(program.items.len())
        .ok_or(ExecutionError::ItemCountOverflow {
            steps: program.steps,
            items_per_step: program.items.len(),
        })?;
    if items.len() != expected_items
    {
        return Err(ExecutionError::ItemArity {
            expected: expected_items,
            found: items.len(),
        });
    }
    let slots_per_step = program.items.len().max(1);
    for (slot, tensor) in items.iter().enumerate()
    {
        tensor
            .validate_layout()
            .map_err(|error| ExecutionError::InvalidItemLayout { item: slot, error })?;
        let declared = &program.items[slot % slots_per_step];
        if &tensor.value_type() != declared
        {
            return Err(ExecutionError::ItemTypeMismatch {
                item: slot,
                expected: declared.clone(),
                found: tensor.value_type(),
            });
        }
        if policy.floats != FloatPolicy::AllowNonFinite && tensor.dtype.is_float()
        {
            if let Some(element) = tensor.data.iter().position(|&value| !value.is_finite())
            {
                return Err(ExecutionError::NonFiniteItem {
                    item: slot,
                    element,
                });
            }
        }
    }
    Ok(())
}

/// External values visible to one section evaluation.
struct SectionContext<'a> {
    inputs: &'a [ValueTensor],
    items: &'a [ValueTensor],
    item_offset: usize,
    state: &'a [ValueTensor],
}

type Registers = Vec<Option<ValueTensor>>;

/// Evaluate one section over its live nodes; returns registers plus the number
/// of nodes actually evaluated.
fn eval_section(
    section: &Section,
    kind: SectionKind,
    active: &[bool],
    types: &[ValueType],
    context: &SectionContext<'_>,
    policy: ExecutionPolicy,
) -> Result<(Registers, u64), ExecutionError> {
    let mut registers: Registers = Vec::with_capacity(section.len());
    let mut executed: u64 = 0;

    for (node, op) in section.ops.iter().enumerate()
    {
        if !active.get(node).copied().unwrap_or(false)
        {
            registers.push(None);
            continue;
        }

        // Resolve operands in reference order; failures are deterministic and
        // cannot occur for verified programs, but stay structured regardless.
        let mut operands: Vec<&ValueTensor> = Vec::with_capacity(3);
        let mut failure: Option<ExecutionError> = None;
        op.for_each_ref(|reference| {
            if failure.is_some()
            {
                return;
            }
            match resolve_ref(reference, node, kind, &registers, context)
            {
                Ok(value) => operands.push(value),
                Err(error) => failure = Some(error),
            }
        });
        if let Some(error) = failure
        {
            return Err(error);
        }

        let value = eval_op(op, node, kind, types[node].clone(), &operands, policy)?;
        registers.push(Some(value));
        executed += 1;
    }

    Ok((registers, executed))
}

fn resolve_ref<'a>(
    reference: Ref,
    node: ValueId,
    kind: SectionKind,
    registers: &'a Registers,
    context: &'a SectionContext<'a>,
) -> Result<&'a ValueTensor, ExecutionError> {
    match reference
    {
        Ref::Input(index) => context
            .inputs
            .get(index)
            .ok_or(ExecutionError::Verification(
                ProgramError::InputOutOfBounds {
                    section: kind,
                    node,
                    input: index,
                    available: context.inputs.len(),
                },
            )),
        Ref::Local(source) =>
        {
            registers
                .get(source)
                .and_then(Option::as_ref)
                .ok_or(ExecutionError::MissingRegister {
                    section: kind,
                    node,
                    source,
                })
        },
        Ref::Item(index) => context
            .items
            .get(context.item_offset + index)
            .ok_or_else(|| {
                ExecutionError::Verification(ProgramError::ItemOutOfBounds {
                    node,
                    item: index,
                    available: context.items.len().saturating_sub(context.item_offset),
                })
            }),
        Ref::StatePrev(slot) | Ref::StateFinal(slot) => context.state.get(slot).ok_or(
            ExecutionError::Verification(ProgramError::StateSlotOutOfBounds {
                section: kind,
                node,
                slot,
                available: context.state.len(),
            }),
        ),
    }
}

// ---------------------------------------------------------------------------
// Element kernels
// ---------------------------------------------------------------------------

/// Deterministic binary32 minimum (`Min`, `ReduceMin` and `Clamp` kernels).
///
/// Normative rule (mirrored by [`deterministic_min_f64`] and the `*_max_*`
/// siblings), chosen so the result depends only on the operand *values*,
/// never on their order:
///
/// 1. if both operands are NaN, the canonical quiet NaN is returned;
/// 2. if exactly one operand is NaN, the other operand is returned;
/// 3. otherwise the numerically smaller operand is returned; when the operands
///    are numerically equal — which includes the pair `+0`/`-0` — the
///    negative-signed operand wins, so `min(+0,-0) = min(-0,+0) = -0`.
///
/// This matches IEEE-754-2019 `minimumNumber`/`maximumNumber` zero handling
/// and coincides with current native `f32::min` on mainstream targets, but it
/// no longer *depends* on them: Rust does not specify native-min/max signed
/// zero or payload behaviour (rust-lang/rust#99640), so a bit-reproducibility
/// contract must own this rule explicitly.
pub(crate) fn deterministic_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan()
    {
        if a.is_nan() && b.is_nan()
        {
            f32::NAN
        }
        else if a.is_nan()
        {
            b
        }
        else
        {
            a
        }
    }
    else if a < b
    {
        a
    }
    else if b < a
    {
        b
    }
    else if a.is_sign_negative()
    {
        a
    }
    else
    {
        b
    }
}

/// Deterministic binary64 minimum; see [`deterministic_min_f32`].
pub(crate) fn deterministic_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan()
    {
        if a.is_nan() && b.is_nan()
        {
            f64::NAN
        }
        else if a.is_nan()
        {
            b
        }
        else
        {
            a
        }
    }
    else if a < b
    {
        a
    }
    else if b < a
    {
        b
    }
    else if a.is_sign_negative()
    {
        a
    }
    else
    {
        b
    }
}

/// Deterministic binary32 maximum; see [`deterministic_min_f32`] for the
/// normative rule. The tie-break is mirrored: among numerically equal
/// operands the positive-signed operand wins, so
/// `max(+0,-0) = max(-0,+0) = +0`.
pub(crate) fn deterministic_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan()
    {
        if a.is_nan() && b.is_nan()
        {
            f32::NAN
        }
        else if a.is_nan()
        {
            b
        }
        else
        {
            a
        }
    }
    else if a > b
    {
        a
    }
    else if b > a
    {
        b
    }
    else if a.is_sign_positive()
    {
        a
    }
    else
    {
        b
    }
}

/// Deterministic binary64 maximum; see [`deterministic_max_f32`].
pub(crate) fn deterministic_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan()
    {
        if a.is_nan() && b.is_nan()
        {
            f64::NAN
        }
        else if a.is_nan()
        {
            b
        }
        else
        {
            a
        }
    }
    else if a > b
    {
        a
    }
    else if b > a
    {
        b
    }
    else if a.is_sign_positive()
    {
        a
    }
    else
    {
        b
    }
}

/// Checked row-major strides of a shape (empty for rank 0).
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![1usize; shape.len()];
    if shape.len() <= 1
    {
        return result;
    }
    let mut acc = 1usize;
    for axis in (0..shape.len() - 1).rev()
    {
        acc *= shape[axis + 1];
        result[axis] = acc;
    }
    result
}

/// Strides of `shape` viewed against an output rank (left-padded with `0`
/// stride so broadcast dimensions collapse onto element 0).
fn broadcast_strides(shape: &[usize], out_rank: usize) -> Vec<usize> {
    let own = strides(shape);
    let offset = out_rank - shape.len();
    let mut result = vec![0usize; out_rank];
    for (axis, &stride) in own.iter().enumerate()
    {
        result[offset + axis] = if shape[axis] == 1 { 0 } else { stride };
    }
    result
}

/// Offset of a multi-index into a buffer with the given strides.
fn offset_at(strides: &[usize], index: &[usize]) -> usize {
    strides
        .iter()
        .zip(index)
        .map(|(&stride, &coordinate)| stride * coordinate)
        .sum()
}

/// Increment a row-major multi-index in place; returns `false` on overflow
/// past the last index.
fn bump(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..index.len()).rev()
    {
        index[axis] += 1;
        if index[axis] < shape[axis]
        {
            return true;
        }
        index[axis] = 0;
    }
    false
}

/// Apply a native-dtype unary kernel elementwise (shapes are identical).
fn unary_float(
    src: &ValueTensor,
    kernel32: impl Fn(f32) -> f32,
    kernel64: impl Fn(f64) -> f64,
) -> Result<ValueTensor, ExecutionError> {
    let data = match src.dtype
    {
        DType::F32 => src
            .data
            .iter()
            .map(|&a| kernel32(a as f32) as f64)
            .collect::<Vec<_>>(),
        DType::F64 => src.data.iter().copied().map(kernel64).collect::<Vec<_>>(),
        DType::Bool => return Err(unreachable_bool("float unary")),
    };
    Ok(ValueTensor::from_parts(src.dtype, src.shape.clone(), data))
}

/// Apply a native-dtype binary kernel with NumPy-style broadcasting.
fn binary_float(
    lhs: &ValueTensor,
    rhs: &ValueTensor,
    out_type: &ValueType,
    kernel32: impl Fn(f32, f32) -> f32,
    kernel64: impl Fn(f64, f64) -> f64,
) -> Result<ValueTensor, ExecutionError> {
    let out_shape = out_type.shape.clone();
    let out_elements = shape_elements(&out_shape).unwrap_or(usize::MAX);
    let lhs_strides = broadcast_strides(&lhs.shape, out_shape.len());
    let rhs_strides = broadcast_strides(&rhs.shape, out_shape.len());

    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; out_shape.len()];
    for element in data.iter_mut()
    {
        let a = lhs.data[offset_at(&lhs_strides, &index)];
        let b = rhs.data[offset_at(&rhs_strides, &index)];
        *element = match out_type.dtype
        {
            DType::F32 =>
            {
                let (x, y) = (a as f32, b as f32);
                kernel32(x, y) as f64
            },
            DType::F64 => kernel64(a, b),
            DType::Bool => return Err(unreachable_bool("float binary")),
        };
        bump(&mut index, &out_shape);
    }
    Ok(ValueTensor::from_parts(out_type.dtype, out_shape, data))
}

/// Apply a native-dtype ternary kernel with broadcasting on all three inputs.
fn ternary_float(
    a: &ValueTensor,
    b: &ValueTensor,
    c: &ValueTensor,
    out_type: &ValueType,
    kernel32: impl Fn(f32, f32, f32) -> f32,
    kernel64: impl Fn(f64, f64, f64) -> f64,
) -> Result<ValueTensor, ExecutionError> {
    let out_shape = out_type.shape.clone();
    let out_elements = shape_elements(&out_shape).unwrap_or(usize::MAX);
    let strides_a = broadcast_strides(&a.shape, out_shape.len());
    let strides_b = broadcast_strides(&b.shape, out_shape.len());
    let strides_c = broadcast_strides(&c.shape, out_shape.len());

    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; out_shape.len()];
    for element in data.iter_mut()
    {
        let va = a.data[offset_at(&strides_a, &index)];
        let vb = b.data[offset_at(&strides_b, &index)];
        let vc = c.data[offset_at(&strides_c, &index)];
        *element = match out_type.dtype
        {
            DType::F32 =>
            {
                let (x, y, z) = (va as f32, vb as f32, vc as f32);
                kernel32(x, y, z) as f64
            },
            DType::F64 => kernel64(va, vb, vc),
            DType::Bool => return Err(unreachable_bool("float ternary")),
        };
        bump(&mut index, &out_shape);
    }
    Ok(ValueTensor::from_parts(out_type.dtype, out_shape, data))
}

fn unreachable_bool(context: &'static str) -> ExecutionError {
    // The verifier guarantees float kernels only ever see float dtypes.
    ExecutionError::Verification(ProgramError::RefIllegalInSection {
        section: SectionKind::Finalize,
        node: 0,
        reference: context,
    })
}

fn const_tensor(value: ScalarValue) -> ValueTensor {
    match value
    {
        ScalarValue::F32(v) => ValueTensor::scalar_f32(v),
        ScalarValue::F64(v) => ValueTensor::scalar_f64(v),
        ScalarValue::Bool(v) => ValueTensor::scalar_bool(v),
    }
}

/// Broadcast comparison producing Boolean 0/1 elements.
fn compare(
    lhs: &ValueTensor,
    rhs: &ValueTensor,
    out_type: &ValueType,
    predicate: impl Fn(f64, f64) -> bool,
    predicate32: impl Fn(f32, f32) -> bool,
) -> Result<ValueTensor, ExecutionError> {
    let out_shape = out_type.shape.clone();
    let out_elements = shape_elements(&out_shape).unwrap_or(usize::MAX);
    let lhs_strides = broadcast_strides(&lhs.shape, out_shape.len());
    let rhs_strides = broadcast_strides(&rhs.shape, out_shape.len());

    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; out_shape.len()];
    for element in data.iter_mut()
    {
        let a = lhs.data[offset_at(&lhs_strides, &index)];
        let b = rhs.data[offset_at(&rhs_strides, &index)];
        let bit = match lhs.dtype
        {
            DType::F32 => predicate32(a as f32, b as f32),
            _ => predicate(a, b),
        };
        *element = f64::from(u8::from(bit));
        bump(&mut index, &out_shape);
    }
    Ok(ValueTensor::from_parts(DType::Bool, out_shape, data))
}

/// Masked selection; mask broadcasts onto the exact branch shape.
fn select(
    mask: &ValueTensor,
    if_true: &ValueTensor,
    if_false: &ValueTensor,
    out_type: &ValueType,
) -> Result<ValueTensor, ExecutionError> {
    let out_shape = out_type.shape.clone();
    let out_elements = shape_elements(&out_shape).unwrap_or(usize::MAX);
    let mask_strides = broadcast_strides(&mask.shape, out_shape.len());

    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; out_shape.len()];
    for (flat, element) in data.iter_mut().enumerate()
    {
        // Branch shapes equal the output shape exactly (verified), so the
        // selected value sits at the same flat position.
        let taken = mask.data[offset_at(&mask_strides, &index)] != 0.0;
        let source = if taken { if_true } else { if_false };
        *element = source.data[flat];
        bump(&mut index, &out_shape);
    }
    Ok(ValueTensor::from_parts(if_true.dtype, out_shape, data))
}

/// Boolean AND/OR with broadcasting (`or == true` selects OR).
fn logic_and_or(
    lhs: &ValueTensor,
    rhs: &ValueTensor,
    out_type: &ValueType,
    or: bool,
) -> Result<ValueTensor, ExecutionError> {
    let out_shape = out_type.shape.clone();
    let out_elements = shape_elements(&out_shape).unwrap_or(usize::MAX);
    let lhs_strides = broadcast_strides(&lhs.shape, out_shape.len());
    let rhs_strides = broadcast_strides(&rhs.shape, out_shape.len());

    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; out_shape.len()];
    for element in data.iter_mut()
    {
        let a = lhs.data[offset_at(&lhs_strides, &index)] != 0.0;
        let b = rhs.data[offset_at(&rhs_strides, &index)] != 0.0;
        *element = f64::from(u8::from(if or { a || b } else { a && b }));
        bump(&mut index, &out_shape);
    }
    Ok(ValueTensor::from_parts(DType::Bool, out_shape, data))
}

fn logic_not(src: &ValueTensor) -> Result<ValueTensor, ExecutionError> {
    let data = src
        .data
        .iter()
        .map(|&bit| f64::from(u8::from(bit == 0.0)))
        .collect();
    Ok(ValueTensor::from_parts(
        DType::Bool,
        src.shape.clone(),
        data,
    ))
}

/// Reduction modes with fixed ascending-flat-index accumulation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Accumulation {
    Sum,
    Product,
    Maximum,
    Minimum,
    Mean,
}

#[allow(clippy::too_many_lines)]
fn reduce_op(
    src: &ValueTensor,
    axis: Option<usize>,
    mode: Accumulation,
) -> Result<ValueTensor, ExecutionError> {
    let out_shape: Vec<usize> = match axis
    {
        None => Vec::new(),
        Some(axis) =>
        {
            let mut shape = src.shape.clone();
            shape.remove(axis);
            shape
        },
    };
    let reduced_count = match axis
    {
        None => src.elements(),
        Some(axis) => src.shape[axis],
    };

    let out_elements = shape_elements(&out_shape).unwrap_or(usize::MAX);
    // Identity elements; max/min identities are unreachable for empty
    // reductions because the verifier rejects those statically. Extrema fold
    // through the deterministic kernels, so the accumulated result is
    // independent of element encounter order (including opposite-signed
    // zeros); NaN elements defer to numeric ones, so an all-NaN reduction
    // keeps its ±Infinity identity.
    let identity = match mode
    {
        Accumulation::Sum | Accumulation::Mean => 0.0,
        Accumulation::Product => 1.0,
        Accumulation::Maximum => f64::NEG_INFINITY,
        Accumulation::Minimum => f64::INFINITY,
    };
    let mut accumulator = vec![identity; out_elements];

    let out_strides = strides(&out_shape);
    let src_rank = src.shape.len();

    for (flat, &value) in src.data.iter().enumerate()
    {
        // Decompose the flat index into the source multi-index.
        let mut remainder = flat;
        // Walk source axes high→low building the kept-axis coordinates.
        let mut coordinates = vec![0usize; src_rank];
        for (axis, coordinate) in coordinates.iter_mut().enumerate().rev()
        {
            let dimension = src.shape[axis].max(1);
            *coordinate = remainder % dimension;
            remainder /= dimension;
        }
        let slot = match axis
        {
            None => 0,
            Some(reduced_axis) =>
            {
                let kept: Vec<usize> = coordinates
                    .iter()
                    .enumerate()
                    .filter(|&(axis, _)| axis != reduced_axis)
                    .map(|(_, coordinate)| *coordinate)
                    .collect();
                offset_at(&out_strides, &kept)
            },
        };

        accumulator[slot] = match (src.dtype, mode)
        {
            (DType::F32, Accumulation::Sum) => ((accumulator[slot] as f32) + (value as f32)) as f64,
            (DType::F32, Accumulation::Product) =>
            {
                ((accumulator[slot] as f32) * (value as f32)) as f64
            },
            (DType::F32, Accumulation::Maximum) =>
            {
                let a = accumulator[slot] as f32;
                let b = value as f32;
                deterministic_max_f32(a, b) as f64
            },
            (DType::F32, Accumulation::Minimum) =>
            {
                let a = accumulator[slot] as f32;
                let b = value as f32;
                deterministic_min_f32(a, b) as f64
            },
            (DType::F32, Accumulation::Mean) =>
            {
                ((accumulator[slot] as f32) + (value as f32)) as f64
            },
            (_, Accumulation::Sum) => accumulator[slot] + value,
            (_, Accumulation::Product) => accumulator[slot] * value,
            (_, Accumulation::Maximum) => deterministic_max_f64(accumulator[slot], value),
            (_, Accumulation::Minimum) => deterministic_min_f64(accumulator[slot], value),
            (_, Accumulation::Mean) => accumulator[slot] + value,
        };
    }

    if mode == Accumulation::Mean
    {
        let count = reduced_count as f64;
        for value in accumulator.iter_mut()
        {
            match src.dtype
            {
                DType::F32 => *value = ((*value as f32) / (count as f32)) as f64,
                _ => *value /= count,
            }
        }
    }

    Ok(ValueTensor::from_parts(src.dtype, out_shape, accumulator))
}

/// Native-dtype inner product of two equal-length vectors.
fn dot(lhs: &ValueTensor, rhs: &ValueTensor) -> Result<ValueTensor, ExecutionError> {
    let mut acc = 0.0f64;
    match lhs.dtype
    {
        DType::F32 =>
        {
            let mut acc32 = 0.0f32;
            for index in 0..lhs.elements()
            {
                acc32 += (lhs.data[index] as f32) * (rhs.data[index] as f32);
            }
            acc = acc32 as f64;
        },
        _ =>
        {
            for index in 0..lhs.elements()
            {
                acc += lhs.data[index] * rhs.data[index];
            }
        },
    }
    let dtype = lhs.dtype;
    Ok(ValueTensor::from_parts(dtype, Vec::new(), vec![acc]))
}

/// Matrix-like products sharing one accumulation discipline (ascending inner
/// index).
#[derive(Debug, Clone, Copy)]
enum MatKind {
    MatVec,
    VecMat,
    MatMul,
}

fn mat_like(
    lhs: &ValueTensor,
    rhs: &ValueTensor,
    kind: MatKind,
) -> Result<ValueTensor, ExecutionError> {
    let dtype = lhs.dtype;
    let is32 = dtype == DType::F32;

    // Uniform native-dtype accumulation: f32 products accumulate in f32,
    // f64 products in f64 (ascending inner index, fixed order).
    macro_rules! accumulate {
        ($acc:expr, $a:expr, $b:expr) => {
            if is32
            {
                $acc = ((($acc) as f32) + (($a) as f32) * (($b) as f32)) as f64;
            }
            else
            {
                $acc += ($a) * ($b);
            }
        };
    }
    match kind
    {
        MatKind::MatVec =>
        {
            let m = lhs.shape[0];
            let n = lhs.shape[1];
            let mut data = vec![0.0f64; m];
            for (slot, row) in lhs.data.chunks(n).enumerate()
            {
                let mut acc = 0.0f64;
                for (a, b) in row.iter().zip(&rhs.data)
                {
                    accumulate!(acc, *a, *b);
                }
                data[slot] = acc;
            }
            Ok(ValueTensor::from_parts(dtype, vec![m], data))
        },
        MatKind::VecMat =>
        {
            let k = rhs.shape[1];
            let mut data = vec![0.0f64; k];
            for (column, slot) in data.iter_mut().enumerate()
            {
                let mut acc = 0.0f64;
                for (a, b) in lhs.data.iter().zip(rhs.data[column..].iter().step_by(k))
                {
                    accumulate!(acc, *a, *b);
                }
                *slot = acc;
            }
            Ok(ValueTensor::from_parts(dtype, vec![k], data))
        },
        MatKind::MatMul =>
        {
            let m = lhs.shape[0];
            let shared = lhs.shape[1];
            let n = rhs.shape[1];
            let mut data = vec![0.0f64; m * n];
            for (row_index, row) in lhs.data.chunks(shared).enumerate()
            {
                // Walk the RHS column with stride n: contiguous LHS row,
                // strided RHS column, identical accumulation order.
                let row_slots = &mut data[row_index * n..row_index * n + n];
                for (column, slot) in row_slots.iter_mut().enumerate()
                {
                    let mut acc = 0.0f64;
                    for (a, b) in row.iter().zip(rhs.data[column..].iter().step_by(n))
                    {
                        accumulate!(acc, *a, *b);
                    }
                    *slot = acc;
                }
            }
            Ok(ValueTensor::from_parts(dtype, vec![m, n], data))
        },
    }
}

fn batched_mat_mul(lhs: &ValueTensor, rhs: &ValueTensor) -> Result<ValueTensor, ExecutionError> {
    let dtype = lhs.dtype;
    let is32 = dtype == DType::F32;
    let batch = lhs.shape[0];
    let m = lhs.shape[1];
    let shared = lhs.shape[2];
    let n = rhs.shape[2];
    let mut data = vec![0.0f64; batch * m * n];
    for b in 0..batch
    {
        let lhs_base = b * m * shared;
        let rhs_base = b * shared * n;
        let out_base = b * m * n;
        for row in 0..m
        {
            for column in 0..n
            {
                let mut acc = 0.0f64;
                for inner in 0..shared
                {
                    acc = if is32
                    {
                        ((acc as f32)
                            + (lhs.data[lhs_base + row * shared + inner] as f32)
                                * (rhs.data[rhs_base + inner * n + column] as f32))
                            as f64
                    }
                    else
                    {
                        acc + lhs.data[lhs_base + row * shared + inner]
                            * rhs.data[rhs_base + inner * n + column]
                    };
                }
                data[out_base + row * n + column] = acc;
            }
        }
    }
    Ok(ValueTensor::from_parts(dtype, vec![batch, m, n], data))
}

fn outer(lhs: &ValueTensor, rhs: &ValueTensor) -> Result<ValueTensor, ExecutionError> {
    let m = lhs.shape[0];
    let n = rhs.shape[0];
    let is32 = lhs.dtype == DType::F32;
    let mut data = vec![0.0f64; m * n];
    for row in 0..m
    {
        for column in 0..n
        {
            let product = if is32
            {
                ((lhs.data[row] as f32) * (rhs.data[column] as f32)) as f64
            }
            else
            {
                lhs.data[row] * rhs.data[column]
            };
            data[row * n + column] = product;
        }
    }
    Ok(ValueTensor::from_parts(lhs.dtype, vec![m, n], data))
}

// ---------------------------------------------------------------------------
// Shape kernels (pure data movement)
// ---------------------------------------------------------------------------

fn reshape_copy(src: &ValueTensor, target_shape: &[usize]) -> ValueTensor {
    // Row-major order is preserved by construction (verified element counts).
    ValueTensor::from_parts(src.dtype, target_shape.to_vec(), src.data.clone())
}

fn transpose(
    src: &ValueTensor,
    perm: &[usize],
    out_shape: &[usize],
) -> Result<ValueTensor, ExecutionError> {
    let src_strides = strides(&src.shape);
    let out_elements = shape_elements(out_shape).unwrap_or(usize::MAX);
    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; out_shape.len()];
    for slot in data.iter_mut()
    {
        // Source coordinate: axis perm[p] takes the output coordinate p.
        let mut source_offset = 0usize;
        for (p, &coordinate) in index.iter().enumerate()
        {
            source_offset += src_strides[perm[p]] * coordinate;
        }
        *slot = src.data[source_offset];
        bump(&mut index, out_shape);
    }
    Ok(ValueTensor::from_parts(src.dtype, out_shape.to_vec(), data))
}

fn broadcast_copy(
    src: &ValueTensor,
    target_shape: &[usize],
) -> Result<ValueTensor, ExecutionError> {
    let out_elements = shape_elements(target_shape).unwrap_or(usize::MAX);
    let src_strides = broadcast_strides(&src.shape, target_shape.len());
    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; target_shape.len()];
    for slot in data.iter_mut()
    {
        *slot = src.data[offset_at(&src_strides, &index)];
        bump(&mut index, target_shape);
    }
    Ok(ValueTensor::from_parts(
        src.dtype,
        target_shape.to_vec(),
        data,
    ))
}

fn concat(
    lhs: &ValueTensor,
    rhs: &ValueTensor,
    axis: usize,
    out_shape: &[usize],
) -> Result<ValueTensor, ExecutionError> {
    let out_elements = shape_elements(out_shape).unwrap_or(usize::MAX);
    let lhs_dim = lhs.shape[axis];
    let lhs_strides = strides(&lhs.shape);
    let rhs_strides = strides(&rhs.shape);

    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; out_shape.len()];
    for slot in data.iter_mut()
    {
        let coordinate = index[axis];
        if coordinate < lhs_dim
        {
            *slot = lhs.data[offset_at(&lhs_strides, &index)];
        }
        else
        {
            let mut rhs_index = index.clone();
            rhs_index[axis] = coordinate - lhs_dim;
            *slot = rhs.data[offset_at(&rhs_strides, &rhs_index)];
        }
        bump(&mut index, out_shape);
    }
    Ok(ValueTensor::from_parts(lhs.dtype, out_shape.to_vec(), data))
}

fn slice_narrow(
    src: &ValueTensor,
    axis: usize,
    start: usize,
    out_shape: &[usize],
) -> Result<ValueTensor, ExecutionError> {
    let out_elements = shape_elements(out_shape).unwrap_or(usize::MAX);
    let src_strides = strides(&src.shape);

    let mut data = vec![0.0f64; out_elements];
    let mut index = vec![0usize; out_shape.len()];
    for slot in data.iter_mut()
    {
        let mut source_index = index.clone();
        source_index[axis] = start + index[axis];
        *slot = src.data[offset_at(&src_strides, &source_index)];
        bump(&mut index, out_shape);
    }
    Ok(ValueTensor::from_parts(src.dtype, out_shape.to_vec(), data))
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn eval_op(
    op: &Op,
    node: ValueId,
    kind: SectionKind,
    result_type: ValueType,
    operands: &[&ValueTensor],
    policy: ExecutionPolicy,
) -> Result<ValueTensor, ExecutionError> {
    // Constants are explicit IEEE bit patterns, not computed results: they
    // bypass the non-finite gate so stable identities (-Infinity running-max
    // initialisers) remain expressible under the default policy.
    if let Op::Const(value) = op
    {
        return Ok(const_tensor(*value));
    }

    let out = match op
    {
        Op::Const(_) => unreachable!("constants are returned before this match"),
        Op::Add(_) => binary_float(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a + b,
            |a, b| a + b,
        )?,
        Op::Sub(_) => binary_float(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a - b,
            |a, b| a - b,
        )?,
        Op::Mul(_) => binary_float(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a * b,
            |a, b| a * b,
        )?,
        Op::Div(_) => binary_float(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a / b,
            |a, b| a / b,
        )?,
        Op::Pow(_) => binary_float(operands[0], operands[1], &result_type, f32::powf, f64::powf)?,

        Op::MulAdd(_) => ternary_float(
            operands[0],
            operands[1],
            operands[2],
            &result_type,
            f32::mul_add,
            f64::mul_add,
        )?,

        Op::Neg(_) => unary_float(operands[0], |a| -a, |a| -a)?,
        Op::Abs(_) => unary_float(operands[0], f32::abs, f64::abs)?,
        Op::Exp(_) => unary_float(operands[0], f32::exp, f64::exp)?,
        Op::Exp2(_) => unary_float(operands[0], f32::exp2, f64::exp2)?,
        Op::Expm1(_) => unary_float(operands[0], f32::exp_m1, f64::exp_m1)?,
        Op::Log(_) => unary_float(operands[0], f32::ln, f64::ln)?,
        Op::Log2(_) => unary_float(operands[0], f32::log2, f64::log2)?,
        Op::Log1p(_) => unary_float(operands[0], f32::ln_1p, f64::ln_1p)?,
        Op::Sqrt(_) => unary_float(operands[0], f32::sqrt, f64::sqrt)?,
        Op::Rsqrt(_) => unary_float(operands[0], |a| 1.0 / a.sqrt(), |a| 1.0 / a.sqrt())?,
        Op::Sin(_) => unary_float(operands[0], f32::sin, f64::sin)?,
        Op::Cos(_) => unary_float(operands[0], f32::cos, f64::cos)?,
        Op::Tanh(_) => unary_float(operands[0], f32::tanh, f64::tanh)?,

        // Extrema use the documented deterministic kernels (never native
        // min/max directly): their signed-zero tie-break must be defined by
        // this contract, not by unspecified platform codegen.
        Op::Min(_) => binary_float(
            operands[0],
            operands[1],
            &result_type,
            deterministic_min_f32,
            deterministic_min_f64,
        )?,
        Op::Max(_) => binary_float(
            operands[0],
            operands[1],
            &result_type,
            deterministic_max_f32,
            deterministic_max_f64,
        )?,

        Op::Clamp(_) => ternary_float(
            operands[0],
            operands[1],
            operands[2],
            &result_type,
            |x, lo, hi| deterministic_max_f32(deterministic_min_f32(x, hi), lo),
            |x, lo, hi| deterministic_max_f64(deterministic_min_f64(x, hi), lo),
        )?,

        Op::Select(_) => select(operands[0], operands[1], operands[2], &result_type)?,

        Op::Eq(_) => compare(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a == b,
            |a, b| a == b,
        )?,
        Op::Ne(_) => compare(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a != b,
            |a, b| a != b,
        )?,
        Op::Lt(_) => compare(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a < b,
            |a, b| a < b,
        )?,
        Op::Le(_) => compare(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a <= b,
            |a, b| a <= b,
        )?,
        Op::Gt(_) => compare(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a > b,
            |a, b| a > b,
        )?,
        Op::Ge(_) => compare(
            operands[0],
            operands[1],
            &result_type,
            |a, b| a >= b,
            |a, b| a >= b,
        )?,

        Op::And(_) => logic_and_or(operands[0], operands[1], &result_type, false)?,
        Op::Or(_) => logic_and_or(operands[0], operands[1], &result_type, true)?,
        Op::Not(_) => logic_not(operands[0])?,

        Op::ReduceSum(reduce) => reduce_op(operands[0], reduce.axis, Accumulation::Sum)?,
        Op::ReduceProd(reduce) => reduce_op(operands[0], reduce.axis, Accumulation::Product)?,
        Op::ReduceMax(reduce) => reduce_op(operands[0], reduce.axis, Accumulation::Maximum)?,
        Op::ReduceMin(reduce) => reduce_op(operands[0], reduce.axis, Accumulation::Minimum)?,
        Op::ReduceMean(reduce) => reduce_op(operands[0], reduce.axis, Accumulation::Mean)?,

        Op::Dot(_) => dot(operands[0], operands[1])?,
        Op::MatVec(_) => mat_like(operands[0], operands[1], MatKind::MatVec)?,
        Op::VecMat(_) => mat_like(operands[0], operands[1], MatKind::VecMat)?,
        Op::MatMul(_) => mat_like(operands[0], operands[1], MatKind::MatMul)?,
        Op::BatchedMatMul(_) => batched_mat_mul(operands[0], operands[1])?,
        Op::Outer(_) => outer(operands[0], operands[1])?,

        Op::Reshape(to) => reshape_copy(operands[0], &to.shape),
        Op::Squeeze(_) | Op::Unsqueeze(_) => reshape_copy(operands[0], &result_type.shape),
        Op::Transpose(permute) => transpose(operands[0], &permute.perm, &result_type.shape)?,
        Op::BroadcastTo(to) => broadcast_copy(operands[0], &to.shape)?,
        Op::Concat { axis, .. } => concat(operands[0], operands[1], *axis, &result_type.shape)?,
        Op::Narrow(window) =>
        {
            slice_narrow(operands[0], window.axis, window.start, &result_type.shape)?
        },
    };

    gate_non_finite(out, kind, node, policy)
}

/// Default-regime gate applied after every evaluated node: NaN aborts
/// immediately; infinities keep flowing (see the [`FloatPolicy`] contract).
fn gate_non_finite(
    tensor: ValueTensor,
    kind: SectionKind,
    node: ValueId,
    policy: ExecutionPolicy,
) -> Result<ValueTensor, ExecutionError> {
    if tensor.dtype.is_float()
    {
        for (element, &value) in tensor.data.iter().enumerate()
        {
            if policy.floats == FloatPolicy::RejectNonFinite && !value.is_finite()
            {
                return Err(ExecutionError::NonFiniteResult {
                    section: kind,
                    node,
                    element,
                });
            }
            if policy.floats == FloatPolicy::FiniteOutputs && value.is_nan()
            {
                return Err(ExecutionError::NanResult {
                    section: kind,
                    node,
                    element,
                });
            }
        }
    }
    Ok(tensor)
}
