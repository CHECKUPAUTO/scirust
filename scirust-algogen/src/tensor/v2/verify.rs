//! Static verification and type inference for V2 research programs.
//!
//! The verifier is the trust boundary between candidate generation and
//! everything else: nothing reaches interpretation, cost modeling, ranking or
//! archival without passing it. It validates, per section: reference legality,
//! use-before-definition, dtype/shape compatibility under the declared
//! broadcasting rules, reduction axes, shape-op legality, resource budgets and
//! illegal non-finite constants; then cross-section recurrence-signature and
//! output rules.
//!
//! Every rule has a distinct structured error variant and a negative test.

use serde::{Deserialize, Serialize};

use super::ir::{Op, Ref, ResearchProgram, Section, ValueId};
use super::types::{
    DType, ShapeError, ValueType, broadcast_shapes, can_broadcast_to, row_major_strides,
    shape_elements,
};

/// Resource limits enforced before a program may be executed or archived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationLimits {
    /// Maximum number of outer inputs.
    pub max_inputs: usize,
    /// Maximum number of item values supplied at each scan step.
    pub max_items_per_step: usize,
    /// Maximum nodes in any one section.
    pub max_nodes_per_section: usize,
    /// Maximum total defined values across all sections.
    pub max_nodes_total: usize,
    /// Maximum tensor rank.
    pub max_rank: usize,
    /// Maximum element count of any single value.
    pub max_elements_per_tensor: usize,
    /// Maximum summed element count over every defined value (each value
    /// counted once; scan temporaries are per-step transients).
    pub max_total_register_elements: usize,
    /// Maximum summed element count across input/item/state signatures.
    pub max_signature_elements: usize,
    /// Maximum materialized step-item elements (`steps × items-per-step`).
    pub max_stream_input_elements: usize,
    /// Conservative host-allocation budget. The reference interpreter stores
    /// every element in an eight-byte carrier, irrespective of logical dtype.
    pub max_temporary_bytes: u64,
    /// Maximum static trip count of the scan.
    pub max_steps: u32,
    /// Maximum number of state components.
    pub max_state_components: usize,
    /// Maximum number of observable outputs.
    pub max_outputs: usize,
}

impl Default for VerificationLimits {
    fn default() -> Self {
        Self {
            max_inputs: 32,
            max_items_per_step: 16,
            max_nodes_per_section: 256,
            max_nodes_total: 1024,
            max_rank: 8,
            max_elements_per_tensor: 16 * 1024 * 1024,
            max_total_register_elements: 64 * 1024 * 1024,
            max_signature_elements: 64 * 1024 * 1024,
            max_stream_input_elements: 64 * 1024 * 1024,
            max_temporary_bytes: 512 * 1024 * 1024,
            max_steps: 4096,
            max_state_components: 8,
            max_outputs: 8,
        }
    }
}

/// Which program section an error refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SectionKind {
    Init,
    Step,
    Finalize,
}

/// Which declared signature contains an invalid value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureKind {
    Input,
    Item,
    State,
}

impl std::fmt::Display for SignatureKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self
        {
            Self::Input => "input",
            Self::Item => "item",
            Self::State => "state",
        })
    }
}

impl SectionKind {
    /// Stable byte tag for canonical encodings.
    pub const fn tag(self) -> u8 {
        match self
        {
            Self::Init => 0,
            Self::Step => 1,
            Self::Finalize => 2,
        }
    }

    /// Decode a canonical byte tag.
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag
        {
            0 => Some(Self::Init),
            1 => Some(Self::Step),
            2 => Some(Self::Finalize),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self
        {
            Self::Init => "init",
            Self::Step => "step",
            Self::Finalize => "finalize",
        }
    }
}

impl std::fmt::Display for SectionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// A deterministic verification failure. Every variant identifies the
/// offending section/node so diagnostics stay actionable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramError {
    TooManyInputs {
        inputs: usize,
        maximum: usize,
    },
    TooManyItemsPerStep {
        items: usize,
        maximum: usize,
    },
    TooManyNodesInSection {
        section: SectionKind,
        nodes: usize,
        maximum: usize,
    },
    TooManyNodesTotal {
        nodes: usize,
        maximum: usize,
    },
    StepsLimitExceeded {
        steps: u32,
        maximum: u32,
    },
    TooManyStateComponents {
        components: usize,
        maximum: usize,
    },
    TooManyOutputs {
        outputs: usize,
        maximum: usize,
    },
    NoOutputs,

    StepsWithoutState,
    StateWithoutSteps {
        components: usize,
    },
    RecurrenceSectionsWithoutState {
        init_nodes: usize,
        step_nodes: usize,
    },

    InitStateCountMismatch {
        expected: usize,
        found: usize,
    },
    NextStateCountMismatch {
        expected: usize,
        found: usize,
    },
    InitStateValueOutOfBounds {
        value: ValueId,
        produced: usize,
    },
    NextStateValueOutOfBounds {
        value: ValueId,
        produced: usize,
    },
    OutputValueOutOfBounds {
        value: ValueId,
        produced: usize,
    },
    OutputDuplicate {
        output: ValueId,
    },

    RefIllegalInSection {
        section: SectionKind,
        node: ValueId,
        reference: &'static str,
    },
    InputOutOfBounds {
        section: SectionKind,
        node: ValueId,
        input: usize,
        available: usize,
    },
    ItemOutOfBounds {
        node: ValueId,
        item: usize,
        available: usize,
    },
    StateSlotOutOfBounds {
        section: SectionKind,
        node: ValueId,
        slot: usize,
        available: usize,
    },
    NonCausalDependency {
        section: SectionKind,
        node: ValueId,
        source: ValueId,
    },

    DTypeMismatch {
        section: SectionKind,
        node: ValueId,
        op: &'static str,
        expected: DType,
        found: DType,
    },
    ShapeMismatchExact {
        section: SectionKind,
        node: ValueId,
        op: &'static str,
        left: Vec<usize>,
        right: Vec<usize>,
    },
    BroadcastIncompatible {
        section: SectionKind,
        node: ValueId,
        left: Vec<usize>,
        right: Vec<usize>,
    },
    MaskNotBroadcastable {
        section: SectionKind,
        node: ValueId,
        mask: Vec<usize>,
        value: Vec<usize>,
    },
    ReductionAxisOutOfRange {
        section: SectionKind,
        node: ValueId,
        axis: usize,
        rank: usize,
    },
    ReductionOverEmptyForbidden {
        section: SectionKind,
        node: ValueId,
        reason: &'static str,
    },
    ReshapeElementMismatch {
        section: SectionKind,
        node: ValueId,
        source_elements: u64,
        target_elements: u64,
    },
    SqueezeAxisNotOne {
        section: SectionKind,
        node: ValueId,
        axis: usize,
        dimension: usize,
    },
    UnsqueezeAxisOutOfRange {
        section: SectionKind,
        node: ValueId,
        axis: usize,
        rank: usize,
    },
    TransposePermutationInvalid {
        section: SectionKind,
        node: ValueId,
        perm: Vec<usize>,
        rank: usize,
    },
    ConcatAxisShapeMismatch {
        section: SectionKind,
        node: ValueId,
        axis: usize,
        left: Vec<usize>,
        right: Vec<usize>,
    },
    NarrowRangeInvalid {
        section: SectionKind,
        node: ValueId,
        axis: usize,
        start: usize,
        len: usize,
        dimension: usize,
    },

    RankLimitExceeded {
        section: SectionKind,
        node: ValueId,
        rank: usize,
        maximum: usize,
    },
    TensorTooLarge {
        section: SectionKind,
        node: ValueId,
        elements: usize,
        maximum: usize,
    },
    TotalRegisterElementsExceeded {
        elements: usize,
        maximum: usize,
    },
    SignatureRankLimitExceeded {
        kind: SignatureKind,
        index: usize,
        rank: usize,
        maximum: usize,
    },
    SignatureTensorTooLarge {
        kind: SignatureKind,
        index: usize,
        elements: usize,
        maximum: usize,
    },
    SignatureElementsExceeded {
        elements: usize,
        maximum: usize,
    },
    StreamInputElementsExceeded {
        elements: usize,
        maximum: usize,
    },
    SignatureStrideOverflow {
        kind: SignatureKind,
        index: usize,
        shape: Vec<usize>,
    },
    StrideOverflow {
        section: SectionKind,
        node: ValueId,
        shape: Vec<usize>,
    },
    TemporaryBytesExceeded {
        bytes: u64,
        maximum: u64,
    },
    DimensionOverflow {
        section: SectionKind,
        node: ValueId,
        op: &'static str,
    },

    /// A constant is NaN. `±Infinity` constants are admissible (stable
    /// identities); NaN constants are not.
    NonNanConstant {
        section: SectionKind,
        node: ValueId,
    },
    NonFiniteConstantInFiniteOnly {
        section: SectionKind,
        node: ValueId,
    },
}

impl std::fmt::Display for ProgramError {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::TooManyInputs { inputs, maximum } => write!(
                formatter,
                "program declares {inputs} inputs, exceeding limit {maximum}"
            ),
            Self::TooManyItemsPerStep { items, maximum } => write!(
                formatter,
                "program declares {items} items per step, exceeding limit {maximum}"
            ),
            Self::TooManyNodesInSection {
                section,
                nodes,
                maximum,
            } => write!(
                formatter,
                "{section} section defines {nodes} nodes, exceeding limit {maximum}"
            ),
            Self::TooManyNodesTotal { nodes, maximum } => write!(
                formatter,
                "program defines {nodes} nodes in total, exceeding limit {maximum}"
            ),
            Self::StepsLimitExceeded { steps, maximum } => write!(
                formatter,
                "scan declares {steps} steps, exceeding limit {maximum}"
            ),
            Self::TooManyStateComponents {
                components,
                maximum,
            } => write!(
                formatter,
                "recurrence declares {components} state components, exceeding limit {maximum}"
            ),
            Self::TooManyOutputs { outputs, maximum } => write!(
                formatter,
                "program declares {outputs} outputs, exceeding limit {maximum}"
            ),
            Self::NoOutputs => write!(formatter, "program declares no outputs"),
            Self::StepsWithoutState =>
            {
                write!(formatter, "scan declares steps but no state components")
            },
            Self::StateWithoutSteps { components } => write!(
                formatter,
                "recurrence declares {components} state components but zero steps"
            ),
            Self::RecurrenceSectionsWithoutState {
                init_nodes,
                step_nodes,
            } => write!(
                formatter,
                "program without recurrence declares {init_nodes} init and {step_nodes} step nodes"
            ),
            Self::InitStateCountMismatch { expected, found } => write!(
                formatter,
                "init produces {found} state values but {expected} are declared"
            ),
            Self::NextStateCountMismatch { expected, found } => write!(
                formatter,
                "step produces {found} next-state values but {expected} are declared"
            ),
            Self::InitStateValueOutOfBounds { value, produced } => write!(
                formatter,
                "init state binding references value {value}, but init produces {produced}"
            ),
            Self::NextStateValueOutOfBounds { value, produced } => write!(
                formatter,
                "next-state binding references value {value}, but step produces {produced}"
            ),
            Self::OutputValueOutOfBounds { value, produced } => write!(
                formatter,
                "output references finalize value {value}, but finalize produces {produced}"
            ),
            Self::OutputDuplicate { output } =>
            {
                write!(
                    formatter,
                    "finalize value {output} is declared as output twice"
                )
            },
            Self::RefIllegalInSection {
                section,
                node,
                reference,
            } => write!(
                formatter,
                "reference {reference} used by {section} node {node} is illegal in that section"
            ),
            Self::InputOutOfBounds {
                section,
                node,
                input,
                available,
            } => write!(
                formatter,
                "{section} node {node} requests input {input}, but only {available} exist"
            ),
            Self::ItemOutOfBounds {
                node,
                item,
                available,
            } => write!(
                formatter,
                "step node {node} requests item {item}, but only {available} exist per step"
            ),
            Self::StateSlotOutOfBounds {
                section,
                node,
                slot,
                available,
            } => write!(
                formatter,
                "{section} node {node} requests state slot {slot}, but only {available} exist"
            ),
            Self::NonCausalDependency {
                section,
                node,
                source,
            } => write!(
                formatter,
                "{section} node {node} reads local value {source}, which is not strictly earlier"
            ),
            Self::DTypeMismatch {
                section,
                node,
                op,
                expected,
                found,
            } => write!(
                formatter,
                "{section} node {node} ({op}) requires dtype {}, found {}",
                expected.name(),
                found.name()
            ),
            Self::ShapeMismatchExact {
                section,
                node,
                op,
                left,
                right,
            } => write!(
                formatter,
                "{section} node {node} ({op}) requires exactly equal shapes {left:?} vs {right:?}"
            ),
            Self::BroadcastIncompatible {
                section,
                node,
                left,
                right,
            } => write!(
                formatter,
                "{section} node {node}: shapes {left:?} and {right:?} cannot broadcast"
            ),
            Self::MaskNotBroadcastable {
                section,
                node,
                mask,
                value,
            } => write!(
                formatter,
                "{section} node {node}: mask shape {mask:?} cannot broadcast onto {value:?}"
            ),
            Self::ReductionAxisOutOfRange {
                section,
                node,
                axis,
                rank,
            } => write!(
                formatter,
                "{section} reduction node {node}: axis {axis} out of range for rank {rank}"
            ),
            Self::ReductionOverEmptyForbidden {
                section,
                node,
                reason,
            } => write!(formatter, "{section} reduction node {node}: {reason}"),
            Self::ReshapeElementMismatch {
                section,
                node,
                source_elements,
                target_elements,
            } => write!(
                formatter,
                "{section} reshape node {node}: {source_elements} source elements do not match {target_elements} target elements"
            ),
            Self::SqueezeAxisNotOne {
                section,
                node,
                axis,
                dimension,
            } => write!(
                formatter,
                "{section} squeeze node {node}: axis {axis} has dimension {dimension}, not 1"
            ),
            Self::UnsqueezeAxisOutOfRange {
                section,
                node,
                axis,
                rank,
            } => write!(
                formatter,
                "{section} unsqueeze node {node}: axis {axis} out of range for rank {rank}"
            ),
            Self::TransposePermutationInvalid {
                section,
                node,
                perm,
                rank,
            } => write!(
                formatter,
                "{section} transpose node {node}: permutation {perm:?} is invalid for rank {rank}"
            ),
            Self::ConcatAxisShapeMismatch {
                section,
                node,
                axis,
                left,
                right,
            } => write!(
                formatter,
                "{section} concat node {node}: operands differ off-axis {axis}: {left:?} vs {right:?}"
            ),
            Self::NarrowRangeInvalid {
                section,
                node,
                axis,
                start,
                len,
                dimension,
            } =>
            {
                // Saturating on purpose: a hostile `start + len` may overflow
                // and this formatter must never panic while reporting it.
                write!(
                    formatter,
                    "{section} narrow node {node}: range [{start}, {}) exceeds axis {axis} dimension {dimension}",
                    start.saturating_add(*len)
                )
            },
            Self::RankLimitExceeded {
                section,
                node,
                rank,
                maximum,
            } => write!(
                formatter,
                "{section} node {node} produces rank {rank}, exceeding limit {maximum}"
            ),
            Self::TensorTooLarge {
                section,
                node,
                elements,
                maximum,
            } => write!(
                formatter,
                "{section} node {node} represents {elements} elements, exceeding limit {maximum}"
            ),
            Self::TotalRegisterElementsExceeded { elements, maximum } => write!(
                formatter,
                "total register elements reach {elements}, exceeding limit {maximum}"
            ),
            Self::SignatureRankLimitExceeded {
                kind,
                index,
                rank,
                maximum,
            } => write!(
                formatter,
                "{kind} signature value {index} has rank {rank}, exceeding limit {maximum}"
            ),
            Self::SignatureTensorTooLarge {
                kind,
                index,
                elements,
                maximum,
            } => write!(
                formatter,
                "{kind} signature value {index} has {elements} elements, exceeding limit {maximum}"
            ),
            Self::SignatureElementsExceeded { elements, maximum } => write!(
                formatter,
                "signature values contain {elements} elements in total, exceeding limit {maximum}"
            ),
            Self::StreamInputElementsExceeded { elements, maximum } => write!(
                formatter,
                "materialized stream input has {elements} elements, exceeding limit {maximum}"
            ),
            Self::SignatureStrideOverflow { kind, index, shape } => write!(
                formatter,
                "{kind} signature value {index} shape {shape:?} has overflowing row-major strides"
            ),
            Self::StrideOverflow {
                section,
                node,
                shape,
            } => write!(
                formatter,
                "{section} node {node} shape {shape:?} has overflowing row-major strides"
            ),
            Self::TemporaryBytesExceeded { bytes, maximum } => write!(
                formatter,
                "conservative interpreter residency is {bytes} bytes, exceeding limit {maximum}"
            ),
            Self::DimensionOverflow { section, node, op } => write!(
                formatter,
                "{section} node {node} ({op}) overflows a static dimension"
            ),
            Self::NonNanConstant { section, node } =>
            {
                write!(formatter, "{section} node {node} uses a NaN constant")
            },
            Self::NonFiniteConstantInFiniteOnly { section, node } => write!(
                formatter,
                "{section} node {node} uses an infinite constant under FiniteOnly semantics"
            ),
        }
    }
}

impl std::error::Error for ProgramError {}

/// Successful static analysis of a research program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedProgram {
    /// Inferred type of every init-section value, by index.
    pub init_types: Vec<ValueType>,
    /// Inferred type of every step-section value, by index.
    pub step_types: Vec<ValueType>,
    /// Inferred type of every finalize-section value, by index.
    pub finalize_types: Vec<ValueType>,
    /// Liveness within `init` (reachable from `init_state`).
    pub init_active: Vec<bool>,
    /// Liveness within `step` (reachable from `next_state`).
    pub step_active: Vec<bool>,
    /// Liveness within `finalize` (reachable from `outputs`).
    pub finalize_active: Vec<bool>,
    /// Inferred types of the observable outputs.
    pub output_types: Vec<ValueType>,
    /// Summed element count over all defined values (each value once).
    pub total_register_elements: usize,
    /// Conservative bytes occupied by all defined values in the reference
    /// interpreter's uniform carrier.
    pub total_register_bytes: u64,
    /// Conservative residency of signatures plus all defined values in the
    /// reference interpreter's uniform carrier.
    pub conservative_resident_bytes: u64,
    /// Summed elements across declared input/item/state signatures.
    pub signature_elements: usize,
    /// Total externally materialized item elements across all scan steps.
    pub stream_input_elements: usize,
    /// Uniform-carrier bytes required by the complete item sequence.
    pub stream_input_bytes: u64,
}

impl VerifiedProgram {
    pub fn active_count(&self) -> usize {
        self.init_active.iter().filter(|&&value| value).count()
            + self.step_active.iter().filter(|&&value| value).count()
            + self.finalize_active.iter().filter(|&&value| value).count()
    }
}

/// Verify `program` against its declared signature and `limits`.
pub fn verify_program(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Result<VerifiedProgram, ProgramError> {
    verify_signature(program, limits)?;
    let declared_signature_elements = signature_elements(program, limits)?;
    let stream_input_elements = stream_input_elements(program, limits)?;
    let signature_bytes = (declared_signature_elements as u64).saturating_mul(8);

    let mut context = Context {
        inputs: &program.inputs,
        items: &program.items,
        state: &program.state,
        limits,
        total_register_elements: 0usize,
        total_register_bytes: 0,
        resident_bytes: signature_bytes,
        semantics: program.semantics,
    };

    let init_types = infer_section(
        &program.init,
        SectionKind::Init,
        &mut context,
        |reference| match *reference
        {
            Ref::Input(index) => Some(SignatureRef::Input(index)),
            Ref::Local(id) => Some(SignatureRef::Local(id)),
            _ => None,
        },
    )?;

    let step_types = infer_section(
        &program.step,
        SectionKind::Step,
        &mut context,
        |reference| match *reference
        {
            Ref::Local(id) => Some(SignatureRef::Local(id)),
            Ref::Item(index) => Some(SignatureRef::Item(index)),
            Ref::StatePrev(slot) => Some(SignatureRef::State(slot)),
            _ => None,
        },
    )?;

    let finalize_types = infer_section(
        &program.finalize,
        SectionKind::Finalize,
        &mut context,
        |reference| match *reference
        {
            Ref::Input(index) => Some(SignatureRef::Input(index)),
            Ref::Local(id) => Some(SignatureRef::Local(id)),
            Ref::StateFinal(slot) => Some(SignatureRef::State(slot)),
            _ => None,
        },
    )?;

    check_state_bindings(program, &init_types, &step_types)?;
    check_outputs(program, &finalize_types, limits)?;

    let output_types: Vec<ValueType> = program
        .outputs
        .iter()
        .map(|&id| finalize_types[id].clone())
        .collect();

    // Root extraction clones state/output values while section registers are
    // still resident. Account explicitly for the larger root tuple.
    let state_copy_elements = program.state.iter().fold(0u64, |sum, value_type| {
        sum.saturating_add(value_type.elements())
    });
    let output_copy_elements = output_types.iter().fold(0u64, |sum, value_type| {
        sum.saturating_add(value_type.elements())
    });
    let root_copy_bytes = state_copy_elements
        .max(output_copy_elements)
        .saturating_mul(8);
    let conservative_resident_bytes = context.resident_bytes.saturating_add(root_copy_bytes);
    if conservative_resident_bytes > limits.max_temporary_bytes
    {
        return Err(ProgramError::TemporaryBytesExceeded {
            bytes: conservative_resident_bytes,
            maximum: limits.max_temporary_bytes,
        });
    }

    Ok(VerifiedProgram {
        init_active: analyze_section_active(&program.init.ops, &program.init_state),
        step_active: analyze_section_active(&program.step.ops, &program.next_state),
        finalize_active: analyze_section_active(&program.finalize.ops, &program.outputs),
        init_types,
        step_types,
        finalize_types,
        output_types,
        total_register_elements: context.total_register_elements,
        total_register_bytes: context.total_register_bytes,
        conservative_resident_bytes,
        signature_elements: declared_signature_elements,
        stream_input_elements,
        stream_input_bytes: (stream_input_elements as u64).saturating_mul(8),
    })
}

fn verify_signature(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Result<(), ProgramError> {
    if program.inputs.len() > limits.max_inputs
    {
        return Err(ProgramError::TooManyInputs {
            inputs: program.inputs.len(),
            maximum: limits.max_inputs,
        });
    }
    if program.items.len() > limits.max_items_per_step
    {
        return Err(ProgramError::TooManyItemsPerStep {
            items: program.items.len(),
            maximum: limits.max_items_per_step,
        });
    }
    let signature = signature_elements(program, limits)?;
    stream_input_elements(program, limits)?;
    let signature_bytes = (signature as u64).saturating_mul(8);
    if signature_bytes > limits.max_temporary_bytes
    {
        return Err(ProgramError::TemporaryBytesExceeded {
            bytes: signature_bytes,
            maximum: limits.max_temporary_bytes,
        });
    }
    if program.node_count() > limits.max_nodes_total
    {
        return Err(ProgramError::TooManyNodesTotal {
            nodes: program.node_count(),
            maximum: limits.max_nodes_total,
        });
    }
    for (kind, section) in [
        (SectionKind::Init, &program.init),
        (SectionKind::Step, &program.step),
        (SectionKind::Finalize, &program.finalize),
    ]
    {
        if section.len() > limits.max_nodes_per_section
        {
            return Err(ProgramError::TooManyNodesInSection {
                section: kind,
                nodes: section.len(),
                maximum: limits.max_nodes_per_section,
            });
        }
    }

    if program.steps > limits.max_steps
    {
        return Err(ProgramError::StepsLimitExceeded {
            steps: program.steps,
            maximum: limits.max_steps,
        });
    }
    if program.state.len() > limits.max_state_components
    {
        return Err(ProgramError::TooManyStateComponents {
            components: program.state.len(),
            maximum: limits.max_state_components,
        });
    }

    // Recurrence co-occurrence rules.
    if program.steps >= 1 && program.state.is_empty()
    {
        return Err(ProgramError::StepsWithoutState);
    }
    if !program.state.is_empty() && program.steps == 0
    {
        return Err(ProgramError::StateWithoutSteps {
            components: program.state.len(),
        });
    }
    if program.state.is_empty() && (!program.init.is_empty() || !program.step.is_empty())
    {
        return Err(ProgramError::RecurrenceSectionsWithoutState {
            init_nodes: program.init.len(),
            step_nodes: program.step.len(),
        });
    }

    if program.init_state.len() != program.state.len()
    {
        return Err(ProgramError::InitStateCountMismatch {
            expected: program.state.len(),
            found: program.init_state.len(),
        });
    }
    if program.next_state.len() != program.state.len()
    {
        return Err(ProgramError::NextStateCountMismatch {
            expected: program.state.len(),
            found: program.next_state.len(),
        });
    }

    Ok(())
}

/// Uniform view of a reference during inference.
enum SignatureRef {
    Input(usize),
    Item(usize),
    Local(ValueId),
    State(usize),
}

struct Context<'a> {
    inputs: &'a [ValueType],
    items: &'a [ValueType],
    state: &'a [ValueType],
    limits: VerificationLimits,
    total_register_elements: usize,
    total_register_bytes: u64,
    resident_bytes: u64,
    semantics: super::semantics::NumericalSemantics,
}

/// Infer types for one section given its legal-reference decoder.
fn infer_section(
    section: &Section,
    kind: SectionKind,
    context: &mut Context<'_>,
    decode: impl Fn(&Ref) -> Option<SignatureRef>,
) -> Result<Vec<ValueType>, ProgramError> {
    let mut types: Vec<ValueType> = Vec::with_capacity(section.len());

    for (node, op) in section.ops.iter().enumerate()
    {
        if context.semantics == super::semantics::NumericalSemantics::FiniteOnly
        {
            let finite = match op
            {
                Op::Const(super::types::ScalarValue::F32(value)) => value.is_finite(),
                Op::Const(super::types::ScalarValue::F64(value)) => value.is_finite(),
                _ => true,
            };
            if !finite
            {
                return Err(ProgramError::NonFiniteConstantInFiniteOnly {
                    section: kind,
                    node,
                });
            }
        }
        // Resolve every operand reference to its type, enforcing section
        // legality, bounds and causality in reference order (deterministic).
        let mut operand_types: Vec<ValueType> = Vec::with_capacity(4);
        let mut failure: Option<ProgramError> = None;
        op.for_each_ref(|reference| {
            if failure.is_some()
            {
                return;
            }
            match decode(&reference)
            {
                Some(resolved) => match lookup(kind, node, &resolved, context, &types)
                {
                    Ok(value_type) => operand_types.push(value_type),
                    Err(error) => failure = Some(error),
                },
                None => failure = Some(illegal_reference(kind, node, &reference)),
            }
        });
        if let Some(error) = failure
        {
            return Err(error);
        }

        let result_type = infer_op(op, kind, node, &operand_types)?;

        let rank = result_type.shape.len();
        if rank > context.limits.max_rank
        {
            return Err(ProgramError::RankLimitExceeded {
                section: kind,
                node,
                rank,
                maximum: context.limits.max_rank,
            });
        }
        if row_major_strides(&result_type.shape).is_none()
        {
            return Err(ProgramError::StrideOverflow {
                section: kind,
                node,
                shape: result_type.shape,
            });
        }
        let elements = shape_elements(&result_type.shape).ok_or(ProgramError::TensorTooLarge {
            section: kind,
            node,
            elements: usize::MAX,
            maximum: context.limits.max_elements_per_tensor,
        })?;
        if elements > context.limits.max_elements_per_tensor
        {
            return Err(ProgramError::TensorTooLarge {
                section: kind,
                node,
                elements,
                maximum: context.limits.max_elements_per_tensor,
            });
        }
        context.total_register_elements = context
            .total_register_elements
            .checked_add(elements)
            .ok_or(ProgramError::TotalRegisterElementsExceeded {
                elements: usize::MAX,
                maximum: context.limits.max_total_register_elements,
            })?;
        if context.total_register_elements > context.limits.max_total_register_elements
        {
            return Err(ProgramError::TotalRegisterElementsExceeded {
                elements: context.total_register_elements,
                maximum: context.limits.max_total_register_elements,
            });
        }

        context.total_register_bytes = context
            .total_register_bytes
            .saturating_add((elements as u64).saturating_mul(8));
        context.resident_bytes = context
            .resident_bytes
            .saturating_add((elements as u64).saturating_mul(8));
        if context.resident_bytes > context.limits.max_temporary_bytes
        {
            return Err(ProgramError::TemporaryBytesExceeded {
                bytes: context.resident_bytes,
                maximum: context.limits.max_temporary_bytes,
            });
        }

        types.push(result_type);
    }

    Ok(types)
}

fn illegal_reference(kind: SectionKind, node: ValueId, reference: &Ref) -> ProgramError {
    let name = match reference
    {
        Ref::Input(_) => "input",
        Ref::Item(_) => "item",
        Ref::Local(_) => "local",
        Ref::StatePrev(_) => "state_prev",
        Ref::StateFinal(_) => "state_final",
    };
    // Out-of-range indices are reported by `lookup`; reaching here means the
    // reference kind itself is illegal in this section.
    ProgramError::RefIllegalInSection {
        section: kind,
        node,
        reference: name,
    }
}

/// Look up a decoded reference, enforcing bounds and causality.
fn lookup(
    kind: SectionKind,
    node: ValueId,
    resolved: &SignatureRef,
    context: &Context<'_>,
    locals: &[ValueType],
) -> Result<ValueType, ProgramError> {
    match resolved
    {
        SignatureRef::Input(index) =>
        {
            context
                .inputs
                .get(*index)
                .cloned()
                .ok_or(ProgramError::InputOutOfBounds {
                    section: kind,
                    node,
                    input: *index,
                    available: context.inputs.len(),
                })
        },
        SignatureRef::Item(index) =>
        {
            context
                .items
                .get(*index)
                .cloned()
                .ok_or(ProgramError::ItemOutOfBounds {
                    node,
                    item: *index,
                    available: context.items.len(),
                })
        },
        SignatureRef::State(slot) =>
        {
            context
                .state
                .get(*slot)
                .cloned()
                .ok_or(ProgramError::StateSlotOutOfBounds {
                    section: kind,
                    node,
                    slot: *slot,
                    available: context.state.len(),
                })
        },
        SignatureRef::Local(id) =>
        {
            if *id >= node
            {
                return Err(ProgramError::NonCausalDependency {
                    section: kind,
                    node,
                    source: *id,
                });
            }
            Ok(locals[*id].clone())
        },
    }
}

fn broadcast_error(kind: SectionKind, node: ValueId, error: ShapeError) -> ProgramError {
    match error
    {
        ShapeError::BroadcastIncompatible { left, right } => ProgramError::BroadcastIncompatible {
            section: kind,
            node,
            left,
            right,
        },
        ShapeError::BroadcastToIncompatible { source, target } =>
        {
            ProgramError::MaskNotBroadcastable {
                section: kind,
                node,
                mask: source,
                value: target,
            }
        },
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn infer_op(
    op: &Op,
    kind: SectionKind,
    node: ValueId,
    operands: &[ValueType],
) -> Result<ValueType, ProgramError> {
    // Helper closures ---------------------------------------------------------
    let at = |index: usize| -> ValueType { operands[index].clone() };
    let require_float = |value: &ValueType, op_name: &'static str| -> Result<DType, ProgramError> {
        if !value.dtype.is_float()
        {
            return Err(ProgramError::DTypeMismatch {
                section: kind,
                node,
                op: op_name,
                expected: DType::F64,
                found: value.dtype,
            });
        }
        Ok(value.dtype)
    };
    let require_bool = |value: &ValueType, op_name: &'static str| -> Result<(), ProgramError> {
        if !value.dtype.is_bool()
        {
            return Err(ProgramError::DTypeMismatch {
                section: kind,
                node,
                op: op_name,
                expected: DType::Bool,
                found: value.dtype,
            });
        }
        Ok(())
    };
    let pair_dtype = |left: &ValueType,
                      right: &ValueType,
                      op_name: &'static str|
     -> Result<DType, ProgramError> {
        let dtype = require_float(left, op_name)?;
        if right.dtype != dtype
        {
            return Err(ProgramError::DTypeMismatch {
                section: kind,
                node,
                op: op_name,
                expected: dtype,
                found: right.dtype,
            });
        }
        Ok(dtype)
    };

    match op
    {
        Op::Const(value) =>
        {
            if !value.is_admissible()
            {
                return Err(ProgramError::NonNanConstant {
                    section: kind,
                    node,
                });
            }
            Ok(ValueType::scalar(value.dtype()))
        },

        // ---- broadcast float arithmetic -------------------------------------
        Op::Add(_)
        | Op::Sub(_)
        | Op::Mul(_)
        | Op::Div(_)
        | Op::Pow(_)
        | Op::Min(_)
        | Op::Max(_) =>
        {
            let left = at(0);
            let right = at(1);
            let dtype = pair_dtype(&left, &right, op.name())?;
            let shape = broadcast_shapes(&left.shape, &right.shape)
                .map_err(|error| broadcast_error(kind, node, error))?;
            Ok(ValueType::new(dtype, shape))
        },

        Op::MulAdd(_) =>
        {
            let a = at(0);
            let b = at(1);
            let c = at(2);
            let dtype = pair_dtype(&a, &b, "mul_add")?;
            if c.dtype != dtype
            {
                return Err(ProgramError::DTypeMismatch {
                    section: kind,
                    node,
                    op: "mul_add",
                    expected: dtype,
                    found: c.dtype,
                });
            }
            let ab = broadcast_shapes(&a.shape, &b.shape)
                .map_err(|error| broadcast_error(kind, node, error))?;
            let shape = broadcast_shapes(&ab, &c.shape)
                .map_err(|error| broadcast_error(kind, node, error))?;
            Ok(ValueType::new(dtype, shape))
        },

        // ---- unary float ------------------------------------------------------
        Op::Neg(_)
        | Op::Abs(_)
        | Op::Exp(_)
        | Op::Exp2(_)
        | Op::Expm1(_)
        | Op::Log(_)
        | Op::Log2(_)
        | Op::Log1p(_)
        | Op::Sqrt(_)
        | Op::Rsqrt(_)
        | Op::Sin(_)
        | Op::Cos(_)
        | Op::Tanh(_) =>
        {
            let src = at(0);
            let dtype = require_float(&src, op.name())?;
            Ok(ValueType::new(dtype, src.shape))
        },

        // ---- clamp --------------------------------------------------------------
        Op::Clamp(_) =>
        {
            let x = at(0);
            let lo = at(1);
            let hi = at(2);
            let dtype = require_float(&x, "clamp")?;
            if lo.dtype != dtype || hi.dtype != dtype
            {
                return Err(ProgramError::DTypeMismatch {
                    section: kind,
                    node,
                    op: "clamp",
                    expected: dtype,
                    found: if lo.dtype != dtype
                    {
                        lo.dtype
                    }
                    else
                    {
                        hi.dtype
                    },
                });
            }
            let xl = broadcast_shapes(&x.shape, &lo.shape)
                .map_err(|error| broadcast_error(kind, node, error))?;
            let shape = broadcast_shapes(&xl, &hi.shape)
                .map_err(|error| broadcast_error(kind, node, error))?;
            Ok(ValueType::new(dtype, shape))
        },

        // ---- select ----------------------------------------------------------------
        Op::Select(_) =>
        {
            let mask = at(0);
            let if_true = at(1);
            let if_false = at(2);
            require_bool(&mask, "select")?;
            if if_true.dtype != if_false.dtype
            {
                return Err(ProgramError::DTypeMismatch {
                    section: kind,
                    node,
                    op: "select",
                    expected: if_true.dtype,
                    found: if_false.dtype,
                });
            }
            if if_true.shape != if_false.shape
            {
                return Err(ProgramError::ShapeMismatchExact {
                    section: kind,
                    node,
                    op: "select",
                    left: if_true.shape,
                    right: if_false.shape,
                });
            }
            if can_broadcast_to(&mask.shape, &if_true.shape).is_err()
            {
                return Err(ProgramError::MaskNotBroadcastable {
                    section: kind,
                    node,
                    mask: mask.shape,
                    value: if_true.shape,
                });
            }
            Ok(if_true)
        },

        // ---- comparisons ------------------------------------------------------------
        Op::Eq(_) | Op::Ne(_) | Op::Lt(_) | Op::Le(_) | Op::Gt(_) | Op::Ge(_) =>
        {
            let left = at(0);
            let right = at(1);
            pair_dtype(&left, &right, op.name())?;
            let shape = broadcast_shapes(&left.shape, &right.shape)
                .map_err(|error| broadcast_error(kind, node, error))?;
            Ok(ValueType::new(DType::Bool, shape))
        },

        // ---- Boolean logic -----------------------------------------------------------
        Op::And(_) | Op::Or(_) =>
        {
            let left = at(0);
            let right = at(1);
            require_bool(&left, op.name())?;
            require_bool(&right, op.name())?;
            let shape = broadcast_shapes(&left.shape, &right.shape)
                .map_err(|error| broadcast_error(kind, node, error))?;
            Ok(ValueType::new(DType::Bool, shape))
        },
        Op::Not(_) =>
        {
            let src = at(0);
            require_bool(&src, "not")?;
            Ok(src)
        },

        // ---- reductions -------------------------------------------------------------
        Op::ReduceSum(reduce)
        | Op::ReduceProd(reduce)
        | Op::ReduceMax(reduce)
        | Op::ReduceMin(reduce)
        | Op::ReduceMean(reduce) =>
        {
            let src = at(0);
            let dtype = require_float(&src, op.name())?;
            let empties_forbidden =
                matches!(op, Op::ReduceMax(_) | Op::ReduceMin(_) | Op::ReduceMean(_));
            match reduce.axis
            {
                None =>
                {
                    if empties_forbidden && src.shape.contains(&0)
                    {
                        return Err(ProgramError::ReductionOverEmptyForbidden {
                            section: kind,
                            node,
                            reason: "max/min/mean over an empty tensor has no finite identity",
                        });
                    }
                    Ok(ValueType::scalar(dtype))
                },
                Some(axis) =>
                {
                    if axis >= src.shape.len()
                    {
                        return Err(ProgramError::ReductionAxisOutOfRange {
                            section: kind,
                            node,
                            axis,
                            rank: src.shape.len(),
                        });
                    }
                    if empties_forbidden && src.shape[axis] == 0
                    {
                        return Err(ProgramError::ReductionOverEmptyForbidden {
                            section: kind,
                            node,
                            reason: "max/min/mean over an empty axis has no finite identity",
                        });
                    }
                    let mut shape = src.shape;
                    shape.remove(axis);
                    Ok(ValueType::new(dtype, shape))
                },
            }
        },

        // ---- linear algebra ----------------------------------------------------------
        Op::Dot(_) =>
        {
            let left = at(0);
            let right = at(1);
            let dtype = pair_dtype(&left, &right, "dot")?;
            if left.shape.len() != 1 || right.shape != left.shape
            {
                return Err(ProgramError::ShapeMismatchExact {
                    section: kind,
                    node,
                    op: "dot",
                    left: left.shape,
                    right: right.shape,
                });
            }
            Ok(ValueType::scalar(dtype))
        },
        Op::MatVec(_) =>
        {
            let matrix = at(0);
            let vector = at(1);
            let dtype = pair_dtype(&matrix, &vector, "mat_vec")?;
            let valid = matrix.shape.len() == 2
                && vector.shape.len() == 1
                && matrix.shape[1] == vector.shape[0];
            if !valid
            {
                return Err(ProgramError::ShapeMismatchExact {
                    section: kind,
                    node,
                    op: "mat_vec",
                    left: matrix.shape,
                    right: vector.shape,
                });
            }
            Ok(ValueType::new(dtype, vec![matrix.shape[0]]))
        },
        Op::VecMat(_) =>
        {
            let vector = at(0);
            let matrix = at(1);
            let dtype = pair_dtype(&vector, &matrix, "vec_mat")?;
            let valid = vector.shape.len() == 1
                && matrix.shape.len() == 2
                && vector.shape[0] == matrix.shape[0];
            if !valid
            {
                return Err(ProgramError::ShapeMismatchExact {
                    section: kind,
                    node,
                    op: "vec_mat",
                    left: vector.shape,
                    right: matrix.shape,
                });
            }
            Ok(ValueType::new(dtype, vec![matrix.shape[1]]))
        },
        Op::MatMul(_) =>
        {
            let left = at(0);
            let right = at(1);
            let dtype = pair_dtype(&left, &right, "mat_mul")?;
            let valid =
                left.shape.len() == 2 && right.shape.len() == 2 && left.shape[1] == right.shape[0];
            if !valid
            {
                return Err(ProgramError::ShapeMismatchExact {
                    section: kind,
                    node,
                    op: "mat_mul",
                    left: left.shape,
                    right: right.shape,
                });
            }
            Ok(ValueType::new(dtype, vec![left.shape[0], right.shape[1]]))
        },
        Op::BatchedMatMul(_) =>
        {
            let left = at(0);
            let right = at(1);
            let dtype = pair_dtype(&left, &right, "batched_mat_mul")?;
            let valid = left.shape.len() == 3
                && right.shape.len() == 3
                && left.shape[0] == right.shape[0]
                && left.shape[2] == right.shape[1];
            if !valid
            {
                return Err(ProgramError::ShapeMismatchExact {
                    section: kind,
                    node,
                    op: "batched_mat_mul",
                    left: left.shape,
                    right: right.shape,
                });
            }
            Ok(ValueType::new(
                dtype,
                vec![left.shape[0], left.shape[1], right.shape[2]],
            ))
        },
        Op::Outer(_) =>
        {
            let left = at(0);
            let right = at(1);
            let dtype = pair_dtype(&left, &right, "outer")?;
            if left.shape.len() != 1 || right.shape.len() != 1
            {
                return Err(ProgramError::ShapeMismatchExact {
                    section: kind,
                    node,
                    op: "outer",
                    left: left.shape,
                    right: right.shape,
                });
            }
            Ok(ValueType::new(dtype, vec![left.shape[0], right.shape[0]]))
        },

        // ---- shape algebra -------------------------------------------------------------
        Op::Reshape(shape_to) =>
        {
            let src = at(0);
            let source_elements = src.elements();
            let target_elements = shape_to
                .shape
                .iter()
                .fold(1u64, |product, &d| product.saturating_mul(d as u64));
            if source_elements != target_elements
            {
                return Err(ProgramError::ReshapeElementMismatch {
                    section: kind,
                    node,
                    source_elements,
                    target_elements,
                });
            }
            Ok(ValueType::new(src.dtype, shape_to.shape.clone()))
        },
        Op::Squeeze(axis_op) =>
        {
            let src = at(0);
            if axis_op.axis >= src.shape.len()
            {
                return Err(ProgramError::SqueezeAxisNotOne {
                    section: kind,
                    node,
                    axis: axis_op.axis,
                    dimension: 0,
                });
            }
            if src.shape[axis_op.axis] != 1
            {
                return Err(ProgramError::SqueezeAxisNotOne {
                    section: kind,
                    node,
                    axis: axis_op.axis,
                    dimension: src.shape[axis_op.axis],
                });
            }
            let mut shape = src.shape;
            shape.remove(axis_op.axis);
            Ok(ValueType::new(src.dtype, shape))
        },
        Op::Unsqueeze(axis_op) =>
        {
            let src = at(0);
            if axis_op.axis > src.shape.len()
            {
                return Err(ProgramError::UnsqueezeAxisOutOfRange {
                    section: kind,
                    node,
                    axis: axis_op.axis,
                    rank: src.shape.len(),
                });
            }
            let mut shape = src.shape;
            shape.insert(axis_op.axis, 1);
            Ok(ValueType::new(src.dtype, shape))
        },
        Op::Transpose(permute) =>
        {
            let src = at(0);
            let rank = src.shape.len();
            let mut seen = vec![false; rank];
            if permute.perm.len() != rank
            {
                return Err(ProgramError::TransposePermutationInvalid {
                    section: kind,
                    node,
                    perm: permute.perm.clone(),
                    rank,
                });
            }
            for &axis in &permute.perm
            {
                if axis >= rank || seen[axis]
                {
                    return Err(ProgramError::TransposePermutationInvalid {
                        section: kind,
                        node,
                        perm: permute.perm.clone(),
                        rank,
                    });
                }
                seen[axis] = true;
            }
            let shape = permute.perm.iter().map(|&axis| src.shape[axis]).collect();
            Ok(ValueType::new(src.dtype, shape))
        },
        Op::BroadcastTo(shape_to) =>
        {
            let src = at(0);
            can_broadcast_to(&src.shape, &shape_to.shape).map_err(|_| {
                ProgramError::MaskNotBroadcastable {
                    section: kind,
                    node,
                    mask: src.shape,
                    value: shape_to.shape.clone(),
                }
            })?;
            Ok(ValueType::new(src.dtype, shape_to.shape.clone()))
        },
        Op::Concat { axis, .. } =>
        {
            let left = at(0);
            let right = at(1);
            if left.dtype != right.dtype
                || left.shape.len() != right.shape.len()
                || *axis >= left.shape.len()
            {
                return Err(ProgramError::ConcatAxisShapeMismatch {
                    section: kind,
                    node,
                    axis: *axis,
                    left: left.shape,
                    right: right.shape,
                });
            }
            for (position, (&left_dim, &right_dim)) in
                left.shape.iter().zip(&right.shape).enumerate()
            {
                if position != *axis && left_dim != right_dim
                {
                    return Err(ProgramError::ConcatAxisShapeMismatch {
                        section: kind,
                        node,
                        axis: *axis,
                        left: left.shape,
                        right: right.shape,
                    });
                }
            }
            let mut shape = left.shape;
            shape[*axis] = shape[*axis].checked_add(right.shape[*axis]).ok_or(
                ProgramError::DimensionOverflow {
                    section: kind,
                    node,
                    op: "concat",
                },
            )?;
            Ok(ValueType::new(left.dtype, shape))
        },
        Op::Narrow(narrow) =>
        {
            let src = at(0);
            let dimension = src.shape.get(narrow.axis).copied().unwrap_or(0);
            let end = narrow.start.saturating_add(narrow.len);
            if narrow.axis >= src.shape.len() || end > dimension
            {
                return Err(ProgramError::NarrowRangeInvalid {
                    section: kind,
                    node,
                    axis: narrow.axis,
                    start: narrow.start,
                    len: narrow.len,
                    dimension,
                });
            }
            let mut shape = src.shape;
            shape[narrow.axis] = narrow.len;
            Ok(ValueType::new(src.dtype, shape))
        },
    }
}

/// Validate all declared signature shapes and return their summed elements.
fn signature_elements(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Result<usize, ProgramError> {
    let mut total = 0usize;
    for (kind, values) in [
        (SignatureKind::Input, program.inputs.as_slice()),
        (SignatureKind::Item, program.items.as_slice()),
        (SignatureKind::State, program.state.as_slice()),
    ]
    {
        for (index, value_type) in values.iter().enumerate()
        {
            if value_type.shape.len() > limits.max_rank
            {
                return Err(ProgramError::SignatureRankLimitExceeded {
                    kind,
                    index,
                    rank: value_type.shape.len(),
                    maximum: limits.max_rank,
                });
            }
            if row_major_strides(&value_type.shape).is_none()
            {
                return Err(ProgramError::SignatureStrideOverflow {
                    kind,
                    index,
                    shape: value_type.shape.clone(),
                });
            }
            let elements =
                shape_elements(&value_type.shape).ok_or(ProgramError::SignatureTensorTooLarge {
                    kind,
                    index,
                    elements: usize::MAX,
                    maximum: limits.max_elements_per_tensor,
                })?;
            if elements > limits.max_elements_per_tensor
            {
                return Err(ProgramError::SignatureTensorTooLarge {
                    kind,
                    index,
                    elements,
                    maximum: limits.max_elements_per_tensor,
                });
            }
            total = total
                .checked_add(elements)
                .ok_or(ProgramError::SignatureElementsExceeded {
                    elements: usize::MAX,
                    maximum: limits.max_signature_elements,
                })?;
            if total > limits.max_signature_elements
            {
                return Err(ProgramError::SignatureElementsExceeded {
                    elements: total,
                    maximum: limits.max_signature_elements,
                });
            }
        }
    }
    Ok(total)
}

/// Checked size of the complete external step-item sequence.
fn stream_input_elements(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Result<usize, ProgramError> {
    let per_step = program.items.iter().try_fold(0usize, |sum, value_type| {
        sum.checked_add(value_type.checked_elements()?)
    });
    let elements = per_step
        .and_then(|per_step| per_step.checked_mul(program.steps as usize))
        .ok_or(ProgramError::StreamInputElementsExceeded {
            elements: usize::MAX,
            maximum: limits.max_stream_input_elements,
        })?;
    if elements > limits.max_stream_input_elements
    {
        return Err(ProgramError::StreamInputElementsExceeded {
            elements,
            maximum: limits.max_stream_input_elements,
        });
    }
    Ok(elements)
}

fn check_state_bindings(
    program: &ResearchProgram,
    init_types: &[ValueType],
    step_types: &[ValueType],
) -> Result<(), ProgramError> {
    for (slot, (&init_id, &next_id)) in program
        .init_state
        .iter()
        .zip(&program.next_state)
        .enumerate()
    {
        let declared = &program.state[slot];
        let produced_init =
            init_types
                .get(init_id)
                .ok_or(ProgramError::InitStateValueOutOfBounds {
                    value: init_id,
                    produced: init_types.len(),
                })?;
        ensure_matches_declared(
            SectionKind::Init,
            init_id,
            "state_init",
            declared,
            produced_init,
        )?;

        let produced_next =
            step_types
                .get(next_id)
                .ok_or(ProgramError::NextStateValueOutOfBounds {
                    value: next_id,
                    produced: step_types.len(),
                })?;
        ensure_matches_declared(
            SectionKind::Step,
            next_id,
            "state_update",
            declared,
            produced_next,
        )?;
    }
    Ok(())
}

fn ensure_matches_declared(
    section: SectionKind,
    node: ValueId,
    role: &'static str,
    declared: &ValueType,
    produced: &ValueType,
) -> Result<(), ProgramError> {
    if declared.dtype != produced.dtype
    {
        return Err(ProgramError::DTypeMismatch {
            section,
            node,
            op: role,
            expected: declared.dtype,
            found: produced.dtype,
        });
    }
    if declared.shape != produced.shape
    {
        return Err(ProgramError::ShapeMismatchExact {
            section,
            node,
            op: role,
            left: declared.shape.clone(),
            right: produced.shape.clone(),
        });
    }
    Ok(())
}

fn check_outputs(
    program: &ResearchProgram,
    finalize_types: &[ValueType],
    limits: VerificationLimits,
) -> Result<(), ProgramError> {
    if program.outputs.is_empty()
    {
        return Err(ProgramError::NoOutputs);
    }
    if program.outputs.len() > limits.max_outputs
    {
        return Err(ProgramError::TooManyOutputs {
            outputs: program.outputs.len(),
            maximum: limits.max_outputs,
        });
    }
    for (position, &output) in program.outputs.iter().enumerate()
    {
        if finalize_types.get(output).is_none()
        {
            return Err(ProgramError::OutputValueOutOfBounds {
                value: output,
                produced: finalize_types.len(),
            });
        }
        if program.outputs[..position].contains(&output)
        {
            return Err(ProgramError::OutputDuplicate { output });
        }
    }
    Ok(())
}

/// Backward liveness inside one section from a set of roots.
///
/// Roots must index into the section (checked earlier); malformed roots yield
/// an all-false map rather than panicking.
pub(crate) fn analyze_section_active(ops: &[Op], roots: &[ValueId]) -> Vec<bool> {
    let mut active = vec![false; ops.len()];
    for &root in roots
    {
        if root < active.len()
        {
            active[root] = true;
        }
    }

    for node in (0..ops.len()).rev()
    {
        if !active[node]
        {
            continue;
        }
        ops[node].for_each_ref(|reference| {
            if let Ref::Local(source) = reference
            {
                if source < node
                {
                    active[source] = true;
                }
            }
        });
    }

    active
}
